from django.core.exceptions import ValidationError
from .models import CSVFile
import csv
import io

def process_csv(csv_file, user):
    csv_file_instance = CSVFile.objects.create(
        user=user, 
        file_name=csv_file.name, 
        file=csv_file
    )

    try:
        # Read the first few lines to get column names and validate
        csv_file.seek(0)  # Ensure we're at the start of the file
        content = csv_file.read().decode('utf-8')
        csv_file.seek(0)  # Reset file pointer
        
        # Use DictReader to handle the CSV
        reader = csv.DictReader(io.StringIO(content))
        
        # Get the fieldnames (headers)
        headers = reader.fieldnames
        
        if not headers:
            raise ValidationError("No headers found in the CSV file.")
        
        csv_file_instance.column_names = headers
        csv_file_instance.save()
        
        # Optionally, we can validate the data here. For example, check if all expected columns are present
        # expected_columns = ['date', 'revenue', 'fb_spend', 'email_clicks', 'search_clicks']
        # if not all(col in headers for col in expected_columns):
        #     raise ValidationError("CSV is missing one or more required columns.")
        
        return csv_file_instance
    except csv.Error as e:
        csv_file_instance.delete()  # Clean up on failure
        raise ValidationError(f'CSV parsing error: {str(e)}')
    except Exception as e:
        csv_file_instance.delete()  # Clean up on failure
        raise ValidationError(f'Error processing CSV: {str(e)}')
    
def clean_euro_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove € symbol and whitespace
        cleaned = value.replace('€', '').strip()
        # Replace comma with dot if comma is used as decimal separator
        if ',' in cleaned and '.' not in cleaned:
            cleaned = cleaned.replace(',', '.')
        # Remove thousands separators
        cleaned = cleaned.replace('.', '').replace(',', '')
        return float(cleaned)
    return 0.0  # Return 0 for any other type