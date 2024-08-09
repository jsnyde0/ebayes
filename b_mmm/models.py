from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .utils import clean_currency_values, get_currency
import pandas as pd
import uuid
import os
import csv
import io

def get_file_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('csv_files', filename)

class CSVFile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to=get_file_path)
    
    # columns
    date_column = models.CharField(max_length=255, default='date')
    sales_column = models.CharField(max_length=255, default='sales')
    predictor_columns = models.JSONField(default=list)  # List of predictor column names
    currency = models.CharField(max_length=3, default='€')

    class Meta:
        verbose_name_plural = 'CSV Files'
        verbose_name = 'CSV File'
        ordering = ['-created_at']

    def __str__(self):
        return self.file_name
    
    def get_data(self):
        """Load the entire dataset"""
        return pd.read_csv(self.file.path)
    
    def get_index(self):
        """Get the index column"""
        return self.get_data().index
    
    def get_currency(self):
        return self.currency
    
    def get_sales(self):
        sales, _ = clean_currency_values(self.get_data()[self.sales_column], currency_symbols=[self.currency])
        return sales

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
        
        # set the first column as date
        csv_file_instance.date_column = headers[0]
        csv_file_instance.sales_column = headers[1]
        csv_file_instance.predictor_columns = headers[2:]

        # we want to read the sales column and extract the currency
        df = pd.read_csv(csv_file_instance.file.path)
        sales_data = df[csv_file_instance.sales_column]
        csv_file_instance.currency = get_currency(sales_data)
        
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



