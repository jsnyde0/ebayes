from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

import pandas as pd
import csv
import io
import os
import uuid
from typing import List, Optional

from .utils import clean_currency_values, get_currency

# Constants
DEFAULT_CURRENCY = '€'
CSV_UPLOAD_DIR = 'csv_files'

def get_file_path(instance: 'CSVFile', filename: str) -> str:
    """Generate a unique file path for uploaded CSV files."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join(CSV_UPLOAD_DIR, filename)

class CSVFile(models.Model):
    """Model to store and process uploaded CSV files in a format handy for MMM."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to=get_file_path)
    
    # columns
    date_column = models.CharField(max_length=255, default='date')
    sales_column = models.CharField(max_length=255, default='sales')
    predictor_columns = models.JSONField(default=list)  # List of predictor column names
    predictor_currencies = models.JSONField(default=list)  # List of predictor currencies
    currency = models.CharField(max_length=3, default=DEFAULT_CURRENCY)

    # _data is for internal use only and can be either a pandas DataFrame or None. Optional is equivalent to Union[pd.DataFrame, None]
    _data: Optional[pd.DataFrame] = None

    class Meta:
        verbose_name_plural = 'CSV Files'
        verbose_name = 'CSV File'
        ordering = ['-created_at']

    def __str__(self) -> str:
        return self.file_name
    
    @property
    def data(self) -> pd.DataFrame:
        """Lazy load the dataset."""
        if self._data is None:
            self._data = pd.read_csv(self.file.path)
        return self._data
    
    def get_index(self) -> pd.Index:
        """Get the index column"""
        return self.data.index

    def get_currency(self) -> str:
        """Get the currency used in the CSV file (derived from sales column)."""
        return self.currency
    
    def get_sales(self) -> pd.Series:
        """Get cleaned sales data (without currency symbol and converted to float)."""
        sales, _ = clean_currency_values(self.data[self.sales_column], currency_symbols=[self.currency])
        return sales
    
    def get_predictors(self) -> List[pd.Series]:
        """Get cleaned predictor data."""
        predictors = []
        for i, predictor_column in enumerate(self.predictor_columns):
            predictor, _ = clean_currency_values(self.data[predictor_column], currency_symbols=[self.currency])
            predictors.append(predictor)
        return predictors
    
    def get_predictor_names(self) -> List[str]:
        """Get the names of the predictor columns."""
        return self.predictor_columns
    
    def get_predictor_currencies(self) -> List[str]:
        """Get the currencies used in predictor columns."""
        return self.predictor_currencies

def process_csv(csv_file: 'UploadedFile', user: User) -> CSVFile:
    """Process the uploaded CSV file and create a CSVFile instance."""
    csv_file_instance = CSVFile.objects.create(
        user=user, 
        file_name=csv_file.name, 
        file=csv_file
    )

    try:
        csv_file.seek(0)  # Ensure we're at the start of the file
        content = csv_file.read().decode('utf-8')
        csv_file.seek(0) 
        
        reader = csv.DictReader(io.StringIO(content))
        headers = reader.fieldnames
        
        if not headers:
            raise ValidationError("No headers found in the CSV file.")
        
        csv_file_instance.date_column = headers[0]
        csv_file_instance.sales_column = headers[1]
        csv_file_instance.predictor_columns = headers[2:]

        # we want to read the sales column and extract the currency
        df = pd.read_csv(csv_file_instance.file.path)
        csv_file_instance.currency = get_currency(df[csv_file_instance.sales_column])

        # get the predictor currencies
        predictor_data = df[csv_file_instance.predictor_columns]
        csv_file_instance.predictor_currencies = [
            get_currency(predictor_data[col]) for col in predictor_data.columns
        ]
        
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



