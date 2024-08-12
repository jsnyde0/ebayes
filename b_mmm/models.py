from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

import pandas as pd
import csv
import io
import os
import uuid
from typing import List, Optional, Dict

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
    currencies = models.JSONField(default=dict)  # List of predictor currencies

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
            self._data.set_index(self.date_column, inplace=True)
        return self._data
    
    @property
    def index(self) -> pd.Index:
        """Get the index column"""
        return self.data.index
    
    def get_currencies(self) -> Dict[str, str]:
        """Lazy load the currencies for each column."""
        if not self.currencies:
            sales_currency = get_currency(self.data[self.sales_column])
            predictor_currencies = [get_currency(self.data[col]) for col in self.predictor_columns]
            self.currencies = {
                self.sales_column: sales_currency,
                **{col: predictor_currencies[i] for i, col in enumerate(self.predictor_columns)}
            }
        return self.currencies
    
    @property
    def currency(self) -> str:
        """Get the currency used in the CSV file (derived from sales column)."""
        return self.get_currencies()[self.sales_column]
    
    @property
    def predictor_currencies(self) -> List[str]:
        """Get the currencies used in predictor columns."""
        return [self.get_currencies()[col] for col in self.predictor_columns]
    
    def get_predictor_currency(self, col: str) -> str:
        """Get the currency used in the predictor columns."""
        if col not in self.predictor_columns:
            raise ValueError(f"Column {col} is not a predictor column. Predictor columns are {self.predictor_columns}")
        return self.get_currencies()[col]
    
    @property
    def sales(self) -> pd.Series:
        """Get cleaned sales data (without currency symbol and converted to float)."""
        return clean_currency_values(
            self.data[self.sales_column],
            currency_symbols=[self.currency]
        )[0]
    
    @property
    def predictors(self) -> pd.DataFrame:
        """Get cleaned predictor data."""
        return pd.DataFrame({
            col: clean_currency_values(self.data[col], currency_symbols=[self.get_predictor_currency(col)])[0]
            for col in self.predictor_columns
        })
    
    @property
    def predictor_names(self) -> List[str]:
        """Get the names of the predictor columns."""
        return self.predictor_columns
    
    @classmethod
    def create_from_csv(cls, csv_file: 'UploadedFile', user: User) -> 'CSVFile':
        """Process the uploaded CSV file and create a CSVFile instance."""
        try:
            csv_file.seek(0)
            content = csv_file.read().decode('utf-8')
            csv_file.seek(0)
            
            reader = csv.DictReader(io.StringIO(content))
            headers = reader.fieldnames
            
            if not headers or len(headers) < 3:
                raise ValidationError("CSV file must have at least 3 columns: date, sales, and at least one predictor.")
            
            csv_file_instance = cls.objects.create(
                user=user,
                file_name=csv_file.name,
                file=csv_file,
                date_column=headers[0],
                sales_column=headers[1],
                predictor_columns=headers[2:]
            )
            
            return csv_file_instance
        except (csv.Error, ValidationError) as e:
            raise ValidationError(f'CSV processing error: {str(e)}')
        except Exception as e:
            raise ValidationError(f'Unexpected error processing CSV: {str(e)}')



