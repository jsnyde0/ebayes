import pandas as pd
import numpy as np
import tempfile
import os
from django.core.files import File

def currency_formatter(x, currency_symbol='€', decimal_places=0, thousands_sep=',', decimal_sep='.'):
    """
    Format a number as a Euro currency string.
    
    Args:
        x (float): The number to format.
        decimal_places (int): Number of decimal places to show. Default is 0.
        thousands_sep (str): The thousands separator. Default is ','.
        decimal_sep (str): The decimal separator. Default is '.'.
    
    Returns:
        str: Formatted Euro currency string.
    """
    try:
        # Convert to float in case it's a string or other type
        value = float(x)
        
        # Format the number
        formatted = f'{value:,.{decimal_places}f}'
        
        # Replace separators if needed
        if thousands_sep != ',':
            formatted = formatted.replace(',', thousands_sep)
        if decimal_sep != '.':
            formatted = formatted.replace('.', decimal_sep)
        
        # Add Euro symbol
        return f'{currency_symbol}{formatted}'
    except ValueError:
        # Return original input if conversion fails
        return f'{currency_symbol}{x}'

def get_currency(values, currency_symbols=None):
    currency_symbols = currency_symbols or ['€', '$', '£', '¥']
    for symbol in currency_symbols:
        if values.astype(str).str.contains(symbol, regex=False).any():
            return symbol
    return None

def clean_currency_value(value):
    currency_symbols = ['€', '$', '£', '¥']
    
    if isinstance(value, (int, float)):
        return float(value), None

    if isinstance(value, str):
        # Find the currency symbol in the value
        found_symbol = next((symbol for symbol in currency_symbols if symbol in value), None)
        
        if found_symbol:
            # Remove currency symbol and whitespace
            cleaned = value.replace(found_symbol, '').strip()
        else:
            cleaned = value.strip()
        
        # Replace comma with dot if comma is used as decimal separator
        if ',' in cleaned and '.' not in cleaned:
            cleaned = cleaned.replace(',', '.')
        
        # Remove thousands separators (commas or dots)
        cleaned = ''.join(c for c in cleaned if c.isdigit() or c == '.')
        
        try:
            return float(cleaned), found_symbol
        except ValueError:
            return 0.0, None

    return 0.0, None  # Return 0 and None for any other type

def clean_currency_values(values, currency_symbols=None):
    currency_symbols = currency_symbols or ['€', '$', '£', '¥']
    
    # Convert input to pandas Series if it's a numpy array
    if isinstance(values, np.ndarray):
        values = pd.Series(values)
    elif not isinstance(values, pd.Series):
        raise ValueError("Input must be a pandas Series or numpy array")

    # Detect currency symbol if not provided
    found_currency = None
    for symbol in currency_symbols:
        if values.astype(str).str.contains(symbol, regex=False).any():
            found_currency = symbol
            break
    
    # Remove currency symbol if found
    if found_currency:
        cleaned = values.astype(str).str.replace(found_currency, '', regex=False)
    else:
        cleaned = values.astype(str)
    
    # Convert to float
    cleaned = cleaned.str.replace(',', '', regex=False) # remove commas
    cleaned = pd.to_numeric(cleaned, errors='coerce')
    
    return cleaned, found_currency

def load_and_preprocess_csv(csv_file):
    df = pd.read_csv(csv_file.file.path)
    date_column = df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column])
    for col in df.columns[1:3]:
        df[col], _ = clean_currency_values(df[col])
    df.set_index(date_column, inplace=True)
    return df

def save_model_to_file_field(model, file_field, filename):
    """
    Save a model with a custom save method to a Django FileField.
    
    :param model: The model object with a save method that takes a filename
    :param file_field: The Django FileField to save the model to
    :param filename: The name to give the saved file
    """
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        model.save(tmp_file.name)
        tmp_file.seek(0)
        file_field.save(filename, File(tmp_file))
    os.unlink(tmp_file.name)