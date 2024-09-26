from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.core.files import File

import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import io
import os
import uuid
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import tempfile


from .utils import clean_currency_values

import matplotlib
matplotlib.use('Agg') # because TKinter backend is not thread-safe

logger = logging.getLogger(__name__)

# Constants
CSV_UPLOAD_DIR = 'csv_files'
CLEANED_DATA_DIR = 'csv_cleaned'
PLOT_DIR = 'plots'

def get_file_path(instance: 'CSVFile', filename: str) -> str:
    """Generate a unique file path for uploaded CSV files."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join(CSV_UPLOAD_DIR, str(instance.user.id), filename)

def get_cleaned_file_path(instance: 'CSVFile', filename: str) -> str:
    """Generate file path for cleaned data file"""
    cleaned_filename = f"{filename}_cleaned"
    return os.path.join(CLEANED_DATA_DIR, str(instance.user.id), cleaned_filename)

def get_model_path(instance: 'MarketingMixModel', filename: str) -> str:
    return f'mmm_models/mmm_{instance.id}.nc'

def get_plot_path(instance: 'MarketingMixModel', filename: str) -> str:
    """Generate a unique file path for plot images."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join(PLOT_DIR, filename)

class CSVFile(models.Model):
    """Model to store and process uploaded CSV files in a format handy for MMM."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to=get_file_path)
    cleaned_data_file = models.FileField(upload_to=get_cleaned_file_path, null=True, blank=True)
    currencies = models.JSONField(default=dict)  # List of predictor currencies
    
    # column names
    date_name = models.CharField(max_length=255, default='date')
    sales_name = models.CharField(max_length=255, default='sales')
    predictor_names = models.JSONField(default=list)  # List of predictor column names
    

    # _cached_data is for internal use only and can be either a pandas DataFrame or None.
    _cached_data: Optional[pd.DataFrame] = None # TODO when lazy loading this, load it from cleaned_data_file

    class Meta:
        verbose_name_plural = 'CSV Files'
        verbose_name = 'CSV File'
        ordering = ['-created_at']

    def __str__(self) -> str:
        return self.file_name

    @property
    def data(self) -> pd.DataFrame:
        """Load the cleaned dataset."""
        if self._cached_data is None:
            if self.cleaned_data_file:
                self._cached_data = pd.read_parquet(self.cleaned_data_file.path)
            else:
                raise ValueError("Cleaned data not available. The CSV file may not have been processed correctly.")
        return self._cached_data
    
    @property
    def index(self) -> pd.Index:
        """Get the index column"""
        return self.data.index
    
    @property
    def currency(self) -> str:
        """Get the currency used in the CSV file (derived from sales column)."""
        return self.currencies[self.sales_name]
    
    @property
    def predictor_currencies(self) -> List[str]:
        """Get the currencies used in predictor columns."""
        return [self.currencies[col] for col in self.predictor_names]
    
    @property
    def sales(self) -> pd.Series:
        """Get cleaned sales data (without currency symbol and converted to float)."""
        return self.data[self.sales_name]
    
    @property
    def predictors(self) -> pd.DataFrame:
        """Get cleaned predictor data."""
        return self.data[self.predictor_names]

    @property
    def date(self) -> pd.Series:
        """Get the date column."""
        return self.data[self.date_name]
    
    @classmethod
    def create_from_csv(cls, csv_file: 'UploadedFile', user: User) -> 'CSVFile': # TODO: ask why we still need classmethod instead of just __init__?
        """Process the uploaded CSV file and create a CSVFile instance."""
        try:
            csv_file.seek(0)
            df = pd.read_csv(csv_file)
            
            headers = df.columns.tolist()
            
            if not headers or len(headers) < 3:
                raise ValidationError("CSV file must have at least 3 columns: date, sales, and at least one predictor.")
            
            csv_file_instance = cls.objects.create(
                user=user,
                file_name=csv_file.name,
                file=csv_file,
                date_name=headers[0],
                sales_name=headers[1],
                predictor_names=headers[2:]
            )

            # Clean the data and get currencies
            cleaned_df, currencies = csv_file_instance.clean_data(df)
            
            # Store cleaned data as parquet file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
                table = pa.Table.from_pandas(cleaned_df)
                pq.write_table(table, tmp.name)
                tmp_path = tmp.name

            with open(tmp_path, 'rb') as f:
                csv_file_instance.cleaned_data_file.save(f'{csv_file.name}_cleaned.parquet', f)

            os.unlink(tmp_path)  # Remove the temporary file
            
            csv_file_instance.currencies = currencies
            
            csv_file_instance.save()
            
            return csv_file_instance
        except (pd.errors.EmptyDataError, ValidationError) as e:
            raise ValidationError(f'CSV processing error: {str(e)}')
        except Exception as e:
            raise ValidationError(f'Unexpected error processing CSV: {str(e)}')

    @staticmethod
    def week_to_date(date_str):
        """Convert various date string formats to date object"""
        try:
            # Try parsing as a standard date format (handles YYYY-MM-DD, MM/DD/YYYY, etc.)
            return pd.to_datetime(date_str).date()
        except ValueError:
            # Check if it's in YYYY-WW format
            if '-' in date_str and len(date_str.split('-')) == 2:
                year, week = map(int, date_str.split('-'))
                return CSVFile._week_number_to_date(year, week)
            else:
                raise ValueError(f"Unable to parse date: {date_str}")

    @staticmethod
    def _week_number_to_date(year, week):
        """Convert year and week number to a date object"""
        if week == 0:
            # Handle the case where week is 0 (last week of previous year)
            date = datetime(year-1, 12, 28)  # December 28th is always in the last week of the year
        else:
            # Find the first day of the given week
            date = datetime(year, 1, 1)  # January 1st of the given year
            date += timedelta(days=(week-1)*7)  # Move to the start of the specified week
        
        # Adjust to get the Monday of that week
        while date.weekday() != 0:
            date -= timedelta(days=1)
        
        return date.date()

    def clean_data(self, df):
        """Clean the dataframe by converting dates, currencies, and sorting."""
        # Replace empty strings with 0
        df = df.replace('', '0')

        # Convert date column to datetime
        try:
            df[self.date_name] = df[self.date_name].apply(self.week_to_date)
            df[self.date_name] = pd.to_datetime(df[self.date_name])
        except ValueError as e:
            raise ValidationError(f"Error converting dates: {str(e)}")

        # Detect currencies and convert values to floats
        currencies = {}
        for col in df.columns:
            if col != self.date_name:
                df[col], currency = clean_currency_values(df[col])
                currencies[col] = currency

        # Drop future dates
        current_date = pd.Timestamp(datetime.now().date())
        df = df[df[self.date_name] <= current_date]

        # Sort by date and reset index
        df = df.sort_values(by=self.date_name, ascending=True).reset_index(drop=True)

        return df, currencies

    

class MarketingMixModel(models.Model):
    csv_file = models.ForeignKey(CSVFile, on_delete=models.CASCADE, related_name='mmm')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='mmm')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    state = models.CharField(max_length=20, default='initialized')

    # plot fields
    trace_plot = models.ImageField(upload_to=get_plot_path, null=True, blank=True)
    y_posterior_predictive_plot = models.ImageField(upload_to=get_plot_path, null=True, blank=True)
    error_percent_plot = models.ImageField(upload_to=get_plot_path, null=True, blank=True)
    
    # computation results
    saved_mmm = models.FileField(upload_to=get_model_path, null=True, blank=True)

    _X: Optional[pd.DataFrame] = None
    _y: Optional[pd.Series] = None
    _mmm: Optional[MMM] = None

    def __str__(self):
        return f"MMM {self.id} for {self.csv_file.file_name}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plotter = MMModelPlotter(self)

    @property
    def y(self):
        if self._y is None:
            self._y = self.csv_file.sales
        return self._y

    @property
    def X(self):
        if self._X is None:
            # PyMC-Marketing expects X to be the predictors with the first column the date
            self._X = pd.concat([self.csv_file.date.rename('date'), self.csv_file.predictors], axis=1)
        return self._X
    
    def _update_state(self, new_state):
        self.state = new_state
        self.save()

    def fit_model_and_evaluate(self):
        logger.info(f"Starting model run for MMM {self.id}")
        if 'completed' in self.state:
            self._load_saved_results()
            return
        
        self._update_state('building')
        logger.info(f"Computing new results for MMM {self.id}")
        # Scale X and y
        self._mmm = self._build_bayesian_model()

        # Run Inference
        n_draw_samples = 6000
        n_chains = 4
        self._update_state('inferencing')
        logger.info(f"Running inference with {n_draw_samples} samples and {n_chains} chains")
        self._run_inference(n_draw_samples, n_chains)
        self._update_state('completed')
        logger.info(f"Model run completed for MMM {self.id}, saving results...")
        self._save_model_to_file_field()

        logger.info(f"Generating all plots for MMM {self.id}")
        self._generate_all_plots()
        self._update_state('completed-and-plotted')

    def _save_model_to_file_field(self):
        logger.info(f"Saving MMM model for instance {self.id}")
        if self._mmm is None:
            raise ValueError("No model found. Please run the model first.")

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            self._mmm.save(tmp_file.name)
            tmp_file.seek(0)
            self.saved_mmm.save(f'mmm_{self.id}.nc', File(tmp_file))
        os.unlink(tmp_file.name)
    
    def _load_saved_results(self):
        logger.info(f"Loading saved trace and y_posterior_predictive for MMM {self.id}")
        if not self.saved_mmm:
            logger.error("No saved model file found.")
            return

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            tmp_file.write(self.saved_mmm.read())
            tmp_file.flush()
            self._mmm = MMM.load(tmp_file.name)

        os.unlink(tmp_file.name)


    def _build_bayesian_model(self) -> MMM:        
        my_sampler_config = {"progressbar": True}

        my_model_config = {
            "intercept": {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            },
        }

        model = MMM(
            model_config=my_model_config,
            sampler_config=my_sampler_config,
            date_column="date",
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            channel_columns=self.csv_file.predictor_names,
            # control_columns=[
            #     "event_1",
            #     "event_2",
            #     "t",
            # ],
            yearly_seasonality=1,
        )

        return model
    
    def _run_inference(self, n_draw_samples: int = 6000, n_chains: int = 4):
        if self._mmm is None:
            logger.error("No model found. Please run the model first.")
            raise ValueError("No model found. Please run the model first.")
        
        # sampler_kwargs = {
        #     "draws": n_draw_samples,
        #     "target_accept": 0.9,
        #     "chains": n_chains,
        #     "random_seed": 42,
        # }

        # self._mmm.fit(X=self.X, y=self.y, nuts_sampler="numpyro", **sampler_kwargs)
        self._mmm.fit(X=self.X, y=self.y, target_accept=0.85, chains=4, random_seed=42)

    def _generate_all_plots(self):
        self.plotter.generate_all_plots()

    def get_plot_url(self, plot_type):
        return self.plotter.get_plot_url(plot_type)

class MMModelPlotter:
    def __init__(self, model):
        self.model = model

    def generate_all_plots(self):
        self._generate_trace_plot()
        self._generate_y_posterior_predictive_plot()
        self._generate_error_percent_plot()

    def get_plot_url(self, plot_type):
        """Get or create a plot of the specified type."""
        if plot_type not in ['trace', 'y_posterior_predictive', 'error_percent']:
            raise ValueError("Invalid plot type. Please choose from 'trace', 'y_posterior_predictive' or 'error_percent'.")
        
        plot_field_name = f"{plot_type}_plot"
        plot_method = getattr(self, f"_generate_{plot_type}_plot")

        if not getattr(self.model, plot_field_name):
            logger.info(f"Generating new {plot_type} plot for MMM {self.model.id}")
            plot_method()
        else:
            logger.info(f"Using existing {plot_type} plot for MMM {self.model.id}")
        
        plot_field = getattr(self.model, plot_field_name)
        plot_url = plot_field.url if plot_field else None
        if plot_url is None:
            logger.debug(f"No {plot_type} plot URL available for MMM {self.model.id}")
        return plot_url

    # Add the plotting methods here
    def _generate_trace_plot(self):
        if self.model._mmm is None:
            logger.debug("No model found so not generating trace plot.")
            return
            # raise ValueError("No model found. Please build and fit the model first.") # TODO remove this

        # Plot the trace
        axes = az.plot_trace(
            data=self.model._mmm.fit_result,
            var_names=[
                "intercept",
                "y_sigma",
                "saturation_beta",
                "saturation_lam",
                "adstock_alpha",
                "gamma_control",
                "gamma_fourier",
            ],
            compact=True,
            backend_kwargs={"figsize": (12, 10), "layout": "constrained"},
        )
        fig = axes.ravel()[0].figure

        # Save to an in-memory buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)

        plt.close(fig)

        # Save the plot directly from the buffer to the ImageField
        self.model.trace_plot.save(f'trace_plot_{self.model.id}.png', ContentFile(buffer.getvalue()), save=True)

    def _generate_y_posterior_predictive_plot(self):
        if self.model._mmm is None:
            logger.debug("No model found so not generating posterior predictive plot.")
            return

        # Sample the posterior predictive
        self.model._mmm.sample_posterior_predictive(self.model.X, extend_idata=True, combined=True)

        # Plot the posterior predictive
        fig = self.model._mmm.plot_posterior_predictive(original_scale=True)

        # Save to an in-memory buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        # Save the plot directly from the buffer to the ImageField
        self.model.y_posterior_predictive_plot.save(f'y_posterior_predictive_plot_{self.model.id}.png', ContentFile(buffer.getvalue()), save=True)

    def _generate_error_percent_plot(self):
        if self.model._mmm is None:
            logger.debug("No model found so not generating error percent plot.")
            return

        # Plot the error percent
        fig = self.model._mmm.plot_errors(original_scale=True)

        # Save to an in-memory buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        # Save the plot directly from the buffer to the ImageField
        self.model.error_percent_plot.save(f'error_percent_plot_{self.model.id}.png', ContentFile(buffer.getvalue()), save=True)