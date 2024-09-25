from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile

import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import csv
import io
import os
import uuid
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from typing import List, Optional, Dict
import logging


from .utils import clean_currency_values, get_currency, save_model_to_file_field, load_model_from_file_field

import matplotlib
matplotlib.use('Agg') # because TKinter backend is not thread-safe

logger = logging.getLogger(__name__)

# Constants
CSV_UPLOAD_DIR = 'csv_files'
PLOT_DIR = 'plots'

def get_file_path(instance: 'CSVFile', filename: str) -> str:
    """Generate a unique file path for uploaded CSV files."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join(CSV_UPLOAD_DIR, filename)

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
            self.save()
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
            self._X = self.csv_file.predictors
        return self._X
    
    def _update_state(self, new_state):
        self.state = new_state
        self.save()

    def fit_model_and_evaluate(self):
        logger.info(f"Starting model run for MMM {self.id}")
        if self.state == 'completed':
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

        self._update_state('plotting')
        logger.info(f"Generating all plots for MMM {self.id}")
        self._generate_all_plots()

        self._update_state('completed')
        logger.info(f"Saving results for MMM {self.id}")
        self._save_results()
        logger.info(f"Model run completed for MMM {self.id}")
    
    def _save_results(self):
        logger.info(f"Saving MMM model for instance {self.id}")
        if self._mmm is None:
            raise ValueError("No model found. Please run the model first.")
        save_model_to_file_field(self._mmm, self.saved_mmm, f'mmm_{self.id}.nc')
    
    def _load_saved_results(self):
        logger.info(f"Loading saved trace and y_posterior_predictive for MMM {self.id}")
        if not self.saved_mmm:
            logger.error("No saved model file found.")
            return
        
        self._mmm = load_model_from_file_field(MMM, self.mmm_file)

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
            date_column="date_week",
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
        
        sampler_kwargs = {
            "draws": n_draw_samples,
            "target_accept": 0.9,
            "chains": n_chains,
            "random_seed": 42,
        }

        self._mmm.fit(X=self.X, y=self.y, nuts_sampler="numpyro", **sampler_kwargs)

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
        
        plot_field = f"{plot_type}_plot"
        plot_method = getattr(self, f"_generate_{plot_type}_plot")

        if not getattr(self.model, plot_field):
            logger.info(f"Generating new {plot_type} plot for MMM {self.model.id}")
            plot_method()
        else:
            logger.info(f"Using existing {plot_type} plot for MMM {self.model.id}")
        
        plot_url = getattr(self.model, plot_field).url
        return plot_url

    # Add the plotting methods here
    def _generate_trace_plot(self):
        if self.model._mmm is None:
            raise ValueError("No model found. Please build and fit the model first.")

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
        # Plot the error percent
        fig = self.model._mmm.plot_errors(original_scale=True)

        # Save to an in-memory buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        # Save the plot directly from the buffer to the ImageField
        self.model.error_percent_plot.save(f'error_percent_plot_{self.model.id}.png', ContentFile(buffer.getvalue()), save=True)