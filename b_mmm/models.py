from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import base64
import csv
import io
import os
import uuid
from typing import List, Optional, Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler


from .utils import clean_currency_values, get_currency

import matplotlib
matplotlib.use('Agg') # because TKinter backend is not thread-safe

# Constants
CSV_UPLOAD_DIR = 'csv_files'
PLOT_DIR = 'plots'

def get_file_path(instance: 'CSVFile', filename: str) -> str:
    """Generate a unique file path for uploaded CSV files."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join(CSV_UPLOAD_DIR, filename)

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
    MODEL_TYPE_LINEAR = 'linear_regression'
    MODEL_TYPE_BAYESIAN = 'bayesian_mmm'
    MODEL_TYPE_CHOICES = [
        (MODEL_TYPE_LINEAR, 'Linear Regression'),
        (MODEL_TYPE_BAYESIAN, 'Bayesian MMM'),
    ]

    csv_file = models.ForeignKey(CSVFile, on_delete=models.CASCADE, related_name='mmm')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='mmm')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPE_CHOICES, default=MODEL_TYPE_BAYESIAN)
    parameters = models.JSONField(default=dict)
    results = models.JSONField(default=dict)

    trace_plot = models.ImageField(upload_to=get_plot_path, null=True, blank=True)

    # _X: Optional[pd.DataFrame] = None
    # _y: Optional[pd.Series] = None
    # _model: Optional[LinearRegression] = None
    _mmm_model: Optional[pm.Model] = None
    _trace: Optional[pm.backends.base.MultiTrace] = None

    def __str__(self):
        return f"MMM for {self.csv_file.file_name} ({self.model_type})"

    def get_csv_file_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.csv_file.predictors, self.csv_file.sales

    def run_model(self):
        if self.model_type == self.MODEL_TYPE_LINEAR:
            return self._run_linear_regression()
        elif self.model_type == self.MODEL_TYPE_BAYESIAN:
            return self._run_bayesian_mmm()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _run_linear_regression(self): 
        X, y = self.get_csv_file_data()
        model = LinearRegression()
        model.fit(X, y)
        
        self.results = {
            'r_squared': round(model.score(X, y), 2),
            'coefficients': dict(zip(X.columns, model.coef_)),
            'intercept': model.intercept_,
            'predictions': model.predict(X).tolist()
        }

        return self.results

    # function that creates the chart data for the predicted values against the actual values
    def create_chart_actual_vs_predicted(self):
        chart_data = {
            'chart_id': 'chart_actual_vs_predicted',
            'index': self.csv_file.index.tolist(),
            'series': [self.csv_file.sales.tolist(), self.results['predictions']],
            'series_labels': ['Actual', 'Predicted'],
            'series_axes': ['y_left', 'y_left'],
            'x_label': 'Date',
            'y_label_left': 'Sales',
            'y_unit_left': self.csv_file.currency,
        }
        return chart_data

    def _run_bayesian_mmm(self):
        # Scale X and y
        X, y = self.get_csv_file_data()
        X_scaled, y_scaled = self._scale_data(X, y)
        self._mmm_model = self._build_bayesian_model(X_scaled, y_scaled)

        # Run Inference
        n_tune_samples = 500
        n_draw_samples = 6000
        n_chains = 4
        n_posterior_samples = n_draw_samples * n_chains # for each trace, we drew 'n_draw_samples' samples

        self._run_inference(n_draw_samples, n_chains) # assigns self._trace

        ## sample the posterior predictive (which we'll then extract later and visualize)
        # model_posterior_predictive = self._sample_posterior_predictive(trace)

        self.results = {
            # 'model_graph': self._visualize_model(),
            # 'trace': trace,
            # 'model_posterior_predictive': model_posterior_predictive
        }

        # Save the entire model instance
        self.save()

        return self.results
    
    def _scale_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        y_scaler = MaxAbsScaler()
        X_scaler = MaxAbsScaler()
        y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).flatten()
        X_scaled = X_scaler.fit_transform(X.to_numpy())
        return X_scaled, y_scaled
    
    def _build_bayesian_model(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> pm.Model:
        x_names = self.csv_file.predictor_names

        # Define the model
        ## define coordinates for clarity and ease of analysis
        dates = self.csv_file.index.to_numpy()
        n_dates = len(dates)
        coords = {"date": dates, "predictor": x_names}

        ## set up scaled linear feature that represents the timeframe
        ## because we want to model a trend (independent of the predictors), we'll set up a linear feature scaled between 0 and 1
        # time_linear_0_to_1 = ((abt.index - abt.index.min()) / (abt.index.max() - abt.index.min())).to_numpy()
        time_linear_0_to_1 = np.linspace(0, 1, n_dates)

        ## set up model definition
        with pm.Model(coords=coords) as model:
            ## 1. Add coordinates to model
            model.add_coord(name="date", values=dates)
            model.add_coord(name="predictor", values=x_names)

            ## 1. data container for our predictors
            predictor_values = pm.MutableData(name="predictor_values", value=X_scaled, dims=("date", "predictor"))

            ## 2. Trend
            ### let's capture the trend a linear line (an intercept and a slope)
            # intercept = pm.Normal(name="intercept", mu=0, sigma=4) # use this one if you also have a slope, perhaps?
            intercept = pm.HalfNormal('intercept', sigma=4)
            slope = pm.Normal(name="slope", mu=0, sigma=2)
            # t_tensor = pt.tensor(dtype='float32', shape=(58,), name='myvar')
            trend = pm.Deterministic(name="trend", var=intercept + slope * time_linear_0_to_1, dims="date")

            ## 3. Predictor / Channel Coefficients
            ### Define priors for the coefficients of the transformed predictors
            # predictor_coefficients = pm.HalfNormal('predictor_coefficients', sigma=2, shape=n_predictors)
            predictor_coefficients = pm.HalfNormal('predictor_coefficients', sigma=2, dims="predictor") # don't need the shape argument because it's derived from the dims argument
            ### predictor effect
            predictor_effect = pm.Deterministic(
                name="predictor_effect",
                var=pm.math.dot(predictor_values, predictor_coefficients),
                dims=("date")
            )
            
            ## 4. Sales (captured by a StudentT distribution for robustness against outliers)
            ### 4.1 Expected value of StudentT
            y_mu = pm.Deterministic(
                name="y_mu",
                var=intercept + predictor_effect,
                dims="date"
            )
            
            ### 4.2 Standard deviation of the StudentT
            sigma = pm.HalfCauchy('sigma', beta=1)
            ### 4.3 Degrees of freedom of the StudentT
            degrees_freedom = pm.Gamma(name="degrees_freedom", alpha=25, beta=2)
            
            ### 4.4 Putting it together in the StudenT likelihood function
            y_observed = pm.StudentT(
                name="y_observed", 
                nu=degrees_freedom, 
                mu=y_mu, sigma=sigma, 
                observed=y_scaled,
                dims="date"
            )
            # y_observed = pm.Normal('Y_obs', mu=y_mu, sigma=residual_sigma, observed=y_scaled)

        return model
    
    def _visualize_model(self):
        return pm.model_to_graphviz(self._mmm_model)
    
    def _run_inference(self, n_draw_samples: int = 6000, n_chains: int = 4):
        self._trace = pm.sample(
            model=self._mmm_model,
            nuts_sampler="numpyro",
            draws=n_draw_samples,
            chains=n_chains,
            idata_kwargs={"log_likelihood": True},
        )
        # return self._trace

    def _sample_posterior_predictive(self):
        model_posterior_predictive = pm.sample_posterior_predictive(
            trace=self._trace,
            model=self._mmm_model
        )
        return model_posterior_predictive

    def plot_trace(self):
        if not self._trace:
            raise ValueError("No trace found. Please run the model first.")

        var_names_to_plot = ["intercept", "predictor_coefficients", "sigma", "degrees_freedom"]
        n_vars = len(var_names_to_plot)

        axes = az.plot_trace(
            self._trace,
            var_names=var_names_to_plot,
            figsize=(15, 4*n_vars)
        )
        fig = axes.ravel()[0].figure

        # Save to an in-memory buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)

        plt.close(fig)

        # Save the plot directly from the buffer to the ImageField
        self.trace_plot.save(f'trace_plot.png', ContentFile(buffer.getvalue()), save=True)

        # Save the entire model instance
        self.save()



