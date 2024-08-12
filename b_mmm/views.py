from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, get_object_or_404
from django.http import HttpResponseForbidden, FileResponse
from .forms import CSVUploadForm
from .models import CSVFile
from django.core.exceptions import ValidationError
from .utils import load_and_preprocess_csv
from sklearn.linear_model import LinearRegression
import pandas as pd
import pandas as pd

# Create your views here.
def view_home(request):
    return render(request, 'mmm/home.html')

def test_htmx(request):
    return HttpResponse('HTMX Works!')

@login_required
def view_upload(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                csv_file: CSVFile = CSVFile.create_from_csv(form.cleaned_data['csv_file'], request.user)
                messages.success(request, f'CSV file "{csv_file.file_name}" uploaded successfully.')
            except ValidationError as e:
                messages.error(request, str(e))
            return redirect('mmm:preview', file_id=csv_file.id)
    else:
        form = CSVUploadForm()
    
    # note we're not passing the form to the template, we're just using it for validation
    return render(request, 'mmm/upload.html')

@login_required
def view_preview(request, file_id=None):
    if file_id is None:
        # get the latest file uploaded by the user
        csv_file = CSVFile.objects.filter(user=request.user).order_by('-created_at').first()
    else:
        csv_file = get_object_or_404(CSVFile, id=file_id, user=request.user)
    
    # Read the CSV file
    df = csv_file.data
    currency = csv_file.currency
    index = csv_file.index # values for the x-axis
    sales = csv_file.sales
    predictors = csv_file.predictors
    predictor_names = csv_file.predictor_names
    predictor_currencies = csv_file.predictor_currencies

    # Create a chart for each predictor against the sales
    charts_data = []
    for i, predictor in enumerate(predictors):
        # Check for currency symbol in the predictor column and clean it up
        
        chart_data = {
            'chart_id': f'chart_{i}',
            'index': index.tolist(),
            'series': [sales.tolist(), predictor.tolist()],
            'series_labels': ['Sales', predictor_names[i]],
            'series_axes': ['y_left', 'y_left' if predictor_currencies[i] else 'y_right'],
            'y_label_left': 'Sales',
            'y_label_right': 'Value',
            'y_unit_left': '€',
            'y_unit_right': predictor_currencies[i] if predictor_currencies[i] else '#'
        }
        charts_data.append(chart_data)
    
    context = {
        'csv_file': csv_file,
        'charts_data': charts_data,
        'x_label': 'Date',
        'y_label': 'Value',
        'y_unit': '€'
    }
    
    return render(request, 'mmm/preview.html', context)
    
@login_required
def serve_csv(request, file_id):
    csv_file = get_object_or_404(CSVFile, id=file_id)
    if csv_file.user != request.user:
        return HttpResponseForbidden("You don't have permission to access this file.")
    
    response = FileResponse(csv_file.file, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{csv_file.file_name}"'
    return response

@login_required
def view_model(request):
    csv_files = CSVFile.objects.filter(user=request.user).order_by('-created_at')

    if request.method == 'POST':
        file_id = request.POST.get('file_id')
        model_type = request.POST.get('model_type')
        
        csv_file = get_object_or_404(CSVFile, id=file_id, user=request.user)
        abt = load_and_preprocess_csv(csv_file)
        
        # index contains the dates
        index = abt.index
        index_as_strings = index.strftime('%Y-%m-%d').tolist() # needed for the chart
        # 1st column is the sales
        y_column = abt.columns[0]
        y = abt[y_column]
        print('y: \n', y.head())
        # all other columns are predictors
        x_columns = abt.columns[1:]
        X = abt[x_columns]
        print('X: \n', X.head())
        if model_type == 'linear_regression':
            # Fit a linear regression model
            model = LinearRegression()
            model.fit(X, y)

            model_coefficients = model.coef_
            model_intercept = model.intercept_
            print('model coefficients: ', model_coefficients)
            print('model intercept: ', model_intercept)

            R_squared = round(model.score(X, y), 2)
            print(f'R Squared:  {R_squared}')

            # Make predictions
            y_column_predicted = y_column + '_predicted'
            # make predictions
            abt[y_column_predicted] = model.predict(X)
            ## convert to integers
            abt[y_column_predicted] = round(abt[y_column_predicted], 2)
            abt.head()

            # Create a chart for the predicted values against the actual values
            chart_data = {
                'chart_id': 'chart_actual_vs_predicted',
                'index': index_as_strings,
                'series': [y.tolist(), abt[y_column_predicted].tolist()],
                'series_labels': ['Actual', 'Predicted'],
                'series_axes': ['y_left', 'y_left'],
                'x_label': 'Date',
                'y_label_left': 'Sales',
                'y_unit_left': '€',
            }

            context = {
                'csv_files': csv_files,
                'show_model_results': True,
                'r_squared': R_squared,
                'coefficients': dict(zip(x_columns, model_coefficients)),
                'intercept': model_intercept,
                'chart_data': chart_data
            }
            
            return render(request, 'mmm/model.html', context)
    
    return render(request, 'mmm/model.html', {'csv_files': csv_files})