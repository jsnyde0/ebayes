from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, get_object_or_404
from django.http import HttpResponseForbidden, FileResponse
from .forms import CSVUploadForm
from .models import CSVFile
from django.core.exceptions import ValidationError
from .utils import process_csv, clean_euro_value
import json
import pandas as pd
import random

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
                csv_file: CSVFile = process_csv(form.cleaned_data['csv_file'], request.user)
                messages.success(request, f'CSV file "{csv_file.file_name}" uploaded successfully.')
            except ValidationError as e:
                messages.error(request, str(e))
            return redirect('mmm:preview', file_id=csv_file.id)
    else:
        form = CSVUploadForm()
    
    # note we're not passing the form to the template, we're just using it for validation
    return render(request, 'mmm/upload.html')

@login_required
def view_preview(request, file_id):
    csv_file = get_object_or_404(CSVFile, id=file_id, user=request.user)
    
    # Read the CSV file
    df = pd.read_csv(csv_file.file.path)
    index = df.iloc[:, 0].tolist()
    sales = df.iloc[:, 1].apply(clean_euro_value).tolist()
    
    # Create a chart for each predictor against the sales
    charts_data = []
    for i, predictor_col in enumerate(df.columns[2:], start=1):
        predictor_raw = df[predictor_col].tolist()
        # if the predictor has a currency value, we plot it on the same y-axis as sales
        predictor_has_currency = any('€' in str(value) for value in predictor_raw)
        
        predictor = df[predictor_col].apply(clean_euro_value if predictor_has_currency else float).tolist()
        
        chart_data = {
            'chart_id': f'chart_{i}',
            'index': index,
            'series': [sales, predictor],
            'series_labels': [df.columns[1], predictor_col],
            'series_axes': ['y_left', 'y_left' if predictor_has_currency else 'y_right'],
            'y_label_left': 'Value',
            'y_label_right': 'Value',
            'y_unit_left': '€',
            'y_unit_right': '€' if predictor_has_currency else '#'
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

    
    # Prepare data for Chart.js
    # index = {
    #     'label': 'Date',
    #     'data': df.iloc[:, 0].tolist(),
    # }
    # sales = {
    #     'label': 'Sales',
    #     'data': df.iloc[:, 1].apply(clean_euro_value).tolist(),
    # }
    
    # predictors = []
    # for column in df.columns[2:]:  # Start from the third column
    #     predictors.append({
    #         'label': column,
    #         'data': df[column].apply(clean_euro_value).tolist(),
    #     })

    # context = {
    #     'csv_file': csv_file,
    #     'index_data': json.dumps(index['data']),
    #     'index_label': index['label'],
    #     'sales_data': json.dumps(sales['data']),
    #     'sales_label': sales['label'],
    #     'predictors': [
    #         {
    #             'label': p['label'],
    #             'data': json.dumps(p['data'])
    #         } for p in predictors
    #     ],
    # }
    
    # return render(request, 'mmm/preview.html', context)

@login_required
def test_chart(request):
    csv_id = "419877a8-25b2-484f-a327-0b6863175bf6"
    csv_file = get_object_or_404(CSVFile, id=csv_id, user=request.user)

    df = pd.read_csv(csv_file.file.path)
    # Prepare data for Chart.js
    index = df.iloc[:, 0].tolist()
    series1 = df.iloc[:, 1].apply(clean_euro_value).tolist()
    series2 = df.iloc[:, 2].apply(clean_euro_value).tolist()
    series_labels = df.columns[1:3].tolist()
    
    context = {
        'csv_file': csv_file,
        'chart_id': f'test_chart_{random.randint(0, 1000000)}',
        'index': index,
        'series': [series1, series2],
        'series_labels': series_labels,
        'x_label': 'Date',
        'y_label': 'Value',
        'y_unit': '€'
    }
    if request.htmx:
        return render(request, 'partials/line_chart.html', context)
    
    return render(request, 'mmm/test_chart.html', context)
    
@login_required
def serve_csv(request, file_id):
    csv_file = get_object_or_404(CSVFile, id=file_id)
    if csv_file.user != request.user:
        return HttpResponseForbidden("You don't have permission to access this file.")
    
    response = FileResponse(csv_file.file, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{csv_file.file_name}"'
    return response