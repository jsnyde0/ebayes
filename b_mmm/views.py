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
    
    # Prepare data for Chart.js
    labels = df.iloc[:, 0].tolist()
    sales_data = df.iloc[:, 1].apply(clean_euro_value).tolist()
    
    predictors = []
    for column in df.columns[2:]:  # Start from the third column
        predictors.append({
            'label': column,
            'data': df[column].apply(clean_euro_value).tolist(),
        })

    context = {
        'csv_file': csv_file,
        'labels': json.dumps(labels),
        'sales_data': json.dumps(sales_data),
        'predictors': predictors,  # Note: not JSON encoded
    }
    
    return render(request, 'mmm/preview.html', context)

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