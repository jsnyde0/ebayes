from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, get_object_or_404
from django.http import HttpResponseForbidden, FileResponse
from .forms import CSVUploadForm
from .models import CSVFile
from django.core.exceptions import ValidationError
from .utils import process_csv, clean_currency_values
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
def view_preview(request, file_id=None):
    if file_id is None:
        # get the latest file uploaded by the user
        csv_file = CSVFile.objects.filter(user=request.user).order_by('-created_at').first()
    else:
        csv_file = get_object_or_404(CSVFile, id=file_id, user=request.user)
    
    # Read the CSV file
    df = pd.read_csv(csv_file.file.path)
    index = df.iloc[:, 0].tolist() # values for the x-axis
    sales, sales_currency = clean_currency_values(df.iloc[:, 1])
    sales = sales.tolist()
    
    # Create a chart for each predictor against the sales
    charts_data = []
    for i, predictor_col in enumerate(df.columns[2:], start=1):
        # Check for currency symbol in the predictor column and clean it up
        predictor, predictor_currency = clean_currency_values(df[predictor_col], currency_symbols=[sales_currency])
        
        chart_data = {
            'chart_id': f'chart_{i}',
            'index': index,
            'series': [sales, predictor.tolist()],
            'series_labels': [df.columns[1], predictor_col],
            'series_axes': ['y_left', 'y_left' if predictor_currency else 'y_right'],
            'y_label_left': 'Sales',
            'y_label_right': 'Value',
            'y_unit_left': '€',
            'y_unit_right': predictor_currency if predictor_currency else '#'
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
    return render(request, 'mmm/model.html')