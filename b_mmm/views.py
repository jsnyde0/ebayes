from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, get_object_or_404
from django.http import HttpResponseForbidden, FileResponse, Http404
from .forms import CSVUploadForm
from .models import CSVFile, MarketingMixModel
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
    index = csv_file.index # values for the x-axis
    sales = csv_file.sales
    predictors = csv_file.predictors
    predictor_currencies = csv_file.predictor_currencies

    # Create a chart for each predictor against the sales
    charts_data = []
    for i, (predictor_name, predictor) in enumerate(predictors.items()):
        # Check for currency symbol in the predictor column and clean it up
        
        chart_data = {
            'chart_id': f'chart_{i}',
            'index': index.tolist(),
            'series': [sales.tolist(), predictor.tolist()],
            'series_labels': ['Sales', predictor_name],
            'series_axes': ['y_left', 'y_left' if predictor_currencies[i] else 'y_right'],
            'y_label_left': 'Sales',
            'y_label_right': '' if predictor_currencies[i] else 'Amount',
            'y_unit_left': '€',
            'y_unit_right': '' if predictor_currencies[i] else '#'
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
        
        try:
            csv_file = csv_files.get(id=file_id)
        except CSVFile.DoesNotExist:
            raise Http404("CSV file does not exist")
        
        mmm, created = MarketingMixModel.objects.get_or_create(
            csv_file=csv_file,
            user=request.user,
            model_type=model_type
        )

        mmm.run_model()

        model_coefficients = mmm.results['coefficients']
        model_intercept = mmm.results['intercept']
        print('model coefficients: ', model_coefficients)
        print('model intercept: ', model_intercept)

        R_squared = mmm.results['r_squared']
        print(f'R Squared:  {R_squared}')

        # Create a chart for the predicted values against the actual values
        chart_data = mmm.create_chart_actual_vs_predicted()

        context = {
            'csv_files': csv_files,
            'show_model_results': True,
            'r_squared': R_squared,
            'coefficients': dict(zip(csv_file.predictor_columns, model_coefficients)),
            'intercept': model_intercept,
            'chart_data': chart_data
        }
        
        return render(request, 'mmm/model.html', context)
    
    return render(request, 'mmm/model.html', {'csv_files': csv_files})