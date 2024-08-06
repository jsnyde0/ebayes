from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from django.shortcuts import redirect
from django.core.files import UploadedFile
import csv
import io

# Create your views here.
def view_home(request):
    return render(request, 'mmm/home.html')

def test_htmx(request):
    return HttpResponse('HTMX Works!')

def view_upload(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file: UploadedFile = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'Please upload a CSV file.')
            return redirect('mmm:upload')
        
        if csv_file.size > 5 * 1024 * 1024:  # 5 MB limit
            messages.error(request, 'File size must be under 5 MB.')
            return redirect('mmm:upload')
        
        try:
            decoded_file = csv_file.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)
            print('Uploaded CSV: ', io_string)
            for row in csv.reader(io_string, delimiter=','):
                # Process each row of the CSV file
                # For example:
                # YourModel.objects.create(field1=row[0], field2=row[1], ...)
                print(row)
            messages.success(request, 'CSV file uploaded successfully.')
        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
            
        return redirect('mmm:upload')
    
    return render(request, 'mmm/upload.html')