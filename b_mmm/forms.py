from django import forms
from django.core.validators import FileExtensionValidator

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        validators=[FileExtensionValidator(allowed_extensions=['csv'])]
    )

    def clean_csv_file(self):
        csv_file = self.cleaned_data['csv_file']
        if csv_file.size > 5 * 1024 * 1024:  # 5 MB limit
            raise forms.ValidationError("File size must be under 5 MB.")
        return csv_file