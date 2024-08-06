from django.contrib import admin
from .models import CSVFile

@admin.register(CSVFile)
class CSVFileAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'user', 'upload_date')
    list_filter = ('upload_date', 'user')
    search_fields = ('file_name', 'user__username')
    readonly_fields = ('upload_date',)

