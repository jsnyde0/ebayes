from django.contrib import admin
from .models import CSVFile

@admin.register(CSVFile)
class CSVFileAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'user', 'created_at')
    list_filter = ('created_at', 'user')
    search_fields = ('file_name', 'user__username')
    readonly_fields = ('created_at',)

