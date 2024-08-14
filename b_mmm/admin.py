from django.contrib import admin
from .models import CSVFile, MarketingMixModel

@admin.register(CSVFile)
class CSVFileAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'user', 'created_at')
    list_filter = ('created_at', 'user')
    search_fields = ('file_name', 'user__username')
    readonly_fields = ('created_at',)


@admin.register(MarketingMixModel)
class MarketingMixModelAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'csv_file', 'created_at', 'updated_at')
    list_filter = ('user',)
    search_fields = ('user__username', 'csv_file__file_name')
    readonly_fields = ('created_at', 'updated_at')

