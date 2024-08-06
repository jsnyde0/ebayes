from django.urls import path
from . import views

app_name = 'mmm'

urlpatterns = [
    path('', views.view_home, name='home'),
    path('test-htmx/', views.test_htmx, name='test-htmx'),
    path('upload/', views.view_upload, name='upload'),
    path('csv/<uuid:file_id>/', views.serve_csv, name='serve_csv'),
]