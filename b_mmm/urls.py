from django.urls import path
from . import views

app_name = 'mmm'

urlpatterns = [
    path('', views.view_home, name='home'),
    path('test-htmx/', views.test_htmx, name='test-htmx'),
]