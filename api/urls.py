from django.urls import path
from . import views

urlpatterns = [
    path('fields/', views.field_list),
    path('fields/create/', views.field_create),
    path('fields/<int:field_id>/soil/', views.soil_data),
    path('fields/<int:field_id>/soil/add/', views.soil_add),
    path('fields/<int:field_id>/forecast/', views.forecast),
    path('fields/<int:field_id>/weather/', views.weather),
    path('fields/<int:field_id>/alerts/', views.alerts),
]