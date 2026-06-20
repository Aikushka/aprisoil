import sys
from pathlib import Path
# Находим корень проекта и добавляем его на первое место в список путей Python
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import Field, SoilReading, WeatherData, Alert, Prediction
from .weather_service import get_current_weather, check_disaster_risk
from ml.predictor import predict_moisture_7days, check_water_needed
import json

# ── ПОЛЯ ──────────────────────────────────────────────

@csrf_exempt
def field_list(request):
    """GET /api/fields/ - список всех полей"""
    # Получаем данные из базы и сразу превращаем в список словарей
    fields = list(Field.objects.all().values(
        'id', 'name', 'latitude', 'longitude',
        'area_hectares', 'crop_type'
    ))

    # Если список пустой, добавляем тестовую точку
    if not fields:
        fields = [
            {
                "id": 999,
                "name": "Тестовый сектор ApriSoil",
                "latitude": 42.8746,   # Широта Бишкека
                "longitude": 74.5698,  # Долгота Бишкека
                "area_hectares": 12.5,
                "crop_type": "Wheat"
            }
        ]

    return JsonResponse({'fields': fields})



@csrf_exempt
def field_create(request):
    """POST /api/fields/create/ — добавить поле"""
    if request.method == 'POST':
        data = json.loads(request.body)
        field = Field.objects.create(
            name=data['name'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            area_hectares=data.get('area_hectares', 1.0),
            crop_type=data.get('crop_type', 'apricot')
        )
        return JsonResponse({'id': field.id, 'name': field.name})


# ── ДАТЧИКИ ───────────────────────────────────────────

def soil_data(request, field_id):
    """GET /api/fields//soil/ — данные почвы"""
    try:
        field = Field.objects.get(id=field_id)
    except Field.DoesNotExist:
        return JsonResponse({'error': 'Поле не найдено'}, status=404)

    readings = SoilReading.objects.filter(
        field=field
    ).values(
        'moisture','temperature','ph','timestamp'
    )[:20]

    last = readings.first() if readings else None
    return JsonResponse({
        'field': field.name,
        'current': last,
        'history': list(readings),
    })


@csrf_exempt
def soil_add(request, field_id):
    """POST /api/fields//soil/add/ — добавить показание"""
    if request.method == 'POST':
        data = json.loads(request.body)
        field = Field.objects.get(id=field_id)
        reading = SoilReading.objects.create(
            field=field,
            moisture=data['moisture'],
            temperature=data['temperature'],
            ph=data.get('ph', 6.5),
        )
        return JsonResponse({'status': 'ok', 'id': reading.id})


# ── ПРОГНОЗ PINN ──────────────────────────────────────

def forecast(request, field_id):
    """GET /api/fields//forecast/ — прогноз PINN"""
    try:
        field = Field.objects.get(id=field_id)
    except Field.DoesNotExist:
        return JsonResponse({'error': 'Поле не найдено'}, status=404)

    last_reading = SoilReading.objects.filter(field=field).first()
    current_moisture = last_reading.moisture if last_reading else 40.0
    current_temp = last_reading.temperature if last_reading else 25.0

    moisture_forecast = predict_moisture_7days(current_moisture, current_temp)
    days_to_water = check_water_needed(moisture_forecast)

    forecast_data = [
        {
            'day': i + 1,
            'moisture': val,
            'needs_water': val < 30,
        }
        for i, val in enumerate(moisture_forecast)
    ]

    recommendation = (
        f'Полив нужен через {days_to_water} дн.'
        if days_to_water else 'Состояние нормальное'
    )

    # Сохраняем прогноз в БД
    Prediction.objects.create(
        field=field,
        forecast_json=forecast_data,
        days_until_water=days_to_water
    )

    return JsonResponse({
        'field': field.name,
        'current_moisture': round(current_moisture, 1),
        'forecast': forecast_data,
        'recommendation': recommendation,
        'days_until_water': days_to_water,
    })


# ── ПОГОДА И АЛЕРТЫ ───────────────────────────────────

def weather(request, field_id):
    """GET /api/fields//weather/ — погода"""
    field = Field.objects.get(id=field_id)
    current = get_current_weather(field.latitude, field.longitude)
    return JsonResponse({'field': field.name, 'weather': current})


def alerts(request, field_id):
    """GET /api/fields//alerts/ — все уведомления"""
    field = Field.objects.get(id=field_id)

    # Проверяем стихийные явления
    disaster_alerts = check_disaster_risk(field.latitude, field.longitude)
    for a in disaster_alerts:
        Alert.objects.get_or_create(
            field=field,
            alert_type=a['type'],
            level=a['level'],
            defaults={'message': a['message'], 'is_active': True}
        )

    # Проверяем влажность
    last = SoilReading.objects.filter(field=field).first()
    if last and last.moisture < 30:
        Alert.objects.get_or_create(
            field=field,
            alert_type='water',
            level='high',
            defaults={
                'message': f'Влажность {round(last.moisture,1)}% — нужен полив',
                'is_active': True
            }
        )

    all_alerts = Alert.objects.filter(
        field=field, is_active=True
    ).values('alert_type','level','message','created_at')

    return JsonResponse({
        'field': field.name,
        'alerts': list(all_alerts),
        'count': all_alerts.count()
    })