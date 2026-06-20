import requests
from django.conf import settings

WEATHER_URL = "https://api.openweathermap.org/data/2.5"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

def get_current_weather(lat, lon):
    """Текущая погода для поля"""
    params = {
        'lat': lat,
        'lon': lon,
        'appid': settings.OPENWEATHER_API_KEY,
        'units': 'metric',
        'lang': 'ru'
    }
    try:
        r = requests.get(f"{WEATHER_URL}/weather", params=params, timeout=5)
        data = r.json()
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'description': data['weather'][0]['description'],
            'rain_mm': data.get('rain', {}).get('1h', 0),
        }
    except Exception:
        # Заглушка если нет интернета
        return {
            'temperature': 25.0,
            'humidity': 60.0,
            'wind_speed': 5.0,
            'description': 'данные недоступны',
            'rain_mm': 0,
        }


def get_weather_forecast(lat, lon):
    """Прогноз на 5 дней"""
    params = {
        'lat': lat, 'lon': lon,
        'appid': settings.OPENWEATHER_API_KEY,
        'units': 'metric', 'lang': 'ru', 'cnt': 7
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=5)
        items = r.json().get('list', [])
        return [{
            'date': item['dt_txt'],
            'temp': item['main']['temp'],
            'humidity': item['main']['humidity'],
            'rain': item.get('rain', {}).get('3h', 0),
            'description': item['weather'][0]['description'],
        } for item in items]
    except Exception:
        return []


def check_disaster_risk(lat, lon):
    """Проверка риска стихийных явлений"""
    alerts = []
    weather = get_current_weather(lat, lon)
    forecast = get_weather_forecast(lat, lon)

    # Риск града: быстрое падение температуры + высокая влажность
    if weather['humidity'] > 80 and weather['wind_speed'] > 10:
        alerts.append({
            'type': 'hail',
            'level': 'high',
            'message': f'Высокий риск града: влажность {weather["humidity"]}%, ветер {weather["wind_speed"]} м/с'
        })

    # Риск селя: сильные осадки в прогнозе
    total_rain = sum(f.get('rain', 0) for f in forecast)
    if total_rain > 30:
        alerts.append({
            'type': 'flood',
            'level': 'high',
            'message': f'Риск селя: ожидается {round(total_rain, 1)}мм осадков за 24 часа'
        })

    return alerts