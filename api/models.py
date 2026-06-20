from django.db import models

class Field(models.Model):
    """Поле фермера"""
    name = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    area_hectares = models.FloatField()
    crop_type = models.CharField(
        max_length=50,
        default='apricot'  # абрикос по умолчанию
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class SoilReading(models.Model):
    """Показания датчиков почвы"""
    field = models.ForeignKey(Field, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    moisture = models.FloatField()      # влажность %
    temperature = models.FloatField()   # температура °C
    ph = models.FloatField()            # кислотность
    depth = models.FloatField(default=0.3)  # глубина м
    nitrogen = models.FloatField(null=True) # азот (опц.)

    class Meta:
        ordering = ['-timestamp']


class WeatherData(models.Model):
    """Погода с OpenWeatherMap"""
    field = models.ForeignKey(Field, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    temperature = models.FloatField()
    humidity = models.FloatField()
    wind_speed = models.FloatField()
    rain_mm = models.FloatField(default=0)
    description = models.CharField(max_length=200)

    class Meta:
        ordering = ['-timestamp']


class Alert(models.Model):
    """Уведомления для фермера"""
    ALERT_TYPES = [
        ('water', 'Нужен полив'),
        ('hail', 'Риск града'),
        ('flood', 'Риск селя'),
        ('fertilize', 'Нужно удобрение'),
    ]
    LEVELS = [
        ('low', 'Низкий'),
        ('medium', 'Средний'),
        ('high', 'Высокий'),
    ]
    field = models.ForeignKey(Field, on_delete=models.CASCADE)
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    level = models.CharField(max_length=10, choices=LEVELS)
    message = models.TextField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']


class Prediction(models.Model):
    """Прогнозы от PINN модели"""
    field = models.ForeignKey(Field, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    forecast_json = models.JSONField()  # прогноз на 7 дней
    days_until_water = models.IntegerField(null=True)
    