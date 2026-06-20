import numpy as np
import pandas as pd
import os
import tensorflow as tf



tf.compat.v1.disable_eager_execution()

DATA_PATH = os.path.join(os.path.dirname(__file__), 'sandy_loam_nod.csv')

_model = None  # кэш модели

def get_model():
    """Загружаем модель один раз и кэшируем"""
    global _model
    if _model is None:
        from ml.pinn_model import PhysicsInformedNN, main_loop
        _model = main_loop(
            hydrus='sandy_loam',
            depth_increment=1,
            noise=0,
            num_layers_psi=8,
            num_neurons_psi=40,
            num_layers_theta=1,
            num_neurons_theta=10,
            num_layers_K=1,
            num_neurons_K=10,
            number_random=111
        )
    return _model


def predict_moisture_7days(current_moisture, temperature):
    """
    Прогноз влажности почвы на 7 дней вперёд.
    Возвращает список из 7 значений.
    """
    df = pd.read_csv(DATA_PATH)
    t_star = df['time'].values[:, None]
    z_star = df['depth'].values[:, None]

    try:
        model = get_model()
        results = model.predict(t_star, z_star)
        theta_pred = results[0].flatten()

        # Масштабируем под текущую влажность
        base = theta_pred[-7:] * 100 if len(theta_pred) >= 7 else [current_moisture]*7
        forecast = []
        for i, val in enumerate(base[:7]):
            adjusted = val * (current_moisture / 40.0)
            forecast.append(round(float(adjusted), 1))
        return forecast

    except Exception:
        # Если модель не загрузилась — простой физический расчёт
        forecast = []
        m = current_moisture
        for _ in range(7):
            evaporation = temperature * 0.15
            m = max(10, m - evaporation + np.random.normal(0, 1))
            forecast.append(round(m, 1))
        return forecast
    
def check_water_needed(forecast):
    """Через сколько дней нужен полив"""
    for i, val in enumerate(forecast):
        if val < 30:
            return i + 1
    return None