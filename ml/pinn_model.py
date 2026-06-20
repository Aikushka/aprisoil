import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt

# Отключаем режим eager execution для работы старого синтаксиса TF 1.x
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf_v1 = tf.compat.v1

class PhysicsInformedNN:
    def __init__(self, t, z, theta, layers_psi, layers_theta, layers_K):
        self.t = t
        self.z = z
        self.theta = theta
        self.layers_psi = layers_psi
        self.layers_theta = layers_theta
        self.layers_K = layers_K

        self.weights_psi, self.biases_psi = self.initialize_NN(layers_psi)
        self.weights_theta, self.biases_theta = self.initialize_MNN(layers_theta)
        self.weights_K, self.biases_K = self.initialize_MNN(layers_K)

        self.sess = tf_v1.Session(config=tf_v1.ConfigProto(allow_soft_placement=True,
                                                           log_device_placement=False))

        self.z_tf = tf_v1.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf_v1.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.theta_tf = tf_v1.placeholder(tf.float32, shape=[None, self.theta.shape[1]])
        
        self.theta_pred, self.psi_pred, self.K_pred, self.f_pred, _, _, _, _ = self.net(self.t_tf, self.z_tf)

        self.loss = tf.reduce_sum(tf.square(self.theta_tf - self.theta_pred)) + tf.reduce_sum(tf.square(self.f_pred))
        self.optimizer_Adam = tf_v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf_v1.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        in_dim, out_dim = size[0], size[1]
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=np.sqrt(2/(in_dim + out_dim))), dtype=tf.float32)

    def initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(len(layers)-1):
            weights.append(self.xavier_init([layers[l], layers[l+1]]))
            biases.append(tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32)))
        return weights, biases

    def initialize_MNN(self, layers):
        weights, biases = [], []
        for l in range(len(layers)-1):
            weights.append(self.xavier_init([layers[l], layers[l+1]])**2)
            biases.append(tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32)))
        return weights, biases

    def net_psi(self, X, weights, biases):
        H = X
        for l in range(len(weights)-1):
            H = tf.tanh(tf.add(tf.matmul(H, weights[l]), biases[l]))
        return -tf.exp(tf.add(tf.matmul(H, weights[-1]), biases[-1]))

    def net_theta(self, X, weights, biases):
        H = X
        for l in range(len(weights)-1):
            H = tf.tanh(tf.add(tf.matmul(H, weights[l]), biases[l]))
        return tf.sigmoid(tf.add(tf.matmul(H, weights[-1]), biases[-1]))

    def net_K(self, X, weights, biases):
        H = X
        for l in range(len(weights)-1):
            H = tf.tanh(tf.add(tf.matmul(H, weights[l]), biases[l]))
        return tf.exp(tf.add(tf.matmul(H, weights[-1]), biases[-1]))

    def net(self, t, z):
        X = tf.concat([t, z], 1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi)
        log_h = tf.math.log(-psi)
        theta = self.net_theta(-log_h, self.weights_theta, self.biases_theta)
        K = self.net_K(-log_h, self.weights_K, self.biases_K)
        
        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        K_z = tf.gradients(K, z)[0]
        f = theta_t - K_z*psi_z - K*psi_zz - K_z
        return theta, psi, K, f, theta_t, psi_z, psi_zz, K_z

    def train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.z_tf: self.z, self.theta_tf: self.theta}
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 200 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print(f'Итерация: {it}, Внутренний Loss: {loss_value:.3e}')

    def predict(self, t_star, z_star):
        tf_dict = {self.t_tf: t_star, self.z_tf: z_star}
        return self.sess.run([self.theta_pred, self.psi_pred], tf_dict)

def main_loop(hydrus, depth_increment, noise, num_layers_psi, num_neurons_psi, 
              num_layers_theta, num_neurons_theta, num_layers_K, num_neurons_K, number_random):
    
    tf_v1.reset_default_graph()
    tf_v1.set_random_seed(number_random)
    random.seed(number_random)
    np.random.seed(number_random)

    # Исправленный путь с учетом внутренней структуры архива
    file_path = f"./Node_Inf/hydrus_nod_files/{hydrus}_nod.csv"
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден.")
        return None

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    
    t = data['time'].values[:,None]
    z = data['depth'].values[:,None]
    theta = data['theta'].values[:,None]
    
    Z_star = np.hstack((t, z))
    theta_star = theta.flatten()[:,None]

    layers_psi = np.concatenate([[2], num_neurons_psi*np.ones(num_layers_psi), [1]]).astype(int).tolist()
    layers_theta = np.concatenate([[1], num_neurons_theta*np.ones(num_layers_theta), [1]]).astype(int).tolist()
    layers_K = np.concatenate([[1], num_neurons_K*np.ones(num_layers_K), [1]]).astype(int).tolist()

    fixed_position_full = [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95]
    fixed_position = fixed_position_full[::depth_increment]
    
    # Фильтруем строго по колонке 'zeta' с округлением, так как в 'depth' этих точек нет
    fixed_list = data.index[np.round(data['zeta'], 3).isin(np.round(fixed_position, 3))].values

    if len(fixed_list) == 0:
        print(f"Ошибка: Обучающая выборка пуста для {hydrus}. Проверь структуру колонок.")
        return None

    theta_train = theta_star[fixed_list, :] + noise*np.random.randn(len(fixed_list), 1)
    t_train, z_train = Z_star[fixed_list, 0:1], Z_star[fixed_list, 1:2]

    model = PhysicsInformedNN(t_train, z_train, theta_train, layers_psi, layers_theta, layers_K)
    print(f" Найдено {len(fixed_list)} точек контроля. Начинаем обучение {hydrus}...")
    model.train(1000) # Измени на 5000 для качественной сходимости
    return model, data
#%matplotlib inline

# Полный список файлов из твоего архива
soil_types = ['sandy_loam', 'loam', 'silt_loam', 'sandy_loam2', 'loam2', 'silt_loam2']

noise = 0
depth_increment = 1
num_layers_psi = 8
num_neurons_psi = 40
num_layers_theta = 1
num_neurons_theta = 10
num_layers_K = 1
num_neurons_K = 10
number_random = 111

for soil in soil_types:
    print(f"\n==========================================")
    print(f"ОБРАБОТКА ПОЧВЫ: {soil.upper()}")
    
    result = main_loop(soil, depth_increment, noise, num_layers_psi, num_neurons_psi, 
                       num_layers_theta, num_neurons_theta, num_layers_K, num_neurons_K, number_random)
    
    if result is None:
        continue
        
    model, data = result
    
    t_star = data['time'].values[:,None]
    z_star = data['depth'].values[:,None]
    theta_actual = data['theta'].values[:,None]

    # Получение предсказаний влажности (theta_pred)
    theta_pred, _ = model.predict(t_star, z_star)

    # Расчет метрики качества MSE по всему объему данных
    mse = np.mean((theta_actual - theta_pred) ** 2)
    print(f"Итоговый глобальный MSE для {soil}: {mse:.6e}")

    # Построение графиков
    unique_times = np.unique(t_star)
    selected_times = [unique_times[0], unique_times[len(unique_times)//2], unique_times[-1]]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f'Результаты PINN для почвы: {soil.upper()} (Глобальный MSE: {mse:.4e})', fontsize=14)

    # 3 временных среза распределения влажности по глубине
    for i, t_val in enumerate(selected_times):
        idx = t_star.flatten() == t_val
        
        axes[i].plot(theta_actual[idx], z_star[idx], 'b-', lw=2, label='Actual (HYDRUS)')
        axes[i].plot(theta_pred[idx], z_star[idx], 'r--', lw=2, label='PINN Prediction')
        
        axes[i].set_title(f'Время: {t_val:.2f}')
        axes[i].set_xlabel('Влажность (theta)')
        axes[i].set_ylabel('Глубина (z)')
        axes[i].legend()
        axes[i].grid(True)

    # 4-й график: Сравнение Real vs Predicted (Scatter plot)
    axes[3].scatter(theta_actual, theta_pred, alpha=0.2, s=2, color='darkgreen')
    
    min_val = min(theta_actual.min(), theta_pred.min())
    max_val = max(theta_actual.max(), theta_pred.max())
    axes[3].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Идеальное совпадение')
    
    axes[3].set_title('График Real vs Predicted')
    axes[3].set_xlabel('Фактическая влажность')
    axes[3].set_ylabel('Предсказанная влажность')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()
    
    # Закрытие сессии для очистки памяти видеокарты/ОЗУ в цикле Colab
    model.sess.close()