from keras.callbacks import TensorBoard, Callback
import tensorflow as tf
import matplotlib as plt
import numpy as np
import json

class CustomCallback(Callback):
    def __init__(self, model, x_test):
        self.model = model
        self.x_test = x_test
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test[:1], batch_size= 2)
        plt.figure(figsize=(10,10))
        plt.imshow(y_pred[0], cmap='gray')
        plt.show()

class MemoryMonitor(Callback):
    def __init__(self):
        self.static_memory = {}
        self.peak_memory = {}
        self.dynamic_memory = {}

    def _get_memory_info(self):
        usage = {}
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            device_name = gpu.name.split(':')[-2] + ':' + gpu.name.split(':')[-1]
            try:
                memory_info = tf.config.experimental.get_memory_info(device_name)
                usage[device_name] = {
                    'current': memory_info['current'] / (1024 ** 3),
                    'peak': memory_info['peak'] / (1024 ** 3)
                }
            except ValueError:
                pass
        return usage
    
    def on_train_begin(self, logs= None):
        current_usage = self._get_memory_info()
        print("-- Línea Base de memoria (memoria estática) --")
        for gpu, info in current_usage.items():
            self.static_memory[gpu] = info['current']
            print(f"{gpu}: {self.static_memory[gpu]:.4f} GB (ocupado por pesos + framework)")
        
        #Reset el contador pico para medir lo que ocurre en el entrenamiento
        for gpu in tf.config.list_physical_devices('CPU'):
            device_name = gpu.name.split(':')[-2] + ':' + gpu.name.split(':')[-1]
            try:
                tf.config.experimental.reset_memory_stats(device_name)
            except:
                pass
    
    def on_train_batch_end(self, batch, logs=None):
        #Se monitorea cada cierto tiempo
        if batch % 3 == 0:
            current_usage = self._get_memory_info()

            for gpu, info in current_usage.items():
                self.peak_memory[gpu] = info['peak']
                #Calculo de memoria dinámica (pico máximo - estática inicial)
                static = self.static_memory.get(gpu, 0)
                dynamic = info['peak'] - static
                self.dynamic_memory[gpu] = dynamic
    
    def on_train_end(self, logs=None):
        print("-- USO completo de memoria --")
        for gpu in self.static_memory.keys():
            static = self.static_memory[gpu]
            peak = self.peak_memory.get(gpu, 0)
            dynamic = self.dynamic_memory.get(gpu, 0)
            
            print(f"Dispositivo: {gpu}")
            print(f"1.- Memoria estática (Modelo):          {static:.4f} GB")
            print(f"2.- Memoria dinámica (Activaciones):    {dynamic:4f} GB")
            print(f"3.- Pico total alcanzado:               {peak:.4f}   GB")
            print("-"*30)

def add_last(data, new_vals):
    x_test_new = data[:, 1:]
    l = []
    for i in range(len(x_test_new)):
        l.append(np.append(x_test_new[i], new_vals[i]))
    x_test_new = np.array(l).reshape(data.shape[:])
    return x_test_new

def read_json_file(filename):
    f = open(f'configurations/{filename}', "r")
    parameters = json.load(f)
    print(type(parameters))
    return parameters