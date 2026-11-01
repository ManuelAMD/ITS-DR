import json
import time
import numpy as np
from mapPreprocessing import Preprocessing
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import TensorBoard, Callback, EarlyStopping, ReduceLROnPlateau
from tools import MemoryMonitor, add_last, read_json_file

class AutoencoderCodification():

    def __init__(self, config_file):
        config_json = read_json_file(config_file)
        self.window = config_json['window_size']
        self.rows = config_json['rows']
        self.cols = config_json['cols']
        self.channels = config_json['channels']
        self.horizon = config_json['horizon']
        self.name = config_json['name'] + f"_model_testing_{int(time.time())}"
        self.optimizer = config_json['optimizer']
        self.data_name = f"{config_json['folder_models_save']}/{config_json['folder']}.npy"
        self.early_stopping_value = config_json['deep_training_early_stopping_patience']
        self.epochs= config_json['epochs']
        self.batch_size = config_json['batch_size']

    def get_autoencoder_config(self):
        #Read form a json configuration file
        return [[8, (3,3), "relu", (2,2)],
                [4, (3,3), "relu", (2,2)],
                [2, (3,3), "relu", (2,2)]]

    def get_model_shape(self, model, layer_i= -1):
        layer = model.layers[layer_i]
        layer_shape = layer.output.shape[1:]
        return layer_shape
        
    def get_encoder(self, model, input_dim):
        encoder = input_dim
        cant = int(len(model.layers)/2) - 1
        for i in range(cant):
            encoder = model.layers[i+1](encoder)
        encoder = keras.Model(input_dim, encoder)
        return encoder

    def get_decoder(self, model):
        cant = int(len(model.layers)/2)
        decoder_input = self.get_model_shape(model, cant)
        print("Actual model shape: {} con {} y {}".format(decoder_input, model, cant))
        input_shape = keras.layers.Input(shape= decoder_input)
        decoder = input_shape
        for i in reversed(range(cant)):
            decoder = model.layers[-(i+1)](decoder)
        decoder = keras.Model(input_shape, decoder)
        return decoder, decoder_input

    def construct_autoencoder(self, CMPBlocks, input_shape: tuple,):
        input = keras.layers.Input(input_shape)
        layers = input
        #Encoder layers
        for CMPBlock in CMPBlocks:
            layers = keras.layers.Conv2D(CMPBlock[0], CMPBlock[1], activation= CMPBlock[2], padding= 'same')(layers)
            layers = keras.layers.MaxPooling2D(CMPBlock[3])(layers)
        #Decoder layers
        for CMPBlock in reversed(CMPBlocks):
            layers = keras.layers.Conv2D(CMPBlock[0], CMPBlock[1], activation= CMPBlock[2], padding= 'same')(layers)
            layers = keras.layers.UpSampling2D(CMPBlock[3])(layers)
        layers = keras.layers.Conv2D(input_shape[2], (3, 3), activation= 'relu', padding= 'same')(layers)
        autoencoder = keras.Model(input, layers)
        print("Autoencoder model configuration")
        autoencoder.summary()
        encoder = self.get_encoder(autoencoder, input)
        decoder, _ = self.get_decoder(autoencoder)
        return autoencoder, encoder, decoder

    def get_prediction_model(self, inp, channels):
        #change for an automatic configuration file
        m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
        #m = keras.layers.Dropout(0.25)(m)
        m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
        return m

    def autoencoder_reduction(self, data, validation_part= 0.3):
        x_train = data[int(len(data) * validation_part):]
        x_val = data[: int(len(data) * validation_part)]
        CMPBlocks = self.get_autoencoder_config()
        autoencoder, encoder, decoder = self.construct_autoencoder(CMPBlocks, (data.shape[1], data.shape[2], 1))
        #Cambiar el valor de paciencia por el leido de un archivo de configuración
        es = EarlyStopping(monitor= 'val_loss', mode= 'min', patience= 10, restore_best_weights= True)
        memory_monitor = MemoryMonitor()
        autoencoder.compile(optimizer= "adam", loss= "mae")
        autoencoder.summary()
        #As well, take parameters from a config file
        history = autoencoder.fit(
            x_train, x_train, 
            epochs= 100, 
            validation_data= (x_val, x_val), 
            shuffle= True, 
            verbose= 1, 
            callbacks= [es, memory_monitor]
        )
        print(history)
        #Whole Data codification
        encoded_data = encoder.predict(data)
        save = input("Do you want to store all the autoencoders models (y/n)")
        if save.lower() == 'y':
            autoencoder.save("Models/actual_autoencoder.h5")
            decoder.save("Models/actual_decoder.h5")
            encoder.save("Models/actual_encoder.h5")
        else:
            print("only the decoder will be saved!")
            #store actual decoder for future decodification
            decoder.save("Models/actual_decoder.h5")
        return encoded_data

    def map_prediction_recursive(self, model: keras.Model, x_test: np.array, horizon: int):
        x_aux = x_test
        total_preds = []
        for i in range(horizon):
            predictions = model.predict(x_aux, batch_size= 2)
            total_preds.append(predictions)
            x_aux = add_last(x_aux, predictions[:])
        total_preds = np.array(total_preds)
        #Redimension for correct positioning
        total_preds = np.transpose(total_preds, (1, 0, 2, 3, 4))
        return total_preds

    def start_autoencoder_processing(self, preprocess: Preprocessing, load_and_predict= False, model_name= '', display= False):
        #Load from a folder of images
        inp = input("Use GPU? (y/n): ")
        if inp.lower() == 'y':
            strategy = tf.distribute.OneDeviceStrategy(device= '/gpu:0')
        else:
            strategy = tf.distribute.OneDeviceStrategy(device= '/cpu:0')
        data = self.autoencoder_reduction(preprocess.get_full_data())
        start_time = time.time()
        x_train_cod, y_train_cod, x_validation_cod, y_validation_cod, x_test_cod, y_test_cod = preprocess.autoencoder_codification(data, self.window)
        count = 0
        i = 0
        while i < len(x_train_cod):
            print(f"Processing codification part {i}")
            x_train = x_train_cod[i]
            y_train = y_train_cod[i]
            x_validation = x_validation_cod[i]
            y_validation = y_validation_cod[i]
            x_test = x_test_cod[i]
            y_test = y_test_cod[i]
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                if load_and_predict:
                    model = keras.models.load_model(model_name)
                    err = model.evaluate(x_test, y_test, batch_size= 2)
                    print(f"Model error: {err}")
                    prediction = self.map_prediction_recursive(model, x_test, self.horizon)
                    prediction_name = f"Models/{model_name}"
                    np.save(prediction_name+'.npy', prediction)
                    print(f"Predictions saved in: {prediction_name}")
                    return
                inp = keras.layers.Input(shape= (None, *x_train.shape[2:]))
                
                m = self.get_prediction_model(inp, self.channels)
                model = keras.models.Model(inp, m)
                #Change parameters to adquire from files
                model.compile(loss= 'mae', optimizer= self.optimizer)
                print(model.summary())
                early_stopping = EarlyStopping(monitor= 'val_loss', patience= self.early_stopping_value, restore_best_weights= True)
                #Change patience from the config file
                reduce_lr = ReduceLROnPlateau(monitor= 'val_loss', patience= 3)
                memory_monitor = MemoryMonitor()
                board = TensorBoard(log_dir= f'logs/{self.name}')
                model.fit(
                    x_train, y_train,
                    batch_size= self.batch_size,
                    epochs= self.epochs,
                    validation_data= (x_validation, y_validation),
                    callbacks= [reduce_lr, early_stopping, memory_monitor]
                )
                if display:
                    example= x_test[np.random.choice(range(len(x_test)), size= 1)[0]]
                    for _ in range(self.horizon):
                        new_prediction= model.predict(example.reshape(1, *example.shape[0:]))
                        example= np.concatenate((example[1:], new_prediction), axis= 0)
                    predictions = example[:-4]

                err = model.evaluate(x_test, y_test, batch_size= 2)
                #If a model does not adquire a minimum loss value, it will try to train other time
                if err > 0.3 and count < 2:
                    print(f"Previous values i: {i} count: {count}")
                    count += 1
                    print(f"Not accomplish enough loss value part {i} times {count}")
                    opt = input("Train has not achieve a good loss value, want to retrain? (y/n): ")
                    if opt.lower() == 'y':
                        continue
                print(f"Model error is: {err}")
                count= 0
                prediction = self.map_prediction_recursive(model, x_test, self.horizon)
                prediction_name = f"Models/{self.name}"
                model.save(prediction_name+'_'+str(i)+'.keras')
                np.save(prediction_name+'_'+str(i)+'.npy', prediction)
                print(f"Predictions stored in: {prediction_name}")
                i += 1
            actual_time = time.time() - start_time
            print(f"Codification {i}, actual time: {actual_time:.4f}")
        processing_time = time.time() - start_time
        print(f"Elapsed time: {processing_time:.4f}")