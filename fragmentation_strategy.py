import time
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
import numpy as np
from tools import read_json_file, add_last, MemoryMonitor
from mapPreprocessing import Preprocessing
import tensorflow as tf

class FragmentationStrategy():
    def __init__(self, config_file):
        #Adapt names from the json file
        config_json = read_json_file(config_file)
        self.window= config_json['window_size']
        self.rows = config_json['rows']
        self.cols = config_json['cols']
        self.channels = config_json['channels']
        self.horizon = config_json['horizon']
        self.name = config_json['name'] + '_model_testing_{}'.format(int(time.time()))
        self.optimizer = config_json['optimizer']
        self.data_name = f'{config_json['folder_models_save']}/{config_json['folder']}.npy'
        self.early_stopping_value = config_json['deep_training_early_stopping_patience']
        self.epochs = config_json['epochs']
        self.batch_size = config_json['batch_size']

    def map_prediction_recursive(self, model: keras.Model, x_test: np.array, horizon: int):
        x_aux = x_test
        total_preds = []
        for i in range(horizon):
            predictions = model.predict(x_aux, batch_size= 2)
            total_preds.append(predictions)
            x_aux = add_last(x_aux, predictions[:])
        total_preds = np.array(total_preds)
        total_preds = np.transpose(total_preds, (1,0,2,3,4))
        return total_preds

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

    def start_fragmentation_processing(self, preprocess, load_and_predict= False, model_name= '', display= False):
        #preprocess.load_from_numpy_array(self.data_name, self.rows, self.cols, self.channels)

        x_train_frags, y_train_frags, x_validation_frags, y_validation_frags, x_test_frags, y_test_frags = preprocess.create_STI_dataset_fragmented(self.window, size= 2, max_filter_size= 5)
        start_time = time.time()
        for i in range(len(x_train_frags)):
            print(f"Processing fragment #{i}")
            x_train = x_train_frags[i]
            y_train = y_train_frags[i]
            x_validation = x_validation_frags[i]
            y_validation = y_validation_frags[i]
            x_test = x_test_frags[i]
            y_test = y_test_frags[i]
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                if load_and_predict:
                    model = keras.models.load_model(model_name)
                    err = model.evaluate(x_test, y_test, batch_size= 2)
                    print(f"Model error is: {err}")
                    prediction = self.map_prediction_recursive(model, x_test, self.horizon)
                    prediction_name = f"Models/{model_name}"
                    np.save(prediction_name+'.npy', prediction)
                    print(f"Predictions stores in {prediction_name}")
                    return
                inp = keras.layers.Input(shape= (None, *x_train.shape[2:]))
                m = self.get_prediction_model(inp, self.channels)
                model = keras.models.Model(inp, m)
                #Change parameters from a json file
                model.compile(loss= 'mae', optimizer= self.optimizer)
                print(model.summary())
                early_stopping = EarlyStopping(monitor= 'val_loss', patience= self.early_stopping_value, restore_best_weights= True)
                reduce_lr = ReduceLROnPlateau(monitor= 'val_loss', patience= 3)
                memory_monitor= MemoryMonitor()
                board = TensorBoard(log_dir= f'logs/{self.name}')
                model.fit(
                    x_train, y_train,
                    batch_size= self.batch_size,
                    epochs= self.epochs,
                    validation_data= (x_validation, y_validation),
                    callbacks= [reduce_lr, early_stopping, memory_monitor]
                )
                if display:
                    example = x_test[np.random.choice(range(len(x_test)), size= 1)[0]]
                    for _ in range(self.horizon):
                        new_prediction = model.predict(example.reshape(1, *example.shape[0:]))
                        example = np.concatenate((example[1:], new_prediction), axis= 0)
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
                prediction = self.map_prediction_recursive(model, x_test, self.horizon)
                prediction_name = f"Models/{self.name}"
                model.save(prediction_name+'_'+str(i)+'.keras')
                np.save(prediction_name+'_'+str(i)+'.npy', prediction)
                print(f"Predictions stored in: {prediction_name}")
            actual_time= time.time() - start_time
            print(f"Fragment {i}, actual time: {actual_time:.4f}")
        processing_time= time.time() - start_time
        print(f"Elapsed time: {processing_time:.4f}")
