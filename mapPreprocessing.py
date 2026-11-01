import os
import numpy as np
import matplotlib.pyplot as plt
import load_imgs as li
from PIL import Image
import cv2
from color_tools import gray_quantized, recolor_greys_image, get_colors
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import math
import multiprocessing

class Preprocessing():

    def __init__(self, original_data_path= '', result_data_path= ''):
        self.org_path = original_data_path
        self.res_path = result_data_path
        self.data = None
        if original_data_path == '':
            self.data_dirs = []
        else:
            self.data_dirs = os.listdir(self.org_path)
        self.data_loaded = False

    def load_data(self, rows: int, cols: int, names_file_path: str, color= "grayscale"):
        data = []
        if color == 'grayscale':
            channels = 1
            data = li.load_imgs(self.org_path, names_file_path, rows, cols, img_type='.png')
            
        else:
            channels = 3
            data = li.load_imgs(self.org_path, names_file_path, rows, cols, 3, color_mode= 'rgb')
        data = data.astype('float32')
        self.data = data.reshape(len(data), rows, cols, channels)
        print('Data Shape: {}'.format(self.data.shape))
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.data_loaded = True
        return True
    
    def load_from_numpy_array(self, path, rows, cols, channels):
        self.data = np.load(path)
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.data_loaded = True
        return True

    #Esta función corta y guarda las imágenes, tener cuidado de no utilizarla junto con las demás
    def crop_images(self, initial_x: int, initial_y: int, finish_x: int, finish_y: int):
        if not self.data_loaded:
            print("Primero carga los datos de las imágenes con la función interna load_data")
            return
        cropSize = (initial_x, initial_y, finish_x, finish_y)
        print(cropSize)
        for item in self.data_dirs:
            if os.path.isfile(self.org_path + '/' + item):
                im = Image.open(self.org_path + '/' + item)
                f, e = os.path.splitext(self.res_path + '/' + item)
                imageCrop = im.crop(cropSize)
                imageCrop.save(f+'.png')

    #Esta función guarda un np array como resultado
    def map_masking(self, save_imgs= False, no_zone= True, display = False):
        x = self.data.reshape(self.data.shape[0:-1])
        print(x[0,0,0])
        print(x[1000].max())
        #Crea un mapa auxiliar para tener la mascara
        aux = np.zeros((x.shape[1], x.shape[2]), dtype = x.dtype)
        print(aux.shape)
        for i in x:
            #Suma los valores de todos los mapas para obtener aquellos que se mueven
            aux += i
        #Donde los pixeles contienen información, son pixeles que cambian en el tiempo
        mascara = np.where(aux > 0, 1, 0)
        mascara = mascara.astype('uint8')
        np.save("Mask.npy", mascara)
        if display:
            plt.imshow(x[0], cmap="gray")
            plt.show()
            plt.imshow(mascara, cmap="gray")
            plt.show()
        if no_zone:
            #Asignar todos los valores 0 del mapa a 255 para activar
            x = np.where(x == 0, 255, x)
            if display:
                plt.imshow(x[0], cmap="gray")
                plt.show()
        aux = x
        index = 0
        new_array = np.array([])
        for i in x:
            img_new = cv2.bitwise_and(i, i , mask= mascara)
            new_array = np.append(new_array, img_new)
            im = Image.fromarray(img_new)
            im = im.convert('RGB')
            if save_imgs:
                im.save(self.res_path + '/{}.png'.format(self.data_dirs[index]))
                index += 1
        new_array = new_array.reshape(x.shape)
        print(new_array.shape)
        if display:
            plt.imshow(new_array[0], cmap="gray")
            plt.show()
        self.data = new_array
        np.save(self.res_path + '.npy', new_array)


    def recolor(args):
        data, pallete = args
        res = gray_quantized(data, pallete)
        res = recolor_greys_image(res, pallete)
        return np.array(res)
    
    def agroup_window(data, window):
        new_data = [data[i : window + i] for i in range(len(data) - window + 1)]
        return np.array(new_data)
    
    def create_shifted_frames(data, horizon):
        x = data[:, 0 : data.shape[1] - horizon, :, :]
        #y = data[:, 1 : data.shape[1], :, :]
        print("DATOSS {}, {}, {}".format(data.shape[1]-horizon, data.shape[1], horizon))
        y = data[:, data.shape[1]-horizon : data.shape[1], :, :]
        return x, y

    def create_shifted_frames_2(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, data.shape[1]-1, :, :]
        return x, y
    
    def categorize(self, categories, display = False):

        x = self.data.astype(np.uint8)
        #x = np.load("Models/DroughtDatasetMask.npy").astype(np.uint8)
        if display:
            fig, axes = plt.subplots(2,3, figsize= (10,8))
            choice = np.random.choice(range(len(x)), size= 1)[0]
            for idx, ax in enumerate(axes.flat):
                ax.imshow(np.squeeze(x[choice + idx]), cmap= 'gray')
                ax.set_title("Frame {}".format(idx + 1))
                ax.axis('off')
            plt.show()
        
        print("Categorias originales", categories)
        print("Colores actuales", get_colors(x[1000]))
        args = [(d, categories) for d in x]
        num_cores = multiprocessing.cpu_count()
        print("INICIANDO PROCEDIMIENTO DE CATEGORIZACIÓN")
        with ProcessPoolExecutor(max_workers= num_cores - 2) as pool:
            with tqdm( total = len(x)) as progress:
                futures = []
                for img in args:
                    future = pool.submit(Preprocessing.recolor, img)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)
        x = np.array(results)
        print("PROCESO TERMINADO")
        print("Colores resultantes", get_colors(x[1000]))
        self.data = x.astype('float32') / 255

    def save_data_numpy_array(self, path):
        np.save(path, self.data)

    def get_full_data(self):
        return self.data

    def create_STI_dataset(self, window_size= 1, train_float= 0.7, validation_float= 0.2, save_test= True):
        x = self.data
        x = Preprocessing.agroup_window(x, window_size)
        print(x.shape)
        x_train = x[: int(len(x) * train_float)]
        x_test = x[int(len(x) * train_float) :]
        div_part = (1-validation_float)
        x_validation = x_train[int(len(x_train) * div_part) :]
        x_train = x_train[: int(len(x_train) * div_part)]
        
        x_train = x_train.reshape(len(x_train), window_size, self.rows, self.cols, self.channels)
        x_validation = x_validation.reshape(len(x_validation), window_size, self.rows, self.cols, self.channels)
        x_test = x_test.reshape(len(x_test), window_size, self.rows, self.cols, self.channels)
        print("Forma de datos de entrenamiento: {}".format(x_train.shape))
        print("Forma de datos de validación: {}".format(x_validation.shape))
        print("Forma de datos de pruebas: {}".format(x_test.shape))

        x_train, y_train = Preprocessing.create_shifted_frames_2(x_train)
        x_validation, y_validation = Preprocessing.create_shifted_frames_2(x_validation)
        x_test, y_test = Preprocessing.create_shifted_frames_2(x_test)
        print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
        print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
        print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))

        #index = 0
        #for n in x_train[0]:
        #    cv2.imshow("d", n)
        #    cv2.waitKey()
        #for n in y_train[0]:
        #    cv2.imshow("f", n)
        #    cv2.waitKey()

        np.save("Models/x_test_data.npy", x_test)
        np.save("Models/y_test_data.npy", y_test)
        return x_train, y_train, x_validation, y_validation, x_test, y_test
    
    def create_STI_multi_output(self, window= 1, horizon= 1, train_float= 0.7, validation_float= 0.2, save_test= True):
        x = self.data
        #Se agrega el tamaño del horizonte y se le resta 1 (este último para evitar que exista un elemento de más y no cambiar la función original)
        window_size = window + horizon-1
        x = Preprocessing.agroup_window(x, window_size)
        print(x.shape)
        x_train = x[: int(len(x) * train_float)]
        x_test = x[int(len(x) * train_float) :]
        div_part = (1-validation_float)
        x_validation = x_train[int(len(x_train) * div_part) :]
        x_train = x_train[: int(len(x_train) * div_part)]

        x_train = x_train.reshape(len(x_train), window_size, self.rows, self.cols, self.channels)
        x_validation = x_validation.reshape(len(x_validation), window_size, self.rows, self.cols, self.channels)
        x_test = x_test.reshape(len(x_test), window_size, self.rows, self.cols, self.channels)
        print("Forma de datos de entrenamiento: {}".format(x_train.shape))
        print("Forma de datos de validación: {}".format(x_validation.shape))
        print("Forma de datos de pruebas: {}".format(x_test.shape))

        x_train, y_train = Preprocessing.create_shifted_frames(x_train, horizon)
        x_validation, y_validation = Preprocessing.create_shifted_frames(x_validation, horizon)
        x_test, y_test = Preprocessing.create_shifted_frames(x_test, horizon)
        #index = 0
        #for n in x_train[0]:
        #    cv2.imshow("d", n)
        #    cv2.waitKey()
        #for n in y_train[0]:
        #    cv2.imshow("f", n)
        #    cv2.waitKey()

        print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
        print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
        print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))

        np.save("Models/x_test_data.npy", x_test)
        np.save("Models/y_test_data.npy", y_test)
        return x_train, y_train, x_validation, y_validation, x_test, y_test

    
    def fragmented(self, frag_size, data, max_filter_size):
        print(data.shape)
        total_rows = data.shape[1]
        total_cols = data.shape[2]
        key_points_rows = []
        key_points_cols = []
        extra_pixels = math.floor(max_filter_size/2)
        #Cortes de filas
        actual_rows = total_rows
        actual_cols = total_cols
        key_row = 0
        key_col = 0
        for i in range(frag_size):
            pixels_rows_amount = math.ceil(actual_rows/(frag_size-i))
            pixels_cols_amount = math.ceil(actual_cols/(frag_size-i))
            p1_row = key_row
            p1_col = key_col
            key_row += pixels_rows_amount
            key_col += pixels_cols_amount
            if key_row >= total_rows:
                if key_row-pixels_rows_amount <= 0:
                    key_points_rows.append((p1_row, key_row))
                else:
                    key_points_rows.append((p1_row-extra_pixels, key_row))
            else:
                if key_row-pixels_rows_amount <= 0:
                    key_points_rows.append((p1_row, key_row+extra_pixels))
                else:
                    key_points_rows.append((p1_row-extra_pixels, key_row+extra_pixels))
            if key_col >= total_cols:
                if key_col-pixels_cols_amount <= 0:
                    key_points_cols.append((p1_col, key_col))
                else:
                    key_points_cols.append((p1_col-extra_pixels, key_col))
            else:
                if key_col-pixels_cols_amount <= 0:
                    key_points_cols.append((p1_col, key_col+extra_pixels))
                else:
                    key_points_cols.append((p1_col-extra_pixels, key_col+extra_pixels))
            actual_rows = total_rows - key_row
            actual_cols = total_cols - key_col
        print(key_points_rows)
        print(key_points_cols)

        sub_images = []
        for i in range(len(key_points_rows)):
            for j in range(len(key_points_cols)):
                print(key_points_rows[i][0], key_points_rows[i][1])
                print(key_points_cols[j][0], key_points_cols[j][1])
                sub_img = data[:, key_points_rows[i][0] : key_points_rows[i][1], key_points_cols[j][0] : key_points_cols[j][1]]
                plt.imshow(sub_img[0], cmap="gray")
                plt.show()
                sub_images.append(sub_img)
        #print(np.array(sub_images).shape)
        return sub_images
    
    def fragment_reconstruction(self, data, frag_size= 2, max_filter_size= 3):
        print(data[0].shape)
        index = 0
        extra_pixels = math.floor(max_filter_size/2)
        new_out = np.zeros((data[0].shape[0], (data[0].shape[1]-extra_pixels)*frag_size, (data[0].shape[2]-extra_pixels)*frag_size))
        new_out = new_out.reshape(new_out.shape[0], new_out.shape[1], new_out.shape[2], 1)
        print(new_out.shape)
        actual_data = []
        for i in range(frag_size):
            for j in range(frag_size):
                print(index)
                print(data[index].shape)
                if i == 0:
                    if j == 0:
                        actual_data = data[index][:, :-extra_pixels, :-extra_pixels]
                    elif j == frag_size-1:
                        actual_data = data[index][:, :-extra_pixels, extra_pixels:]
                    else:
                        actual_data = data[index][:, :-extra_pixels, extra_pixels:-extra_pixels]
                elif i == frag_size-1:
                    if j == 0:
                        actual_data = data[index][:, extra_pixels:, :-extra_pixels]
                    elif j == frag_size-1:
                        actual_data = data[index][:, extra_pixels:, extra_pixels:]
                    else:
                        actual_data = data[index][:, extra_pixels:, extra_pixels:-extra_pixels]
                else:
                    if j == 0:
                        actual_data = data[index][:, extra_pixels:-extra_pixels, :-extra_pixels]
                    elif j == frag_size-1:
                        actual_data = data[index][:, extra_pixels:-extra_pixels, extra_pixels:]
                    else:
                        actual_data = data[index][:, extra_pixels:-extra_pixels, extra_pixels:-extra_pixels]
                print(actual_data.shape)
                #print(actual_data[0])
                new_out[:, actual_data.shape[1]*i : (actual_data.shape[1]*i) + actual_data.shape[1],  actual_data.shape[2]*j : (actual_data.shape[2]*j) + actual_data.shape[2]] += actual_data
                index += 1
        print(new_out.shape)
        plt.imshow(new_out[0], cmap="gray")
        plt.show()
        #print(new_out[0])
        return new_out
    
    def autoencoder_codification(self, data, window_size, train_float= 0.7, validation_float= 0.2, save_test= True):
        x = data
        print(x.shape)
        
        x_train_cod = []
        x_validation_cod = []
        x_test_cod = []
        y_train_cod = []
        y_validation_cod = []
        y_test_cod = []
        sub_data = x[:,:,:,0]
        #print(sub_data.shape)
        for i in range(x.shape[-1]):
            sub_data = x[:,:,:,i]
            sub_data = sub_data.reshape(sub_data.shape[0], sub_data.shape[1], sub_data.shape[2], 1)
            print(sub_data.shape)
            x_part = Preprocessing.agroup_window(sub_data, window_size)
            print(x_part.shape)
            x_train = x_part[: int(len(x_part) * train_float)]
            x_test = x_part[int(len(x_part) * train_float) :]
            div_part = (1-validation_float)
            x_validation = x_train[int(len(x_train) * div_part) :]
            x_train = x_train[: int(len(x_train) * div_part)]
            
            x_train = x_train.reshape(len(x_train), window_size, sub_data.shape[1], sub_data.shape[2], self.channels)
            x_validation = x_validation.reshape(len(x_validation), window_size, sub_data.shape[1], sub_data.shape[2], self.channels)
            x_test = x_test.reshape(len(x_test), window_size, sub_data.shape[1], sub_data.shape[2], self.channels)
            print("Forma de datos de entrenamiento: {}".format(x_train.shape))
            print("Forma de datos de validación: {}".format(x_validation.shape))
            print("Forma de datos de pruebas: {}".format(x_test.shape))

            x_train, y_train = Preprocessing.create_shifted_frames_2(x_train)
            x_validation, y_validation = Preprocessing.create_shifted_frames_2(x_validation)
            x_test, y_test = Preprocessing.create_shifted_frames_2(x_test)
            print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
            print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
            print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))
            
            x_train_cod.append(x_train)
            y_train_cod.append(y_train)
            x_validation_cod.append(x_validation)
            y_validation_cod.append(y_validation)
            x_test_cod.append(x_test)
            y_test_cod.append(y_test)
        return x_train_cod, y_train_cod, x_validation_cod, y_validation_cod, x_test_cod, y_test_cod


    
    def create_STI_dataset_fragmented(self, window_size, train_float= 0.7, validation_float= 0.2, save_test= True, size=2, max_filter_size=3):
        x = self.data

        plt.imshow(x[0], cmap="gray")
        plt.show()

        sub_data = self.fragmented(size, x, max_filter_size)
        #result_data = self.fragment_reconstruction(sub_data, size)
        #print((x[0]==result_data[0]).all())
        x_test_frags = []
        x_train_frags = []
        x_validation_frags = []
        y_test_frags = []
        y_train_frags = []
        y_validation_frags = []
        res_rows = sub_data[0].shape[1]
        res_cols = sub_data[0].shape[2]
        
        for i in range(len(sub_data)):
            x_part = Preprocessing.agroup_window(sub_data[i], window_size)
            print(x.shape)
            x_train = x_part[: int(len(x_part) * train_float)]
            x_test = x_part[int(len(x_part) * train_float) :]
            div_part = (1-validation_float)
            x_validation = x_train[int(len(x_train) * div_part) :]
            x_train = x_train[: int(len(x_train) * div_part)]
            
            x_train = x_train.reshape(len(x_train), window_size, sub_data[i].shape[1], sub_data[i].shape[2], self.channels)
            x_validation = x_validation.reshape(len(x_validation), window_size, sub_data[i].shape[1], sub_data[i].shape[2], self.channels)
            x_test = x_test.reshape(len(x_test), window_size, sub_data[i].shape[1], sub_data[i].shape[2], self.channels)
            print("Forma de datos de entrenamiento: {}".format(x_train.shape))
            print("Forma de datos de validación: {}".format(x_validation.shape))
            print("Forma de datos de pruebas: {}".format(x_test.shape))

            x_train, y_train = Preprocessing.create_shifted_frames_2(x_train)
            x_validation, y_validation = Preprocessing.create_shifted_frames_2(x_validation)
            x_test, y_test = Preprocessing.create_shifted_frames_2(x_test)
            print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
            print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
            print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))

            #np.save("Models/x_test_data.npy", x_test)
            #np.save("Models/y_test_data.npy", y_test)
            x_train_frags.append(x_train)
            y_train_frags.append(y_train)
            x_validation_frags.append(x_validation)
            y_validation_frags.append(y_validation)
            x_test_frags.append(x_test)
            y_test_frags.append(y_test)
        return x_train_frags, y_train_frags, x_validation_frags, y_validation_frags, x_test_frags, y_test_frags