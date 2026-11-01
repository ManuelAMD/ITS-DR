import numpy as np
import pandas as pd
from pandas import DataFrame
import keras.utils as image
import cv2

def get_names(file: str):
    names = DataFrame(pd.read_csv(file, header= None))
    #Get the first column
    return names[0]

def load_imgs(data_folder: str, names_file: str, rows: int, cols: int, channels= 1, img_type= '.png', color_mode= 'grayscale'):
    names = get_names(names_file)
    print(names)
    x = []
    for name in names:
        img = image.load_img('{}/{}{}'.format(data_folder, name, img_type), color_mode= color_mode, target_size= (rows, cols, channels))
        img = np.array(img)
        x.append(img)
    x = np.array(x)
    return x

def load_imgs_with_names(data_folder: str, names_file: str, rows: int, cols: int, invert= False, channels= 1, img_type= '.png', color_mode= 'grayscale'):
    names = names_file
    x = []
    for name in names:
        img = image.load_img('{}/{}'.format(data_folder, name), color_mode= color_mode, target_size= (rows, cols, channels))
        img = np.array(img)
        x.append(img)
    x = np.array(x)
    if invert:
        x = np.invert(x)
    return x

def load_imgs_names(data_folder: str, names_file, rows: int, cols: int, invert= False, channels= 1, img_type= '.png', color_mode= 'grayscale'):
    names = names_file
    x = []
    for name in names:
        img = image.load_img('{}/{}'.format(data_folder, name), color_mode= color_mode, target_size= (rows, cols, channels))
        img = np.array(img)
        x.append(img)
    x = np.array(x)
    return x

def to_monochromatic(img_data, min_val= 10, max_val= 255):
    x_mono = []
    for i in img_data:
        (thresh, monoImg) = cv2.threshold(i, min_val, max_val, cv2.THRESH_BINARY)
        x_mono.append(monoImg)
    x_mono = np.array(x_mono)
    return x_mono

