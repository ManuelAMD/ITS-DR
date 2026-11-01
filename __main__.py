from autoencoder_codification import AutoencoderCodification
from fragmentation_strategy import FragmentationStrategy
from evaluation import AutoencoderEvaluation, Evaluation, FragmentationEvaluation
from tkinter import filedialog
from pathlib import Path
import numpy as np
import os
from mapPreprocessing import Preprocessing
from tools import read_json_file

def data_loading(config_file: None):
    opt = input("Load data from images folder (1) or numpy array(2): ")
    if opt == '1':
        print("Select the dataset names csv.")
        names_files_path = filedialog.askopenfilename(title= "Select dataset names csv")
        print("Select original images")
        original_images_path = filedialog.askdirectory(title= "Select non-modified images")
        print("Select resulting folder")
        results_images_path = filedialog.askdirectory(title= "Select resulted images folder")
        p = Preprocessing(original_images_path, results_images_path)
        rows = int(input("Specify original rows size: "))
        cols = int(input("Specify original cols size: "))
        p.load_data(rows, cols, names_files_path)
        crop = input("Image cropping? (y/n): ")
        if crop.lower() == 'y':
            x1 = int(input("Specify x1 position: "))
            y1 = int(input("Specify y1 position: "))
            x2 = int(input("Specify x2 position: "))
            y2 = int(input("Specify y2 position: "))
            p.crop_images(x1, y1, x2, y2)
        masking = input("Generate map masking for non-changing zone not of interest? (y/n): ")
        if masking.lower() == 'y':
            p.map_masking(save_imgs= False, no_zone= True, display= False)
        categorize = input("Categorize maps? (y/n): ")
        if categorize.lower() == 'y':
            conf_file = read_json_file(config_file)
            p.categorize(np.array(conf_file['categories']))
        save = input("save processed data as numpy array? (y/n): ")
        if save.lower() == 'y':
            name = input("Stablish a numpy array name: ")
            print("Data stored in resulting folder as .npy")
            p.save_data_numpy_array(results_images_path+'/'+name+'.npy')
        return p
    elif opt == '2':
        print("Select the dataset numpy array file .npy")
        np_array = filedialog.askopenfilename(title= "Select dataset numpy array")
        rows = int(input("Specify data rows size: "))
        cols = int(input("Specify data cols size: "))
        channels = int(input("Specify data channels: "))
        p = Preprocessing()
        p.load_from_numpy_array(np_array, rows, cols, channels)
        return p
    else:
        print("Select a valid option")
        return None

if __name__ == '__main__':
    opt = input("Process autoencoder (1), fragmentation (2) estrategy, or model evaluation (3): ")
    preprocess = None
    if opt == '1':
        preprocess = data_loading("Autoencoder_configuration.json")
        autoencoder_prediction = AutoencoderCodification('Autoencoder_configuration.json')
        autoencoder_prediction.start_autoencoder_processing(preprocess)
    elif opt == '2':
        preprocess = data_loading("Fragmentation_strategy.json")
        fragmentation_prediction = FragmentationStrategy("Fragmentation_strategy.json")
        fragmentation_prediction.start_fragmentation_processing(preprocess)
    elif opt == '3':
        #print("Select prediction path")
        #predictions_path = filedialog.askopenfilename(title= "Select prediction path")
        predictions_filesnames = input("Specify the general training name: ")
        print("Select config file path")
        config_file_path = filedialog.askopenfilename(title= "Select config file path")
        config_file_path = os.path.basename(config_file_path)
        horizon = int(input("Specify horizon generation (default to 1): ")) or 1
        models_amount = int(input("Specify number of used model (z size): "))
        if preprocess == None:
            print("Load dataset for evaluation from the numpy array used to train the model to evaluate")
            preprocess = data_loading(config_file_path)
        sel = input("Autoencoder codififer (1) or fragmentation (2): ")
        if sel == '1':
            print("Select decoder model")
            decoder_path = filedialog.askopenfilename(title= "Select decoder model path")
            ae = AutoencoderEvaluation(predictions_filesnames, config_file_path, preprocess, horizon, models_amount, "actual_decoder", True)
            ae.begin_evaluation()
        elif sel == '2':
            frag_parts = int(input("Specify number of fragments (n): "))
            print(f"Processing {frag_parts*frag_parts} fragments")
            fe = FragmentationEvaluation(predictions_filesnames, config_file_path, preprocess, horizon, frag_parts, True)
            filter_size = int(input("Specify the largest filter size in prediction model n: "))
            fe.begin_evaluation(filter_size)

    else:
        print("Select a valid option")