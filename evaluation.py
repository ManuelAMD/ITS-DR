import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2
from tkinter import filedialog
from tools import read_json_file
from mapPreprocessing import Preprocessing
from color_tools import gray_quantized, recolor_greys_image
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

class Evaluation():
    def __init__(self, prediction_path= None, config_file= None, data: Preprocessing= None, horizon: int = 1, mode= 1, display= False):
        config_json = read_json_file(config_file)
        self.classes = np.array(config_json["classes"])
        self.rows = config_json["rows"]
        self.cols = config_json["cols"]
        self.prediciton_path = prediction_path
        self.horizon = horizon
        self.display = display
        self.window = config_json["window_size"] - 1
        self.predictions = np.load(f"Models/{prediction_path}.npy")
        data.create_STI_multi_output(self.window, self.horizon)
        self.x_test = np.load("Models/x_test_data.npy")
        self.y_test = np.load("Models/y_test_data.npy")
        self.begin_evaluation()

    def get_positions(self, data, rows: int, cols: int):
        elements = []
        for i in data:
            ix = int(i/cols)
            iy = int(np.round(((i/cols) - ix) * cols))
            elements.append((ix, iy))
        print("Prosiciones!!! {}, {}".format(ix, iy))

    #Crea cubos con su propia información de tamaño h
    def get_cubes(self, data, h):
        new_data = []
        for i in range(0, len(data)-h):
            new_data.append(data[i:i+h])
        new_data = np.array(new_data)
        print(new_data.shape)
        return new_data

    def recolor(self, args):
        data, pallete = args
        aux = []
        for i in data:
            res = gray_quantized(i, pallete)
            res = recolor_greys_image(res, pallete)
            aux.append(res)
        return np.array(aux)

    def evaluation(self, args):
        test, pred, naive, l_clas, classes = args
        f = np.zeros((l_clas, l_clas), dtype=np.uint64)
        n = np.zeros((l_clas, l_clas), dtype=np.uint64)
        for k in range(test.shape[0]):
            for i in range(test.shape[1]):
                for j in range(test.shape[2]):
                    pos1 = np.where(classes == test[k, i, j])[0][0]
                    pos2 = np.where(classes == pred[k, i, j])[0][0]
                    pos3 = np.where(classes == naive[k, i, j])[0][0]
                    #print(np.where(classes == y_test[e, k, i, j]))
                    #print(pos1, pos2, pos3)
                    f[pos1, pos2] += 1
                    n[pos1, pos3] += 1
        return f, n

    def multi_process_recolor(self, data, pallete):
        args = [(d, pallete) for d in data]
        num_cores = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_cores-1) as pool:
            with tqdm(total = len(data)) as progress:
                futures = []

                for img in args:
                    future = pool.submit(self.recolor, img)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)

        return np.array(results)

    def multi_process_evaluation(self, test, prediction, naive, cm_f, cm_n, l_clas, classes):
        #cm_f_aux = np.zeros((l_clas, l_clas), dtype=np.uint64)
        #cm_n_aux = np.zeros((l_clas, l_clas), dtype=np.uint64)
        cm_f_indv = []
        cm_n_indv = []
        print(cm_f)
        res_lists = zip(test, prediction, naive)
        args = [(t, p, n, l_clas, classes) for (t, p, n) in res_lists]
        num_cores = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_cores-1) as pool:
            with tqdm(total= len(test)) as progress:
                futures = []

                for eval in args:
                    future = pool.submit(self.evaluation, eval)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)
        print(len(result))
        print(result[0])
        print(result[1])
        for result in results:
            cm_f = np.add(cm_f, result[0])
            cm_n = np.add(cm_n, result[1])
            cm_f_indv.append(result[0])
            cm_n_indv.append(result[1])
        cm_f_indv = np.array(cm_f_indv)
        cm_n_indv = np.array(cm_n_indv)
        return cm_f, cm_n, cm_f_indv, cm_n_indv

    def mae_image(self, y_true, y_pred):
        err = np.sum((y_true.astype('float') - y_pred.astype('float')) ** 2)
        err /= float(y_true.shape[0] * y_true.shape[1])
        return err

    def calculate_errors(self, y_true, y_pred):
        MAE = MSE = RMSE = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                MAE += mean_absolute_error(y_true[i,j], y_pred[i,j])
                MSE += mean_squared_error(y_true[i,j], y_pred[i,j])
                RMSE += root_mean_squared_error(y_true[i,j], y_pred[i,j])

        cant = y_true.shape[0] * y_true.shape[1]
        if MAE == 0:
            MAE = 0
        else:
            MAE /= cant
        if MSE == 0:
            MSE = 0
        else:
            MSE /= cant
        if RMSE == 0:
            RMSE = 0
        else:
            RMSE /= cant
        return MAE, MSE, RMSE

    def confusion_matrix_evaluation(self, confusion_matrix):
        len_categories = len(confusion_matrix)
        TP = np.zeros(len_categories, np.uint32)
        TN = np.zeros(len_categories, np.uint32)
        FP = np.zeros(len_categories, np.uint32)
        FN = np.zeros(len_categories, np.uint32)

        for i in range(len_categories):
            for j in range (len_categories):
                if i == j:
                    TP[i] += confusion_matrix[i,j]
                    for k in range(len_categories):
                        if k != j:
                            FP[i] += confusion_matrix[k, j]
                    for k in range(i):
                        for l in range(j):
                            TN[i] += confusion_matrix[k, l]
                    for k in range(i+1, len_categories):
                        for l in range(j+1, len_categories):
                            TN[i] += confusion_matrix[k, l]
                    for k in range(i):
                        for l in range(j+1, len_categories):
                            TN[i] += confusion_matrix[k, l]
                    for k in range(i+1, len_categories):
                        for l in range(j):
                            TN[i] += confusion_matrix[k, l]
                else:
                    FN[i] += confusion_matrix[i,j]
        
        print("Verdaderos positivos", TP)
        print("Verdaderos negativos", TN)
        print("Falsos positivos", FP)
        print("Falsos negativos", FN)

        precision = np.zeros(len_categories, np.float32)
        recall = np.zeros(len_categories, np.float32)
        f1_score = np.zeros(len_categories, np.float32)
        accuracy = np.zeros(len_categories, np.float32)

        for i in range(len_categories):
            if (TP[i] + FP[i]) == 0:
                precision[i] = 0
            else:
                precision[i] = TP[i] / (TP[i] + FP[i])
            if (TP[i] + FN[i]) == 0:
                recall[i] = 0
            else:
                recall[i] = TP[i] / (TP[i] + FN[i])
            if (precision[i] + recall[i]) == 0:
                f1_score[i] = 0
            else:
                f1_score[i] = 2*((precision[i] * recall[i]) /(precision[i] + recall[i]))
            if (TP[i] + TN[i] + FP[i] + FN[i]) == 0:
                accuracy[i] = 0
            else:
                accuracy[i] = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])

        print("Precisión", precision)
        print("Recuperación", recall)
        print("Micro f1_score", f1_score)
        print("Exactitud", accuracy)

        macro_f1 = np.sum(f1_score)/len_categories

        print("Macro f1-score", macro_f1)

        return np.array([TP, TN, FP, FN]), np.array([precision, recall, f1_score, accuracy])

    def generate_differences_images(self, originals, predictions, horizon):
        print(originals.shape)
        print(predictions.shape)
        org = originals#.reshape(originals.shape[:-1])
        preds = predictions.reshape(predictions.shape[:-1])
        print(org.shape)
        print(preds.shape)
        preds = preds.astype('uint8')
        org = org.astype('uint8')
        preds = np.stack((preds,)*3, axis= -1)[-1]
        org = np.stack((org,)*3, axis= -1)[-1]
        print(org.shape)
        print(preds.shape)
        if self.display:
            plt.imshow(org[0], cmap= "gray")
            plt.show()
            plt.imshow(preds[0], cmap= "gray")
            plt.show()
        for i in range(len(org)):
            diff = 255 - cv2.absdiff(preds[i], org[i])
            res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            diff[mask != 0] = [0, 0, 255]
            result = preds[i]
            result[mask != 0] = [0, 0, 255]
            cv2.imwrite("Models/DroughtDifference_t+{}.png".format(i), result)

    def begin_evaluation(self):
        opt = input("Generate visual differences by stored arrays insted? (y/n): ")
        if opt.lower() == 'y':
            y_test = np.load(filedialog.askopenfilename(title= "Select original images array .npy"))
            data = np.load(filedialog.askopenfilename(title= "Select predictions array .npy"))
            self.generate_differences_images(y_test, data, self.horizon)
            return
        y_test = self.get_cubes(self.y_test, self.horizon)
        naive = self.x_test[: -self.horizon]
        data = self.predictions[: -self.horizon]
        if self.horizon == 1:
            data = self.predictions[1:]
        else:
            data = self.predictions[1: -(self.horizon - 1)]
        data = data[:, :self.horizon]
        aux_naive = []
        for ele in naive:
            aux = [ele[-1] for _ in range(self.horizon)]
            aux_naive.append(np.array(aux))
        naive = np.array(aux_naive)
        print(y_test.shape)
        y_test *= 255
        y_test = y_test.astype(np.uint8)
        y_test = y_test.reshape(y_test.shape[0],y_test.shape[1], y_test.shape[3], y_test.shape[4])
        data *= 255
        data = data.astype(np.uint8)
        data = data.reshape(data.shape[: -1])
        print(data.shape)
        if self.display:
            plt.imshow(y_test[0, 0], cmap= "gray")
            plt.show()
            plt.imshow(data[0, 0], cmap="gray")
            plt.show()
            plt.imshow(naive[0, 0], cmap="gray")
            plt.show()
        #Apply mask to prediction for balanced evaluation
        mask = np.load("Mask.npy")
        aux = np.array([])
        for i in data:
            aux2 = np.array([])
            for j in i:
                img_new = cv2.bitwise_and(j, j, mask= mask)
                aux2 = np.append(aux2, img_new)
            aux = np.append(aux, aux2)
        data = aux.reshape(data.shape[:]).astype(np.uint8)
        if self.display:
            plt.imshow(y_test[0, 0], cmap="gray")
            plt.show()
            plt.imshow(data[0,0], cmap="gray")
            plt.show()
            plt.imshow(naive[0,0], cmap="gray")
            plt.show()
        #colors = self.get_colors(data[-10, 0])
        print("Categorizong predictions")
        data = self.multi_process_recolor(data, self.classes)
        #colors = self.get_colors(data[-10, 0])
        s = data.shape[:]
        print(data.shape)
        data = data.reshape(s[0], s[1], s[2], s[3], 1)
        print(data.shape)
        if self.display:
            plt.imshow(y_test[0, 0], cmap="gray")
            plt.imshow(data[0, 0], cmap="gray")
            plt.imshow(naive[0, 0], cmap="gray")
            plt.show()
        y_test *= 255
        naive *= 255
        y_test = y_test.astype(np.uint8)
        naive = naive.astype(np.uint8)
        data = data.astype(np.uint8)
        np.save("Models/DiferencesOriginal"+ str(self.window) +".npy", y_test)
        np.save("Models/DiferencesNaive"+ str(self.window) +".npy", naive)
        np.save("Models/DiferencesPrediction"+ str(self.window) + self.prediciton_path[:-4] +".npy", data)
        opt = input("Generate visual differences? (y/n): ")
        if opt.lower() == 'y':
            self.generate_differences_images(y_test, data, self.horizon)

class AutoencoderEvaluation():
    def __init__(self, prediction_path, config_file, data: Preprocessing, horizon: int, cod_parts= 2, decoder_name= "actual_decoder", display= False):
        self.prediction_path = prediction_path
        self.config_file = config_file
        self.horizon = horizon
        self.cod_parts = cod_parts
        self.decoder_name = decoder_name
        self.display = display
        self.data = data

    def begin_evaluation(self):
        pp = self.prediction_path#[: -4]
        data = []
        for i in range(self.cod_parts):
            path_name = f"{pp}_{str(i)}.npy"
            prediction_coded = np.load(f"Models/{path_name}")
            data.append(prediction_coded)
            print(prediction_coded.shape)
            plt.imshow(prediction_coded[0,0], cmap= "gray")
            plt.show()

        #Image position is the most important for a correct evaluation
        data = np.array(data)
        data = np.moveaxis(data, 0, -1)
        data = data.reshape(data.shape[0], data.shape[2], data.shape[3], self.cod_parts)
        model = keras.models.load_model(f"Models/{self.decoder_name}.h5")
        print("DATA SIZE", data.shape)
        data_decoded = model.predict(data)
        print("DECODED SIZE", data_decoded.shape)
        data_decoded = data_decoded.reshape(data_decoded.shape[0], self.horizon, data_decoded.shape[1], data_decoded.shape[2], data_decoded.shape[3])
        np.save(f"Models/{pp}.npy", data_decoded)
        e = Evaluation(self.prediction_path, self.config_file, self.data, self.horizon, 1, self.display)
        
class FragmentationEvaluation():
    def __init__(self, prediction_path, config_file, data: Preprocessing, horizon: int, frag_parts= 2, display= False):
        self.prediction_path = prediction_path
        self.config_file = config_file
        self.horizon = horizon
        self.frag_parts = frag_parts
        self.display = display
        self.data = data

    def begin_evaluation(self, max_filter_size= 5):
        pp= self.prediction_path
        data= []
        for i in range(self.frag_parts*self.frag_parts):
            path_name = pp+'_'+str(i)+'.npy'
            forecast_fragment = np.load("Models/{}".format(path_name))[:,0]
            data.append(forecast_fragment)
            print(forecast_fragment.shape)

        data = self.data.fragment_reconstruction(data, self.frag_parts, max_filter_size= max_filter_size)
        data = np.array(data)
        data = data.reshape(data.shape[0], self.horizon, data.shape[1], data.shape[2], data.shape[3])
        np.save(f"Models/{pp}.npy", data)
        e = Evaluation(self.prediction_path, self.config_file, self.data, self.horizon, 1, self.display)
