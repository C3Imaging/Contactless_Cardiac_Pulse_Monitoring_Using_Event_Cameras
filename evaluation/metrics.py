import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags

from neural_methods.loss.NegPearsonLoss import Neg_Pearson


def _reform_data_from_dict(data):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))

def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()

    #for tracking rmse in each range
    ids = [[],[],[],[],[],[]]
    rmse_s = [[],[],[],[],[],[]]

    for index in predictions.keys():
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])
        
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")
        
        gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
        gt_hr_peak, pred_hr_peak = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
        
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)

        rmse_temp = np.sqrt(np.mean(np.square(pred_hr_fft - gt_hr_fft)))
        print("Subject ID: ", index)
        print("RMSE: ", rmse_temp)
        if rmse_temp < 1:
            ids[0].append(index)
            rmse_s[0].append(rmse_temp)
        elif rmse_temp < 2:
            ids[1].append(index)
            rmse_s[1].append(rmse_temp)
        elif rmse_temp < 3:
            ids[2].append(index)
            rmse_s[2].append(rmse_temp)
        elif rmse_temp < 4:
            ids[3].append(index)
            rmse_s[3].append(rmse_temp)
        elif rmse_temp < 5:
            ids[4].append(index)
            rmse_s[4].append(rmse_temp)
        else:
            ids[5].append(index)
            rmse_s[5].append(rmse_temp)
    
    predict_hr_peak_all = np.array(predict_hr_peak_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)

    print("Subject number based on each RMSE range:")
    print("less than 1 rmse: ", len(ids[0]))
    print("less than 1 rmse ids: ", ids[0])
    print("less than 1 rmse values: ", rmse_s[0])
    
    print("1 - 2 rmse: ", len(ids[1]))
    print("1 - 2 rmse ids: ", ids[1])
    print("1 - 2 rmse values: ", rmse_s[1])
    
    print("2 - 3 rmse: ", len(ids[2]))
    print("2 - 3 rmse ids: ", ids[2])
    print("2 - 3 rmse values: ", rmse_s[2])
    
    print("3 - 4 rmse: ", len(ids[3]))
    print("3 - 4 rmse ids: ", ids[3])
    print("3 - 4 rmse values: ", rmse_s[3])
    
    print("4 - 5 rmse: ", len(ids[4]))
    print("4 - 5 rmse ids: ", ids[4])
    print("4 - 5 rmse values: ", rmse_s[4])
    
    print("greater than 5 rmse: ", len(ids[5]))
    print("greater than 5 rmse ids: ", ids[5])
    print("greater than 5 rmse values: ", rmse_s[5])
    
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        else:
            raise ValueError("Wrong Test Metric Type")

def calculate_metrics_ev(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    
    #for tracking rmse in each range
    ids = [[],[],[],[],[],[]]
    rmse_s = [[],[],[],[],[],[]]
    
    for index in predictions.keys():
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])
        
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")
        
        gt_hr_fft, pred_hr_fft = calculate_metric_per_video_ev(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
        
        gt_hr_peak, pred_hr_peak = calculate_metric_per_video_ev(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)

        rmse_temp = np.sqrt(np.mean(np.square(pred_hr_fft - gt_hr_fft)))
        print("Subject ID: ", index)
        print("RMSE: ", rmse_temp)
        if rmse_temp < 1:
            ids[0].append(index)
            rmse_s[0].append(rmse_temp)
        elif rmse_temp < 2:
            ids[1].append(index)
            rmse_s[1].append(rmse_temp)
        elif rmse_temp < 3:
            ids[2].append(index)
            rmse_s[2].append(rmse_temp)
        elif rmse_temp < 4:
            ids[3].append(index)
            rmse_s[3].append(rmse_temp)
        elif rmse_temp < 5:
            ids[4].append(index)
            rmse_s[4].append(rmse_temp)
        else:
            ids[5].append(index)
            rmse_s[5].append(rmse_temp)
    
    predict_hr_peak_all = np.array(predict_hr_peak_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)


    print("Subject number based on each RMSE range:")
    print("less than 1 rmse: ", len(ids[0]))
    print("less than 1 rmse ids: ", ids[0])
    print("less than 1 rmse values: ", rmse_s[0])
    
    print("1 - 2 rmse: ", len(ids[1]))
    print("1 - 2 rmse ids: ", ids[1])
    print("1 - 2 rmse values: ", rmse_s[1])
    
    print("2 - 3 rmse: ", len(ids[2]))
    print("2 - 3 rmse ids: ", ids[2])
    print("2 - 3 rmse values: ", rmse_s[2])
    
    print("3 - 4 rmse: ", len(ids[3]))
    print("3 - 4 rmse ids: ", ids[3])
    print("3 - 4 rmse values: ", rmse_s[3])
    
    print("4 - 5 rmse: ", len(ids[4]))
    print("4 - 5 rmse ids: ", ids[4])
    print("4 - 5 rmse values: ", rmse_s[4])
    
    print("greater than 5 rmse: ", len(ids[5]))
    print("greater than 5 rmse ids: ", ids[5])
    print("greater than 5 rmse values: ", rmse_s[5])
    
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        else:
            raise ValueError("Wrong Test Metric Type")