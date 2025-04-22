import glob
import json
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class RGBLoader(BaseLoader):
    """The data loader for the RGB dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes dataloader.
            Args:
                name(str): name of the dataloader.
                data_path(str): path of a folder which stores frames and ecg data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 1012/
                     |      |-- data.npy
                     |   |-- 1013/
                     |      |-- data.npy
                     |...
                     |   |-- wxyz/
                     |      |-- data.npy
                -----------------
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*")
        if not data_dirs:
            print("path: ", data_path)
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search('(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']
        
        frames, signal = self.read_data(os.path.join(data_dirs[i]['path'], "data.npy"))
        frames_ts, signal_ts = self.read_timestamps(os.path.join(data_dirs[i]['path'], "data.npy"))

        labels = BaseLoader.process_resample(signal, frames_ts, signal_ts)

        frame_clips, label_clips = self.preprocess(frames, labels, config_preprocess)
        
        input_name_list, label_name_list = self.save_multi_process(frame_clips, label_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_data(file):
        """Reads a data file."""
        f=np.load(file, allow_pickle=True)
        frames = f.item().get('frames')
        signal = f.item().get('ECG')
        return np.asarray(frames), np.asarray(signal)

    @staticmethod
    def read_timestamps(file):
        """Reads a data file."""
        f=np.load(file, allow_pickle=True)
        frames = f.item().get('frames_ts')
        labels = f.item().get('ECG_ts')
        return np.asarray(frames), np.asarray(labels)