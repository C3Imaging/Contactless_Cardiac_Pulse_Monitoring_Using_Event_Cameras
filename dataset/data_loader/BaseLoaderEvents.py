import csv
import glob
import os
import re
from math import ceil
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
from scipy.signal import savgol_filter

class BaseLoaderEvents(Dataset):
    """The base class for data loading based on pytorch Dataset.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument(
            "--cached_path", default=None, type=str)
        parser.add_argument(
            "--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, raw_data_path, config_data):
        """Inits dataloader with lists of files.

        Args:
            dataset_name(str): name of the dataloader.
            raw_data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.inputs = list()
        self.labels = list()        
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.raw_data_dirs = self.get_raw_data(self.raw_data_path)

        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)
        
        if config_data.DO_PREPROCESS:
            self.preprocess_dataset(self.raw_data_dirs, config_data.PREPROCESS, config_data.BEGIN, config_data.END)
        else:
            if not os.path.exists(self.cached_path):
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')

            self.load_preprocessed_data()
        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Returns a clip of events (ChunkSize, 9616, W, H) and it's corresponding signals(ChunkSize)."""
        
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        
        data = np.float32(data)
        label = np.float32(label)
        
        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        # split_idx represents the point in the previous filename where we want to split the string 
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        filename = item_path_filename[:split_idx]
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments, 
        # the chunk_id for example would be 0
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id

    def get_raw_data(self, raw_data_path):
        """Returns raw data directories under the path.

        Args:
            raw_data_path(str): a list of video_files.
        """
        raise Exception("'get_raw_data' Not Implemented")

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        raise Exception("'split_raw_data' Not Implemented")

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Parses and preprocesses all the raw data based on split.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)  # partition dataset 
        # send data directories to be processed
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess)
        self.build_file_list(file_list_dict)  # build file list
        self.load_preprocessed_data()  # load all data and corresponding labels (sorted for consistency)
        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def gen_voxel_rep(events, H, W):
        voxel_grid = np.zeros((H * W,), dtype=np.float32)

        if events.shape[0] == 0:
            print("No events available")
            return voxel_grid.reshape(H, W)
        
        # Boolean mask to filter x-coordinates between 280 and 999
        mask = (events[:, 1] >= 280) & (events[:, 1] < 1000)
        cropped_bin = events[mask].copy()
        
        # Adjust x-coordinates
        cropped_bin[:, 1] -= 280

        # Downsample factors
        df_x = 720 // W
        df_y = 720 // H
            
        # Convert coordinates to voxel indices
        xs = (cropped_bin[:, 1] / df_x).astype(np.uint32)
        ys = (cropped_bin[:, 2] / df_y).astype(np.uint32)
        pols = cropped_bin[:, 3].astype(np.float32)
        pols[pols == 0] = -1  # Convert 0 to -1

        index = xs + ys * W
        np.add.at(voxel_grid, index, pols)

        voxel_grid = voxel_grid.reshape(H, W)

        # Normalize and convert to uint8
        m, M = -8.0, 8.0
        voxel_grid = (M-m) * (voxel_grid - voxel_grid.min()) / (voxel_grid.max() - voxel_grid.min()) + m
        voxel_grid = (255.0 * voxel_grid).astype(np.uint8)

        return voxel_grid
        
    def gen_voxel_stream(event_stream, H, W):
        voxel_list = [BaseLoaderEvents.gen_voxel_rep(events, H, W) for events in event_stream]
        voxel_stream = np.stack(voxel_list)[:, None, :, :].astype(np.float32)
        return voxel_stream
    
    def preprocess(self, events, labels, config_preprocess):
        """Preprocesses a pair of data.

        Args:
            events(np.array): array of event series ((ev1, ev2, ..., evx), (ev1, ev2, ..., evx), ...)
            labels(np.array): processed and resampled ECG signal values
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
        Returns:
            event_clips(np.array): chunked event frames
            label_clips(np.array): chunked labels for each frame
        """
        #resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        print("preprocessing")
        
        data = list()  # Video data
        print("cropped event frame")
        
        data.append(
        BaseLoaderEvents.gen_voxel_stream(
            events, config_preprocess.H, config_preprocess.W
            )
        )
        
        data = np.concatenate(data, axis=1)  # concatenate all channels

        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            labels = BaseLoaderEvents.diff_normalize_label(labels)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            labels = BaseLoaderEvents.standardized_label(labels)
        else:
            raise ValueError("Unsupported label type!")
        
        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            event_clips, label_clips = self.chunk(data, labels, config_preprocess.CHUNK_LENGTH)
        else:
            event_clips = np.array([data])
            label_clips = np.array([labels])
            
        return event_clips, label_clips

    def chunk(self, events, labels, chunk_length):
        """Chunk the data into small chunks.

        Args:
            events(np.array): event frames
            labels(np.array): frame labels
            chunk_length(int): the length of each chunk.
        Returns:
            event_clips: all chunks of event frames
            label_clips: all chunks of frame labels
        """

        clip_num = labels.shape[0] // chunk_length
        event_clips = [events[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        label_clips = [labels[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]        
        
        return np.array(event_clips), np.array(label_clips)

    def save(self, event_clips, label_clips, filename):
        """Save all the chunked data.

        Args:
            event_clips: all chunks of event frames
            label_clips: all chunks of frame labels
            filename: name the filename
        Returns:
            count: count of preprocessed data
        """

        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        
        count = 0
        for i in range(len(label_clips)):
            assert (len(self.inputs) == len(self.labels))
            
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            
            np.save(input_path_name, event_clips[i])
            np.save(label_path_name, label_clips[i])
            
            count += 1
        return count

    def save_multi_process(self, event_clips, label_clips, filename):
        """Save all the chunked data with multi-thread processing.

        Args:
            event_clips: all chunks of event frames
            label_clips: all chunks of frame labels
            filename: name the filename
        Returns:
            input_path_name_list: list of input path names
            label_path_name_list: list of label path names
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
            
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        
        for i in range(len(label_clips)):
            assert (len(self.inputs) == len(self.labels))
            
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            
            np.save(input_path_name, event_clips[i])
            np.save(label_path_name, label_clips[i])
            
            count += 1
        return input_path_name_list, label_path_name_list

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=8):
        """Allocate dataset preprocessing across multiple processes.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(Dict): a dictionary of preprocessing configurations
            multi_process_quota(Int): max number of sub-processes to spawn for multiprocessing
        Returns:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        """
        print('Preprocessing dataset...')
        file_num = len(data_dirs)
        choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))

        # shared data resource
        manager = Manager()  # multi-process manager
        file_list_dict = manager.dict()  # dictionary for all processes to store processed files
        p_list = []  # list of processes
        running_num = 0  # number of running processes

        # in range of number of files to process
        for i in choose_range:
            process_flag = True
            while process_flag:  # ensure that every i creates a process
                if running_num < multi_process_quota:  # in case of too many processes
                    # send data to be preprocessing task
                    p = Process(target=self.preprocess_dataset_subprocess, 
                                args=(data_dirs,config_preprocess, i, file_list_dict))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        return file_list_dict

    def build_file_list(self, file_list_dict):
        """Build a list of files used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """
        #print("file_list_dict: ", file_list_dict)
        file_list = []
        # iterate through processes and add all processed file paths
        for process_num, file_paths in file_list_dict.items():
            file_list = file_list + file_paths

        if not file_list:
            #print(file_list_dict)
            raise ValueError(self.dataset_name, 'No files in file list')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """ If a file list has not already been generated for a specific data split build a list of files 
        used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """

        # Get data split based on begin and end indices.
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)

        # generate a list of unique raw-data file names
        filename_list = []
        for i in range(len(data_dirs_subset)):
            filename_list.append(data_dirs_subset[i]['index'])
        filename_list = list(set(filename_list))  # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.npy".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        
        self.inputs = inputs
        self.labels = labels
        
        self.preprocessed_data_len = len(inputs)
    
    @staticmethod
    def diff_normalize_label(label):
        #main code
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = (diff_label - np.mean(diff_label)) / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0

        return diffnormalized_label
    
    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def bin_events(events, config_preprocess):
        fps = config_preprocess.FS #get chosen FPS from config
        period = int((1/fps)*(10**6)) #calculate period

        data_events = []
        chunks = []
        
        j = 0
        ref = 0
        k=0
        
        #create event windows
        print("create event windows")
        while j<len(events) and ref<len(events):
            if events[j][0] - events[ref][0] >=period:
                chunk = events[ref:j]
                data_events.append(chunk)

                ref = j
                j+=1
            else:
                j+=1

        return data_events

    @staticmethod
    def process_resample(signal, events, signal_ts, t0):
        """Apply Pre-processing"""
        #inversion
        flipped_ecg = [-1*x for x in signal]
        #smoothing
        smoothed = savgol_filter(flipped_ecg, 101, 2, 0)
        #filtering
        [b,a] = scipy.signal.butter(1,[0.75/1000*2, 2.5/1000*2],analog=False,btype='bandpass')
        filtered=scipy.signal.filtfilt(b,a,smoothed)
        #clipping
        upper = np.percentile(filtered, 99)
        lower = np.percentile(filtered, 1)
        clipped= np.clip(filtered, lower, upper)

        #Downsample signal
        data_labels = []
        i = 0
        j = 0

        #compensating for possible 1-hour gap between sensor and camera reference due to daylight saving
        if abs(t0 - signal_ts[0])>3000:
            offset = 3600
        else:
            offset = 0

        while i<len(events):
            frame_ts = t0+(events[i][-1][0]/1e6)-offset
            
            while j<len(signal_ts) and signal_ts[j]<frame_ts:
                j+=1
                
            if j >= len(signal_ts):
                print("j exceeded signal ts len")
                j = j-1
            
            if j-1 >= 0:
                ts1 = signal_ts[j-1]
                ts2 = signal_ts[j]

                if abs(ts2 - frame_ts) < abs(ts1 - frame_ts):
                    temp_label = clipped[j]
                else:
                    temp_label = clipped[j-1]
            else:
                temp_label = clipped[j]
            
            data_labels.append(temp_label)
            i+=1

        return data_labels
