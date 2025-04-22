""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from torch.utils.data import DataLoader

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/Events_NTSCAN.yaml", type=str, help="The name of the model.")
    '''
    Config File Options:
    Events_NTSCAN
    RGB_TSCAN
    '''
    return parser


def train_and_test(config, data_loader_dict):
    """Trains the model."""
    print("train test function")
    if config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Ntscan":
        model_trainer = trainer.NtscanTrainer.NtscanTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)

def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Ntscan":
        model_trainer = trainer.NtscanTrainer.NtscanTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)

if __name__ == "__main__":
    # parse arguments.
    print("main function")
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print(config)

    data_loader_dict = dict()
    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        # neural method dataloader
        # train_loader
        if config.TRAIN.DATA.DATASET == "EVENT_HR_TRAIN":
            train_loader = data_loader.EventLoader.EventLoader
        elif config.TRAIN.DATA.DATASET == "RGB_HR_TRAIN":
            train_loader = data_loader.RGBLoader.RGBLoader
        else:
            raise ValueError("Unsupported dataset! Currently only supporting internal dataset.")
            
        if config.TEST.USE_LAST_EPOCH:
                print("Testing uses last epoch, validation dataset is not required.")
        
        # valid_loader
        if config.VALID.DATA.DATASET == "EVENT_HR_VALID":
            valid_loader = data_loader.EventLoader.EventLoader
        elif config.VALID.DATA.DATASET == "RGB_HR_VALID":
            valid_loader = data_loader.RGBLoader.RGBLoader
        else:
            raise ValueError("Unsupported dataset! Currently only supporting internal dataset.")
        
        # test_loader
        if config.TEST.DATA.DATASET == "EVENT_HR_TEST":
            test_loader = data_loader.EventLoader.EventLoader
        elif config.TEST.DATA.DATASET == "RGB_HR_TEST":
            test_loader = data_loader.RGBLoader.RGBLoader
        else:
            raise ValueError("Unsupported dataset! Currently only supporting internal dataset.")

        if config.TRAIN.DATA.DATASET is not None and config.TRAIN.DATA.DATA_PATH:
            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA)
            
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=8,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['train'] = None
        
        if config.VALID.DATA.DATASET is not None and config.VALID.DATA.DATA_PATH:
            valid_data_loader = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA)
            
            data_loader_dict['valid'] = DataLoader(
                dataset=valid_data_loader,
                num_workers=8,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['valid'] = None
        
        if config.TEST.DATA.DATASET is not None and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=8,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['test'] = None

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test.")
    
    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !")