from utils.config import load_config
from config.Constants import config_path

from train import base_train, kfold_train
from inference import base_inference, kfold_inference

if __name__ == '__main__':
    path = config_path
    train_cfg, inference_cfg = load_config(path)
    
    # kfold 적용
    if train_cfg.kfold:
        kfold_train(train_cfg)
        kfold_inference(inference_cfg)
    else:
        base_train(train_cfg)
        base_inference(inference_cfg)
    
    