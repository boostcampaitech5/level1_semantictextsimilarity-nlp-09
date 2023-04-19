import yaml
import wandb
import os, random
import numpy as np
import torch

from constants import CONFIG
from constants import WANDB

from train import base_train
from sweep_train import sweep_train
from inference import base_inference
from utils.log import make_log_dirs
from utils.config import load_config, load_omegaconf
from utils.wandb import *
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
!set CUBLAS_WORKSPACE_CONFIG=:4096:8

if __name__ == '__main__':
    # config/config.yaml에서 파라미터 정보를 가져옵니다.
    # train_config, inference_config = load_config(CONFIG.CONFIG_PATH)
    config = load_omegaconf()
    
    # torch, np 설정
    SEED = config.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)   

    # my_log 폴더를 생성하는 코드
    if not os.path.isdir(CONFIG.LOGDIR_PATH):
        os.mkdir(CONFIG.LOGDIR_NAME)

    folder_name = make_log_dirs(CONFIG.LOGDIR_NAME)

    # config에 my_log 폴더 경로 기록
    config.folder_dir = folder_name

    ## config의 sweep 사용 여부에 따라 sweep config 설정 판단
    if not config.train.sweep:        
        base_train(config, folder_name)
        base_inference(config)
    else:
        sweep_config = load_config(WANDB.CONFIG_PATH)  
        sweep_id = wandb.sweep(
            sweep=sweep_config,         # config 딕셔너리를 추가합니다.
            project='boostcamp_STS'     # project의 이름을 추가합니다.
        )
        wandb.agent(
            sweep_id=sweep_id,          # sweep의 정보를 입력하고
            function=sweep_train,       # 해당 코드를
            project='boostcamp_STS',
            count=5                     # 총 n회 실행해봅니다.
        )

        base_inference(config)
        wandb.finish()