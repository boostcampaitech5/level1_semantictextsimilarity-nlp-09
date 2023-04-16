import os
import json
import wandb

from constants import CONFIG

from train import base_train
from inference import base_inference
from utils.log import make_log_dirs
from utils.config import load_config
from utils.wandb import *

if __name__ == '__main__':
    ## config/config.yaml에서 파라미터 정보를 가져옵니다.
    train_config, inference_config = load_config(CONFIG.CONFIG_PATH)
    
    ## my_log 폴더를 생성하는 코드
    if not os.path.isdir(CONFIG.LOGDIR_PATH):
        os.mkdir(CONFIG.LOGDIR_NAME)
        
    folder_name = make_log_dirs(CONFIG.LOGDIR_NAME)
    
    # config에 my_log 폴더 경로 기록
    train_config.set_folder_dir(folder_name)
    inference_config.set_folder_dir(folder_name)
    
    wandb_logger, sweep_config = sweep_main(folder_name)

    ## config의 sweep 사용 여부에 따라 sweep config 설정 판단 및 sweep 실행
    if train_config.sweep:
        # sweep_config['metric'] = {'name':'val_pearson', 'goal':'maximize'}  # pearson 점수가 최대화가 되는 방향으로 학습을 진행합니다. (미션2)
        sweep_config = { 
            "method" : "random",
            "metric": {
                "goal": "minimize", # val_loss가 최소화가 되는 방향으로 진행
                "name": "val_loss"
            },
            "parameters" : {
                "batch_size": {"values": [8, 16]},
                "lr":{
                    'distribution': 'uniform',  # parameter를 설정하는 기준을 선택합니다. uniform은 연속적으로 균등한 값들을 선택합니다.
                    'min':1e-6,                 # 최소값을 설정합니다.
                    'max':1e-5                  # 최대값을 설정합니다.
                }
            }
        }
            
        sweep_id = wandb.sweep(
            sweep=sweep_config,     # config 딕셔너리를 추가합니다.
            project='boostcamp_STS'  # project의 이름을 추가합니다.
        )
        
        wandb.agent(
            sweep_id=sweep_id,      # sweep의 정보를 입력하고
            function=sweep_main,    # 해당 코드를
            project='boostcamp_STS',
            count=3                 # 총 3회 실행해봅니다.
        )
    else:
        base_train(train_config, sweep_config, wandb_logger)

    base_inference(inference_config, sweep_config, wandb_logger)
