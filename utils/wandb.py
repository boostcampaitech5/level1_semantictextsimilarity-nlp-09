import os
from datetime import datetime

## wandb
import wandb
from pytorch_lightning.loggers import WandbLogger

from constants import WANDB
  
## Sweep을 통해 실행될 학습 코드를 작성합니다.
## Sweep을 사용하지 않아도, 돌아갈 학습 코드입니다.
def sweep_main(dir_path, config=None):
    
    ## wandb 설정
    wandb.init(project=WANDB.PROJECT_NAME, config=config, name = dir_path)
    wandb_logger = WandbLogger(project=WANDB.PROJECT_NAME, name = dir_path) ## 실험 이름을 날짜로 지정 (my_log폴더의 날짜와 동일)
    
    if config != None:
        config = wandb.config
    
    return wandb_logger, config