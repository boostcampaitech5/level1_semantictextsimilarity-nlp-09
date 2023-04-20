import os
import pandas as pd
import torch
import pytorch_lightning as pl

from constants import CONFIG
from model import Dataloader, Model

from model import Dataloader, Dataset, Model

def base_inference(inference_config):
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        inference_config.model_name,
        inference_config.inference.batch_size,
        inference_config.inference.shuffle,
        inference_config.train.num_workers,
        inference_config.path.train_path,
        inference_config.path.test_path,
        inference_config.path.dev_path,
        inference_config.path.predict_path,
        )
    model = Model(
        inference_config.model_name,
        inference_config.train.learning_rate,
        inference_config.train.hidden_dropout_prob,
        inference_config.train.attention_probs_dropout_prob,
        )

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator='gpu', max_epochs=inference_config.inference.max_epoch, log_every_n_steps=1)

    model_path = os.path.join(inference_config.train.folder_dir, 'best.ckpt')
    check_model = model.load_from_checkpoint(model_path)
    # model.load_state_dict(torch.load(model_path))

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    # model = torch.load(os.path.join(inference_config.folder_dir, CONFIG.MODEL_NAME))
    predictions = trainer.predict(model=check_model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    ## 예측된 결과에서 음수값이 발생하는것을 확인, 최솟값을 0으로 설정합니다.
    predictions = list(max(0.0, round(float(i), 1)) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    ## output.csv는, my_log 안에 날짜 폴더에 저장됩니다.
    output = pd.read_csv(CONFIG.SUBMISSION_PATH)
    output['target'] = predictions
    output.to_csv(os.path.join(inference_config.folder_dir, 'output.csv'), index=False) #