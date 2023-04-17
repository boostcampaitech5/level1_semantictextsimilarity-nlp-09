# my_log
import os
import os.path
import pandas as pd
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping

from model import Dataloader, Dataset, Model
from constants import CONFIG
from utils.wandb import *


def base_train(train_config, sweep_config, logger):
    # dataloader와 model을 생성합니다.
    # 주의, sweep을 사용한다면, 해당하는 부분을 parser -> config로 바꿔 주셔야 합니다! ex) lr을 하이퍼 파라미터 튜닝을 한다면, parser['learning_rate] -> config.lr
    dataloader = Dataloader(train_config)
    model = Model(train_config)

    # log에 batch_size 기록
    if sweep_config == None:
        model.log("batch_size", train_config.batch_size)
    else:
        model.log("batch_size", sweep_config.batch_size)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    trainer = pl.Trainer(accelerator='gpu', max_epochs=train_config.max_epoch, log_every_n_steps=1,
                         logger=logger, callbacks=[early_stopping], default_root_dir=train_config.folder_dir)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 배치로 구성된 예측값을 합칩니다.
    results = trainer.predict(model=model, datamodule=dataloader)
    test_pred = torch.cat(results)

    # 테스트로 확인한 데이터 중 절댓값이 1.0 이상 차이나는 경우를 기록
    wrongs = []
    for i, pred in enumerate(test_pred):
        # test dataset에서 i번째에 해당하는 input값과 target값을 가져옵니다
        input_ids, target = dataloader.predict_dataset.__getitem__(i)
        # 예측값과 정답값이 크게 다를 경우 기록합니다.
        if abs(pred.item()-target.item()) > 1.0:
            wrongs.append([dataloader.tokenizer.decode(input_ids).replace(
                '[PAD]', '').strip(),  pred.item(), target.item()])
    wrong_df = pd.DataFrame(wrongs, columns=['text', 'pred', 'target'])
    wrong_df.to_csv(os.path.join(train_config.folder_dir, 'wrong.csv'))

    # 학습이 완료된 모델을 저장 / my_log 안에 날짜 폴더에 모델을 저장
    torch.save(model, os.path.join(train_config.folder_dir, 'model.pt'))
