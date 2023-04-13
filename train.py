import pandas as pd
import transformers
import torch
import pytorch_lightning as pl  # 파이토치 툴

from model import Dataloader, Dataset, KFoldDataloader, Model

def base_train(cfg):
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(cfg)
    model = Model(cfg)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(accelerator='gpu', max_epochs=cfg.max_epoch, log_every_n_steps=1)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'model.pt')

def kfold_train(cfg):
    Kmodel = Model(cfg)

    results = []
    # K fold 횟수 3
    nums_folds = 3
    split_seed = 974981

    # nums_folds는 fold의 개수, k는 k번째 fold datamodule
    for k in range(nums_folds):
        print("="*20,k+1,"/",nums_folds,"="*20, sep="\t")
        
        datamodule = KFoldDataloader(cfg, k=k, split_seed=split_seed, num_splits=nums_folds)
        datamodule.prepare_data()
        datamodule.setup()

        trainer = pl.Trainer(max_epochs=cfg.max_epoch)
        trainer.fit(model=Kmodel, datamodule=datamodule)
        
    trainer.test(model=Kmodel, datamodule=datamodule)
    # 학습이 완료된 모델을 저장합니다.
    torch.save(Kmodel, 'kfold_model.pt')