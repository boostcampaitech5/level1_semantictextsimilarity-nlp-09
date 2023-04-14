# my_log
import os
import os.path
import shutil
from datetime import datetime
# import argparse 사용 X
import json

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

## wandb
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            # self.predict_dataset = Dataset(predict_inputs, [])
            self.predict_dataset = Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=parser['shuffle'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 MSELoss를 호출합니다.
        self.loss_func = torch.nn.MSELoss() ## 

    def forward(self, x):
        x = self.plm(x)['logits'] # [CLS] embedding vector를 반환

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
            if len(batch) == 2:
                x, y = batch
                logits = self(x)
                return logits
            else:
                x = batch
                logits = self(x)
                return logits.squeeze()
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
  
## Sweep을 통해 실행될 학습 코드를 작성합니다.
## Sweep을 사용하지 않아도, 돌아갈 학습 코드입니다.
def sweep_main(config=None):
    now = datetime.now() ## 시작 시간
    ## my_log안에 날짜폴더를 생성, config.json파일을 복사
    folder_name = now.strftime('%Y-%m-%d-%H:%M:%S')
    folder_path = os.path.join(dir_path, folder_name)
    os.mkdir(folder_path)
    shutil.copyfile(os.path.join('code','config.json'), os.path.join(folder_path, 'config.json'))
    
    ## wandb 설정
    wandb.init(project="boostcamp_STS", config=config, name = folder_name)
    wandb_logger = WandbLogger(project="boostcamp_STS", name = folder_name) ## 실험 이름을 날짜로 지정 (my_log폴더의 날짜와 동일)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience=3, mode='min')
    if config != None:
        config = wandb.config

    # dataloader와 model을 생성합니다.
    ## 주의, sweep을 사용한다면, 해당하는 부분을 parser -> config로 바꿔 주셔야 합니다!
    ## ex) lr을 하이퍼 파라미터 튜닝을 한다면, parser['learning_rate] -> config.lr
    dataloader = Dataloader(parser['model_name'], parser['batch_size'], parser['shuffle'], parser['train_path'], parser['dev_path'], parser['test_path'], parser['predict_path'])
    # test_path로 예측이 틀린 Data확인하기 위함
    dataloader_for_test = Dataloader(parser['model_name'], parser['batch_size'], parser['shuffle'], parser['train_path'], parser['dev_path'], parser['test_path'], parser['test_path'])
    model = Model(parser['model_name'], parser['learning_rate'])
    if config==None:
        model.log("batch_size", parser['batch_size'])
    else:
        model.log("batch_size", config.batch_size)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(accelerator='gpu', max_epochs=parser['max_epoch'], log_every_n_steps=1, logger=wandb_logger, callbacks =[early_stopping], default_root_dir=folder_path)
    
    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    ## 배치로 구성된 예측값을 합칩니다.
    results = trainer.predict(model=model, datamodule=dataloader_for_test)
    test_pred = torch.cat(results)
    wrongs = []
    for i, pred in enumerate(test_pred):
        # test dataset에서 i번째에 해당하는 input값과 target값을 가져옵니다
        input_ids, target = dataloader_for_test.predict_dataset.__getitem__(i)
        # 예측값과 정답값이 다를 경우 기록합니다.
        if round(pred.item()) != target.item():
            wrongs.append([dataloader_for_test.tokenizer.decode(input_ids).replace(' [PAD]', ''), max(0.0, round(float(pred.item()))), target.item()])
    wrong_df = pd.DataFrame(wrongs, columns=['text', 'pred', 'target'])
    wrong_df.to_csv(os.path.join(folder_path, 'wrong.csv'))
    
    # 학습이 완료된 모델을 저장합니다.
    ## my_log 안에 날짜 폴더에 모델을 저장
    model_path = os.path.join(folder_path, 'model.pt')
    torch.save(model, model_path)
    
    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load(model_path)
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    ## 예측된 결과에서 음수값이 발생하는것을 확인, 최솟값을 0으로 설정합니다.
    predictions = list(max(0.0, round(float(i), 1)) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    ## output.csv는, my_log 안에 날짜 폴더에 저장됩니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv(os.path.join(folder_path, 'output.csv'), index=False) #

if __name__ == '__main__':
    ## code/config.json 파일을 수정해 주세요
    ## code/config.json에서 파라미터 정보를 가져옵니다.
    with open('./code/config.json') as f:
        parser = json.load(f)
    
    ## my_log 폴더를 생성하는 코드
    dir_path = "my_log"
    if not os.path.isdir(dir_path):
        os.mkdir("my_log")
    
    ## config.json의 sweep항목을 0이라고 설정하는경우, sweep을 사용하지 않습니다!
    if parser['sweep'] == 0:
        sweep_main()
    else:
        ## sweep_config['metric'] = {'name':'val_pearson', 'goal':'maximize'}  # pearson 점수가 최대화가 되는 방향으로 학습을 진행합니다. (미션2)
        sweep_config = { 
            "method" : "random",
            "metric": {
                "goal": "minimize", 
                "name": "val_loss"
            },
            "parameters" : {
                "batch_size": {"values": [8, 16, 32]}
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