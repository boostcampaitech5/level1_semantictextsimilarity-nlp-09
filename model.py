import torch
import torchmetrics
import transformers
import pandas as pd
import pytorch_lightning as pl
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs    # 입력 데이터
        self.targets = targets  # 정답 데이터

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
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_name, max_length=160)
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
            train_data = pd.read_csv(self.config.train_path)
            val_data = pd.read_csv(self.config.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.config.test_path)
            predict_data = pd.read_csv(self.config.predict_path)
            
            # 학습데이터 준비
            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            
            # test, predict 데이터셋 가져옴
            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.config.batch_size)

class KFoldDataloader(pl.LightningDataModule):
    def __init__(self, cfg, k: int = 1, split_seed: int = 974981, num_splits: int = 10):
        super().__init__()
        self.config = cfg
        
        self.k = k                      # 
        self.split_seed = split_seed    # 
        self.num_splits = num_splits    # 

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_name, max_length=160)
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
        inputs = self.tokenizing(data)      # point : 계속 같은 데이터를 불러오고 전처리를 진행하고 있다. 이 부분을 수정할 필요가 있을거 같다

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.config.train_path)
            val_data = pd.read_csv(self.config.dev_path)
            total_data = pd.concat([train_data,val_data], ignore_index=True)       # 2개의 데이터셋을 합침

            # 전체 데이터 준비
            total_inputs, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_inputs, total_targets)
            
            # 데이터셋 num_splits 번 fold
            kf = KFold(n_splits=self.num_splits, shuffle=self.config.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_dataset)]
            
            # k번째 fold 된 데이터셋의 index 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes] 
            self.val_dataset = [total_dataset[x] for x in val_indexes]
            
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.config.test_path)
            predict_data = pd.read_csv(self.config.predict_path)
            
            # 학습데이터 준비
            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            
            # print(predict_inputs[:5])
            
            # test, predict 데이터셋 가져옴
            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.config.batch_size)

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = cfg.model_name    # 모델명
        self.lr = cfg.learning_rate    # 학습률

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=cfg.model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    # 순전파 계산
    def forward(self, x):
        return self.plm(x)['logits']

    # Training
    def training_step(self, batch, batch_idx):
        x, y = batch    # input, target
        logits = self(x)    # output
        loss = self.loss_func(logits, y.float())        # loss 계산
        
        self.log("train_loss", loss)        # log에 저장

        return loss

    # Validation
    def validation_step(self, batch, batch_idx):
        x, y = batch    # input, target
        logits = self(x)    # output
        loss = self.loss_func(logits, y.float())        # loss 계산
        
        self.log("val_loss", loss)        # log에 저장
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    # Test
    def test_step(self, batch, batch_idx):
        x, y = batch    # input, target
        logits = self(x)    # output

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))        # log에 저장

    # Predict
    def predict_step(self, batch, batch_idx):
        x = batch    # input, target
        logits = self(x)    # output

        return logits.squeeze()     # output 결과를 얻기 위해 출력

    # Optimizer 불러오기
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer