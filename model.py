import pandas as pd
from tqdm.auto import tqdm
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

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
    def __init__(self, model_name, batch_size, shuffle, num_workers, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, max_length=128)
        
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []   
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        try:
            data = data.drop(columns=self.delete_columns)
        except:
            # 이미 삭제할 컬럼이 제거된 상황에서 제거 작업은 패스함
            pass

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

            # 학습데이터, 검증데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)

            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, predict_targets)     # 어차피 비어있으면 빈 배열이 리턴됨

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class Model(pl.LightningModule):
    def __init__(self, model_name, loss, learning_rate, multi, hidden_dropout_prob, attention_probs_dropout_prob):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = learning_rate
        self.multi = multi

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1, hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob)
        
        # Loss 계산을 위해 사용될 MSELoss를 호출합니다.
        self.loss_func = get_loss_func(loss)
        
        # multi loss 계산에서 활용할 ratio
        self.weight_correlation = 0.7
        self.weight_mse = 0.3

    # Loss 함수 설정
    def get_loss_func(self, loss):
        if loss == "MSE":
            return torch.nn.MSELoss()  # mse Loss 값
        elif loss == "L1":
            return torch.nn.L1Loss()  # L1 Loss 값
        elif loss == "CorLoss":
            return self.correlation_loss

    def forward(self, x):
        x = self.plm(x)['logits']  # [CLS] embedding vector를 반환
        
        return x

    # 학습(Train)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        if self.multi:
            mse_loss = self.mse_loss_func(logits, y.float())
            correlation_loss = self.correlation_loss_function(logits, y.float())
            loss = (self.weight_correlation * correlation_loss) + (self.weight_mse * mse_loss)
        else:
            loss = self.mse_loss(logits, y.float())
            
        self.log("train_loss", loss)
        
        return loss
        
    # 검증(Validation)
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        if self.multi:
            mse_loss = self.mse_loss_func(logits, y.float())
            correlation_loss = self.correlation_loss_function(logits, y.float())
            loss = (self.weight_correlation * correlation_loss) + (self.weight_mse * mse_loss)
        else:
            loss = self.mse_loss(logits, y.float())
            
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        
        return loss
    
    # 테스트(Test)
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()))

        return logits

    # 예측(Prediction)
    def predict_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, _ = batch        # 기존 x, y에서 y는 사용하지 않아 무시처리 하였습니다.
            logits = self(x)
            return logits
        else:
            x = batch
            logits = self(x)
            return logits.squeeze()
    
    # 피어슨 상관계수를 사용한 loss function
    def correlation_loss_function(self, x, y):
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        x = x / (torch.sqrt(torch.sum(torch.square(x))) + 1e-5)
        y = y / (torch.sqrt(torch.sum(torch.square(y))) + 1e-5)
        corr = torch.mean(torch.mul(x, y))
        return 1 - corr

    # optimizer 속성
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer