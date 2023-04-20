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
            text = '[SEP]'.join([item[text_column]
                                for text_column in self.text_columns])
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding='max_length', truncation=True)
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
            # self.predict_dataset = Dataset(predict_inputs, [])
            self.predict_dataset = Dataset(
                predict_inputs, predict_targets)     # 어차피 비어있으면 빈 배열이 리턴됨

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class Model(pl.LightningModule):
    
    ## correlation loss function 함수
    def correlation_loss_function(self, x, y):
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        x = x / (torch.sqrt(torch.sum(torch.square(x))) + 1e-5)
        y = y / (torch.sqrt(torch.sum(torch.square(y))) + 1e-5)
        corr = torch.mean(torch.mul(x, y))
        return 1 - corr
        
    def __init__(self, model_name, learning_rate, hidden_dropout_prob, attention_probs_dropout_prob):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = learning_rate

        ## 누적을 위해 추가
        self.train_count = 0
        self.val_count = 0
        self.acc_loss = 0
        self.acc_val_loss = 0
        
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            hidden_dropout_prob=hidden_dropout_prob, 
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            output_hidden_states=True, ## 추가
            # num_labels=1
        ) 
        
        ## 추가할 모델 정의
        # 마지막 1부분을 수정하면, linear모델도 쌓을 수 있습니다 self.linear = torch.nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size)
        self.linear = torch.nn.Linear(self.plm.config.hidden_size, 1)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM 참고
        # num_layers로 layer수를 지정해 주세요!
        self.lstm = torch.nn.LSTM(self.plm.config.hidden_size, self.plm.config.hidden_size, num_layers=2, bidirectional=False, batch_first=True)
        
        self.dropout = torch.nn.Dropout(p=0.1)
        
        # Loss 계산을 위해 사용될 MSELoss를 호출합니다.
        self.mse_loss_func = torch.nn.MSELoss()  # mse Loss 값
        self.mse_loss_l1 = torch.nn.L1Loss()  # L1 Loss 값
        
        ## loss 추가
        # self.correlation_loss_function = correlation_loss_function()

    def forward(self, x):
        ## 마지막에 linear 모델 추가
        plm_outputs = self.plm(x)
        plm_logits = plm_outputs.logits # [CLS] embedding vector를 반환
        last_hidden_state = plm_outputs.hidden_states[-1]
        
        ## 이부분을 선택해서 변경해 주셔야 합니다!
        ## lstm 예시
        lstm_outputs, _ = self.lstm(last_hidden_state)
        latm_last_hidden_state = lstm_outputs[:,-1,:]
        output = self.linear(latm_last_hidden_state)
        return output

    def training_step(self, batch, batch_idx):
        
        if self.train_count == 0:
            self.acc_loss = 0
        
        x, y = batch
        plm_outputs = self.plm(x)
        plm_logits = plm_outputs.logits # [CLS] embedding vector를 반환
        origin_last_hidden_state = plm_outputs.hidden_states[-1]
        
        ## dropout augmentation 적용
        dropout_last_hidden_state = self.dropout(origin_last_hidden_state)
        last_hidden_state = torch.cat((origin_last_hidden_state, dropout_last_hidden_state), 0)
        
        lstm_outputs, _ = self.lstm(last_hidden_state)
        lstm_last_hidden_state = lstm_outputs[:,-1,:]
        logits = self.linear(lstm_last_hidden_state)
        
        y = torch.cat((y,y),0)
        
        mse_loss = self.mse_loss_func(logits, y.float())
        correlation_loss = self.correlation_loss_function(logits, y.float())
        loss = (0.6 * correlation_loss) + (0.4 * mse_loss)
        
        self.acc_loss += loss
        
        ## 누적 loss적용
        self.train_count+=1
        
        if self.train_count == 4:
            self.train_count =0
            self.log("train_loss", self.acc_loss/4)
            return self.acc_loss/4
        else:
            return None
        

    def validation_step(self, batch, batch_idx):
        
        if self.val_count == 0:
            self.acc_val_loss = 0
        
        x, y = batch
        logits = self(x)
        
        mse_loss = self.mse_loss_func(logits, y.float())
        correlation_loss = self.correlation_loss_function(logits, y.float())
        loss = (0.7 * correlation_loss) + (0.3 * mse_loss)

        self.acc_val_loss += loss
        self.val_count += 1
        
        if self.val_count == 4:
            self.val_count = 0
            self.log("val_loss", self.acc_val_loss/4)
            self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()))
            return self.acc_val_loss/4
        else:
            return None

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()))

        return logits

    def predict_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, _ = batch        # 기존 x, y에서 y는 사용하지 않아 무시처리 하였습니다.
            logits = self(x)
            return logits
        else:
            x = batch
            logits = self(x)
            return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
