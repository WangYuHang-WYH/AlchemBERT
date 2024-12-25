from fire import Fire
import lightning as l
from transformers import BertModel, BertTokenizerFast
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn.functional as f
from transformers import BertConfig
import get_data
import json
import sys
import pandas as pd
from pymatgen.io.cif import CifFile
import os
import torch.distributed as dist
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.cuda.empty_cache()
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.set_float32_matmul_precision('high')  # 'medium'

max_length = 512
train_batch_size = 32
val_batch_size = 32
epoch = 5000
# epoch = 1

gpus = True     # gpus for training
pred_gpus = False   # gpus for prediction. True for all while False for device 0

# min_delta = 1e-4
patience = 200
log_every_n_steps = 50
save_top_k = 2
l_r = 1e-6

# task = "cif"
# task = "NL"
task = "NL"
bert_path = 'bert-base-cased'
pre_model_path = "checkpoints/model_epoch5000_NL/epoch=485_val_MAE=0.0674_best_model.ckpt"
# pre_model = torch.load(pre_model_path, weights_only=True)
if task == "cif":
    train_pad_cased_path = "train_pad_cased_inputs.json"
    test_pad_cased_path = "test_pad_cased_inputs.json"
    train_dir = "train_cif/"
    test_dir = "test_cif/"
elif task == "NL":
    train_pad_cased_path = "train_nl_pad_cased_inputs.json"
    test_pad_cased_path = "test_nl_pad_cased_inputs.json"
    train_dir = "train_nl/"
    test_dir = "test_nl/"
elif task == "NL_angle":
    train_pad_cased_path = "train_nl_angle_pad_cased_inputs.json"
    test_pad_cased_path = "test_nl_angle_pad_cased_inputs.json"
    train_dir = "train_nl_angle/"
    test_dir = "test_nl_angle/"

# predictions_path = f"predictions/predictions_batch_size{train_batch_size}_epoch{epoch}_pred_gpus{pred_gpus}_{task}.json"


# %%

class TextLoader:

    def __init__(
            self,
            train_d=train_dir,
            test_d=test_dir,
            train_num=154718,
            test_num=256963
              ):

        self.train_dir = train_d
        self.test_dir = test_d
        self.train_num = train_num
        self.test_num = test_num

    def load_train_data(self):
        train_dataset = []
        for i in range(self.train_num):
            data = CifFile.from_file(f'{self.train_dir}/{i}.cif')
            data = str(data)
            train_dataset.append(data)
        return train_dataset

    def load_test_data(self):
        test_dataset = []
        for i in range(self.test_num):
            data = CifFile.from_file(f'{self.test_dir}/{i}.cif')
            data = str(data)
            test_dataset.append(data)
        return test_dataset


# %%
class Text2Input:

    def __init__(self, b_path):

        self.tokenizer = BertTokenizerFast.from_pretrained(b_path)

    def preprocess(self, dataset):
        data = []
        for i in range(len(dataset)):
            d = self.tokenizer(
                dataset[i], padding='max_length', truncation=True, max_length=max_length
            )
            dd = {
                'input_ids': d['input_ids'],
                'attention_mask': d['attention_mask']
            }
            data.append(dd)
        return data


# %%
class MyDataset(Dataset):

    def __init__(self, input_ids, attention_mask, labels):

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]


# %%
class MatBert(l.LightningModule):

    def __init__(self, b_path, p_g):
        super(MatBert, self).__init__()
        self.bert = BertModel.from_pretrained(b_path, output_hidden_states=True)
        self.config = BertConfig.from_pretrained(bert_path)
        self.linear = nn.Linear(self.config.hidden_size, 1)
        self.pred_gpus = p_g

    def forward(self, input_ids, attention_mask):

        # _, x, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]

        y = self.linear(cls_representation).squeeze(-1)
        return y

    def training_step(self, batch):

        input_ids, attention_mask, y = batch
        input_ids.cuda()
        attention_mask.cuda()
        y.cuda()
        y_hat = self(input_ids, attention_mask)

        loss = f.mse_loss(y_hat.float(), y.float())
        self.log('train_mse_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch):

        input_ids, attention_mask, y = batch
        input_ids.cuda()
        attention_mask.cuda()
        y.cuda()
        y_hat = self(input_ids, attention_mask)

        loss = nn.functional.mse_loss(y_hat.float(), y.float())
        mae = torch.mean(torch.absolute(y_hat-y))
        self.log("val_MAE", mae, on_epoch=True, sync_dist=True)
        return {'val_loss': loss, 'val_MAE': mae}

    def predict_step(self, batch):

        input_ids, attention_mask, y = batch
        prediction = self(input_ids, attention_mask)
        if self.pred_gpus:
            result = torch.stack((prediction, y))
            predictions_gathered = [torch.zeros_like(result) for _ in range(dist.get_world_size())]
            dist.all_gather(predictions_gathered, result)
            return predictions_gathered
        else:
            return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=l_r)
        return optimizer


# %% data
def main():

    text2input = Text2Input(bert_path)
    train_outputs = get_data.get_train_data(only_y=True)
    # train_outputs = train_outputs[:32]
    # test_outputs = test_outputs[:32]
    if os.path.exists(train_pad_cased_path):
        print("file exists")
        train_inputs = pd.read_json(train_pad_cased_path)
        # train_inputs = train_inputs.iloc[:32]
        # test_inputs = test_inputs.iloc[:32]

        input_ids = torch.tensor(train_inputs['input_ids'])
        attention_mask = torch.tensor(train_inputs['attention_mask'])
        train_outputs = torch.tensor(train_outputs.values)

        print("text loaded")

    else:
        print("file not exist")
        print("downloading...")
        text_loader = TextLoader()
        train_inputs = text2input.preprocess(text_loader.load_train_data())

        with open(train_pad_cased_path, 'w') as file:
            json.dump(train_inputs, file)

        test_inputs = text2input.preprocess(text_loader.load_test_data())
        with open(test_pad_cased_path, 'w') as file:
            json.dump(test_inputs, file)
        print("file downloaded, please rerun")
        sys.exit()

    dataset = MyDataset(input_ids, attention_mask, train_outputs)
    train_set, val_set = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=2)

    # %% train

    print("model init")
    model = MatBert(bert_path, pred_gpus)
    model.cuda()
    early_stopping = EarlyStopping(
        monitor="val_MAE",
        patience=patience,
        verbose=True,
        mode="min"
    )
    check_point = ModelCheckpoint(
        monitor="val_MAE",
        save_top_k=save_top_k,
        dirpath=f"checkpoints/model_epoch{epoch}_{task}",
        filename="{epoch}_{val_MAE:.4f}_best_model",
        mode="min"
    )
    if gpus is False:
        trainer = l.Trainer(
            max_epochs=epoch,
            accelerator='gpu',
            callbacks=[check_point, early_stopping],
            log_every_n_steps=log_every_n_steps,
            devices=[0]
        )
    else:
        trainer = l.Trainer(
            max_epochs=epoch,
            accelerator='gpu',
            callbacks=[check_point, early_stopping],
            log_every_n_steps=log_every_n_steps,
            devices=[0, 1, 2, 3, 4, 5, 6],
            strategy='ddp_find_unused_parameters_true'
        )
    print("training start")
    model.train()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=pre_model_path)
    print("training end")


# %%
if __name__ == '__main__':
    Fire(main)
