from fire import Fire
import lightning as l
from transformers import BertModel
import torch.nn as nn
import torch
import warnings
from torch.utils.data import DataLoader, Dataset, random_split
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn.functional as f
from transformers import BertConfig
from get_data import get_train_data
import sys
import pandas as pd
import os

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

max_length = 512
train_batch_size = 32
val_batch_size = 32
epoch = 5000
patience = 200
log_every_n_steps = 50
save_top_k = 1
l_r = 1e-5

task = "nl"
bert_path = 'bert-base-cased'

train_pad_cased_path = "train_nl_pad_cased_inputs.json"
test_pad_cased_path = "test_nl_pad_cased_inputs.json"


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
    def __init__(self, b_path):
        super(MatBert, self).__init__()
        self.bert = BertModel.from_pretrained(b_path, output_hidden_states=True)
        self.config = BertConfig.from_pretrained(bert_path)
        self.linear = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
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
        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=l_r)
        return optimizer


# %% data
def main():

    if os.path.exists(train_pad_cased_path):
        print(f"file {train_pad_cased_path} exists")
        train_inputs = pd.read_json(train_pad_cased_path)
        train_outputs = get_train_data(only_y=True)

        input_ids = torch.tensor(train_inputs['input_ids'])
        attention_mask = torch.tensor(train_inputs['attention_mask'])
        train_outputs = torch.tensor(train_outputs.values)
    else:
        warnings.warn("file doesn't exist", UserWarning)
        sys.exit()

    dataset = MyDataset(input_ids, attention_mask, train_outputs)
    train_set, val_set = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=2)

    # %% train

    model = MatBert(bert_path)
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
    trainer = l.Trainer(
        max_epochs=epoch,
        accelerator='gpu',
        callbacks=[check_point, early_stopping],
        log_every_n_steps=log_every_n_steps,
        devices=-1,
        strategy='ddp_find_unused_parameters_true'
    )
    model.train()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


# %%
if __name__ == '__main__':
    Fire(main)
