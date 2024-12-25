from fire import Fire
import lightning as l
import torch
from torch.utils.data import DataLoader
import get_data
import json
import sys
import pandas as pd
from bert_train import MatBert, MyDataset
from bert_train import epoch, task
torch.manual_seed(42)

print(task)

pred_gpus = False
bert_path = 'bert-base-cased'

# predictions_path = f"predictions/predictions_epoch{epoch}_pred_gpus{pred_gpus}_{task}.json"
predictions_path = "2024-12-25-alchembert-wbm-IS2RE.csv.gz"
test_pad_cased_path = "test_nl_pad_cased_inputs.json"


# %% model load

def main(
    best_epoch="485",
    val_mae="0.0674"
):
    best_model = f"epoch={best_epoch}_val_MAE={val_mae}_best_model.ckpt"
    best_model_path = f"checkpoints/model_epoch5000_{task}/{best_model}"
    test_inputs = pd.read_json(test_pad_cased_path)
    test_outputs = get_data.get_test_data(only_y=True)

    # test_inputs = test_inputs[:32]
    # test_outputs = test_outputs[:32]

    best_model = torch.load(best_model_path, weights_only=True)
    model = MatBert(bert_path, pred_gpus)
    model.load_state_dict(best_model['state_dict'])
    model.eval()

    # %% test
    trainer = l.Trainer(accelerator='gpu', devices=[7])
    print("predict start")
    test_input_ids = torch.tensor(test_inputs['input_ids'])
    test_attention_mask = torch.tensor(test_inputs['attention_mask'])
    test_outputs = torch.tensor(test_outputs.values)
    test_dataset = MyDataset(test_input_ids, test_attention_mask, test_outputs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    predictions = trainer.predict(model, test_loader)
    print("predict end")
    predictions = [tensor.cpu().item() for tensor in predictions]
    predictions = {"e_form_per_atom_alchembert": predictions}
    predictions = pd.DataFrame(predictions)
    print(predictions)
    predictions.to_csv(predictions_path, index=False, compression='gzip')
    # with open(predictions_path, 'w') as file:
    #     json.dump(predictions, file)
    print(predictions_path)


# %%
if __name__ == '__main__':
    Fire(main)
