from str_matbench_data_generation import get_structure_factory
from constants import des_dict, units_dict
import os
import json
from get_data import get_train_data, get_test_data
from transformers import BertTokenizerFast
import warnings
warnings.simplefilter("ignore")

max_length = 512
bert_path = 'bert-base-cased'


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


def generate_json(inputs, task, train_or_test, structure_str_format=""):
    data = []

    file_name = f"{train_or_test}_{structure_str_format}_pad_cased_inputs1.json"
    text2input = Text2Input(bert_path)

    for structure in inputs:
        new_item = (f"Please tell me {des_dict[task]} {units_dict[task]}of the following structure:"
                    + get_structure_factory(structure, structure_str_format))
        data.append(new_item)

    inputs = text2input.preprocess(data)

    with open(file_name, 'w') as file:
        json.dump(inputs, file)


def generate_md_nl():
    train_inputs, _ = get_train_data()
    test_inputs, _ = get_test_data()
    train_inputs = train_inputs.values.flatten().tolist()
    test_inputs = test_inputs.values.flatten().tolist()

    task = "matbench_mp_e_form"
    structure_str_format = "nl"

    generate_json(train_inputs, task, "train",
                  structure_str_format=structure_str_format)
    generate_json(test_inputs, task, "test",
                  structure_str_format=structure_str_format)


def main():
    generate_md_nl()


if __name__ == '__main__':
    main()
