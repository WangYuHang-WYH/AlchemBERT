from str_matbench_data_generation import get_structure_factory, get_composition_factory
from constants import des_dict, units_dict
import os
import sys
import time
import gzip
import json
from tqdm import tqdm
from get_data import get_train_data, get_test_data
from pymatgen.core import Structure
import warnings
warnings.simplefilter("ignore")


def generate_json(inputs, task, fold, train_or_test, structure_str_format=""):
    data = []
    fmt = ''
    if structure_str_format != "":
        fmt = structure_str_format
        for structure in inputs:

            new_item = {
                "input": f"Please tell me {des_dict[task]} {units_dict[task]}of the following structure:" +
                         get_structure_factory(structure, structure_str_format),
            }
            data.append(new_item)

    json_str = json.dumps(data, indent=0)
    compressed_data = gzip.compress(json_str.encode('utf-8'))

    cnt = len(data)
    file_path = f'data_{fmt}_v2/{train_or_test}_{fold}_{task}_{fmt}_{cnt}.json.gz'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as file:
        file.write(compressed_data)

    print(f"Data has been successfully dumped into {file_path}.")


def generate_md_nl():
    train_inputs, _ = get_train_data()
    test_inputs, _ = get_test_data()
    print(train_inputs[0])
    print(type(train_inputs[0]))
    print(type(train_inputs))
    train_inputs = train_inputs.values.flatten().tolist()
    test_inputs = test_inputs.values.flatten().tolist()

    # train_inputs = train_inputs[:10]
    # test_inputs = test_inputs[:10]

    task = "matbench_mp_e_form"
    fold = 0
    structure_str_format = "nl"

    generate_json(train_inputs, task, fold, "Train",
                  structure_str_format=structure_str_format)
    generate_json(test_inputs, task, fold, "Test",
                  structure_str_format=structure_str_format)


def main():
    generate_md_nl()


if __name__ == '__main__':
    main()
