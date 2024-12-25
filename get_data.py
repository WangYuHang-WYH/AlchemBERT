import pandas as pd
from pymatgen.core import Structure


def get_test_data(only_y=False):

    id_col = "material_id"
    input_col = "initial_structure"
    target_col = "e_form_per_atom_mp2020_corrected"

    data_path = "2022-10-19-wbm-init-structs.json"
    df_wbm = pd.read_csv("2022-10-19-wbm-summary.csv")

    if only_y is False:
        df_in = pd.read_json(data_path).set_index(id_col)

        X = pd.Series([Structure.from_dict(x) for x in df_in[input_col]], index=df_in.index)
        y = pd.Series(df_wbm[target_col])

        return X[y.index], y

    else:
        y = pd.Series(df_wbm[target_col])
        return y


def get_train_data(only_y=False):

    target_col = "formation_energy_per_atom"
    input_col = "structure"
    id_col = "material_id"

    if only_y is False:

        df_cse = pd.read_json("2023-02-07-mp-computed-structure-entries.json").set_index(id_col)
        df_eng = pd.read_csv("2023-01-10-mp-energies.csv").set_index(id_col)

        X = pd.Series([Structure.from_dict(cse[input_col]) for cse in df_cse.entry], index=df_cse.index)
        y = pd.Series(df_eng[target_col], index=df_eng.index)

        return X[y.index], y

    else:
        df_eng = pd.read_csv("2023-01-10-mp-energies.csv").set_index(id_col)
        y = pd.Series(df_eng[target_col], index=df_eng.index)
        return y
