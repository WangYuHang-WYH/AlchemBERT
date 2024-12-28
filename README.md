# AlchemBERT
We tested **BERT** on IS2RE task of Matbench Discovery.

We described the structural information in natural language and then performed a regression from 768 to 1 on BERT.

Requirments|version
------|-----
  torch|2.5.1
  lightning| 2.4.0
  transformers| 4.46.3
  pymatgen| 2024.11.13

# Usage 
1. Run load_md_data.py to exchange Structures to the input_ids and attention_mask of bert
2. Train and predict
