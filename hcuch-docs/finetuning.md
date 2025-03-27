# Finetuning on HCUCH data

To finetune a pretained model on HCUCH data, follow these steps:

1. Create the folder with the raw pretraining data in the folder `data/raw`.
2. Create the folder with the preprocessed  pretraining data in the folder `data/preprocessed` and add the corresponding files `dataset.json` and `splits_final.json`.
3. Create the folder with the raw HCUCH data to finetune on. Use the script `convert_hcuch_format.py`.
4. Follow the instructions on `documentation/pretraining_and_finetuning.md`. 
