# Baby Logic Language Model

An autoregressive language model trained on logic and next utterance prediction

## Description

We utilize pre pretraining on data containing a hierarchical format. After that we train the model on child directed speech from the childes data base but organized in such a way that the model learns to predict the next utterance given the last. 


## Self note
Evaluation suites currently truncated.
Training models takes hours for both prepre training and childes training.

## Instructions
Data prereqs:
1. a data/childes.train file 
2. a pre-predata/shuff_dyck/dyck_sequences.txt file

Generate raw datasets for next token, utterance, sentence prediction. The first being saved as a txt file and the later two as jsonl files. Do this via the following, which overwties the data previously there everytime. 
```bash
python scripts/format.py
```

Generate dyck_sequences with 
```bash
python scripts/make_paren.py 
```
which takes an optional argument to limit the amount generated, will make pre-predata/tokenized_paren.txt

Modify the tokenizer to create the paren tokenizer with 
```bash
python scripts/modify_tokenizer.py 
```

Generate childes training data with datasets with dataprep.py 
```bash
python scripts/dataprep.py
```

Generate and/or create the POS tokenizer and make the POS data with
```bash
python scripts/pos_data.py
```
modify the parameter in the main function to toggle wheather to modify tokenizer

Finally, train and evaluate the model with 
```bash
python scripts/train.py 
```

