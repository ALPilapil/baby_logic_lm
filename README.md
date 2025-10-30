# Baby Logic Language Model

An autoregressive language model trained on logic and next utterance prediction

## Description

We utilize pre pretraining on data containing a hierarchical format. After that we train the model on child directed speech from the childes data base but organized in such a way that the model learns to predict the next utterance given the last. 


## Self note
Evaluation suites currently truncated.
Training models takes hours for both prepre training and childes training.

## Instructions
From childes.train run format.py to generate raw datasets for next token, next sentence, and next utterance prediction. These will be turned into proper datasets later

Start with just data from childes and a dyck_sequences.txt file in pre-data/shuff_dyck

Generate dyck_sequences with python scripts/make_paren.py which takes an optional argument to limit the amount generated, will make tokenized_paren.txt

Modify the tokenizer with modify_tokenizer.py 

Generate childes training data with datasets with dataprep.py 

Finally, train and evaluate the model with train.py 

