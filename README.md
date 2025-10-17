# Baby Logic Language Model

An autoregressive language model trained on logic and next utterance prediction

## Description

We utilize pre pretraining on data containing a hierarchical format. After that we train the model on child directed speech from the childes data base but organized in such a way that the model learns to predict the next utterance given the last. 


## Self note
All the functions currently are truncated

## Instructions
Generate dyck_sequences with python scripts/make_paren_data.py which takes an optional argument to limit the amount generated

