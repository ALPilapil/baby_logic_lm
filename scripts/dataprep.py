from datasets import Dataset, DatasetDict
import re
from itertools import chain
import json

#--------------- DATA PREP ---------------#
def clean_data(text):
  '''
  given the childes.train data remove the speaker tags and anything in []
  which indicate annotations
  '''
  # Split text into lines for processing
  lines = text.strip().split('\n')
  cleaned_lines = []

  for line in lines:
      # Remove speaker tags (pattern: *SPEAKER_TAG: )
      # This handles tags like *CHI:, *MOT:, *COL:, etc.
      line = re.sub(r'\*[A-Z]+:\s*', '', line)

      # Remove bracketed content including the brackets
      # This handles both single and nested brackets
      line = re.sub(r'\[.*?\]', '', line)

      # Clean up extra whitespace
      line = re.sub(r'\s+', ' ', line).strip()

      # Only keep non-empty lines
      if line:
          cleaned_lines.append(line)

  return '\n'.join(cleaned_lines)

def chunk(examples):
    chunk_size = 1 # modify this accordingly
    input_ids = examples["input_ids"][0] # List[List], pass the inner list
    attention_mask = examples["attention_mask"][0] # List[List]
    input_ids_truncated = []
    attention_mask_truncated = []

    #slice with step_size=chunk_size
    for i in range(0,len(input_ids),chunk_size):
        chunk = input_ids[i:i+chunk_size]
        if len(chunk)==chunk_size: # drop the last chunk if not equal
            input_ids_truncated.append(chunk)
            attention_mask_truncated.append(attention_mask[i:i+chunk_size])
    examples['input_ids']=input_ids_truncated
    examples["attention_mask"]=attention_mask_truncated

    return examples

# Make samples to a size of 1024, fast for GPU
def concat(examples):
    examples["input_ids"]=[list(chain.from_iterable(examples['input_ids']))] # convert chain to list of tokens
    examples["attention_mask"]=[list(chain.from_iterable(examples['attention_mask']))] # convert chain to list of tokens
    return examples

def make_paren_data(data_path, tokenizer):
    with open(data_path, 'r') as file:
        paren_data = file.read()
    
    paren_data = paren_data.split("\n")

    # turn into a dataset
    dataset = Dataset.from_dict({"text": paren_data})
    paren_dataset = dataset.train_test_split(test_size=0.01, seed=42)


    # tokenize
    def tokenize_function(example):
        return tokenizer(example["text"])

    tokenized_paren = paren_dataset.map(tokenize_function,batched=True,remove_columns=['text'])

    # # save to disk if required (use load_from_disk latter)
    # tokenized_ds.save_to_disk('bookcorpus/tokenized_ds')

    # takes a lot of time (worth saving it to disk)
    concated_ds = tokenized_paren.map(concat,batched=True,batch_size=1,num_proc=8)

    chunked_ds = concated_ds.map(chunk,batched=True,batch_size=2,num_proc=2)

    return chunked_ds



def make_nt_data(data_path, tokenizer):
  '''
  turns the childes data into a train and eval dataset that can be passed into 
  training a model
  '''
  #--------------- DATA PREP ---------------#
  # load in the data
  with open(data_path, 'r') as file:
    raw_next_token_data = file.read()

  raw_next_token_data = raw_next_token_data[:10000]

  # clean data
  clean_next_token_data = clean_data(raw_next_token_data)
  clean_next_token_data = clean_next_token_data.split("\n")

  # turn into a dataset
  dataset = Dataset.from_dict({"text": clean_next_token_data})
  next_token_data = dataset.train_test_split(test_size=0.01, seed=42)


  # tokenize
  def tokenize_function(example):
    return tokenizer(example["text"])

  tokenized_nt_data = next_token_data.map(tokenize_function,batched=True,remove_columns=['text'])

  # # save to disk if required (use load_from_disk latter)
  # tokenized_ds.save_to_disk('bookcorpus/tokenized_ds')

  # takes a lot of time (worth saving it to disk)
  concated_ds = tokenized_nt_data.map(concat,batched=True,batch_size=1,num_proc=8)

  chunked_ds = concated_ds.map(chunk,batched=True,batch_size=2,num_proc=2)
  chunked_ds.save_to_disk('next_token/chunked_ds') # will use this latter for diff experimentation

  return chunked_ds


#-----------------------------------------------#
#----------- NEXT SENTENCE/UTTERANCE -----------#
#-----------------------------------------------#
def make_jsonl_list(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]

def make_nsp_data(file_path, nt_model, max_length, tokenizer):
  '''
  makes nsp OR nup data by making pairs out of the childes text file
  '''
  pairs = make_jsonl_list('./data/nsp_text.jsonl')
  raw = Dataset.from_list(pairs)
  ds = DatasetDict({"train": raw, "validation": raw.select(range(1))})
  tok = tokenizer

  tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad by default
  nt_model.resize_token_embeddings(len(tokenizer))

  MAX_LEN = max_length

  def build_example(ex):
    # Format: [s1] <eos> [s2] <eos>
    text = ex["s1"] + tok.eos_token + ex["s2"] + tok.eos_token
    x = tok(
        text,
        truncation=True,
        max_length=MAX_LEN,
        padding='max_length',  # Add padding to max length
        return_tensors="pt"
    )
    input_ids = x["input_ids"][0]
    attn = x["attention_mask"][0]

    # Find the FIRST eos (end of s1). We inserted one between s1 and s2.
    eos_positions = (input_ids == tok.eos_token_id).nonzero(as_tuple=False)
    if eos_positions.numel() == 0:
        # If truncated before the first EOS, skip this example by returning None-like (we'll filter later)
        return {"drop": True}

    first_eos_idx = eos_positions[0].item()

    labels = input_ids.clone()
    labels[:first_eos_idx + 1] = -100  # ignore s1 + its EOS; learn only on s2 tokens

    # Also mask padding tokens in labels
    labels[attn == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "drop": False
    }

  # preprocess, note max len is defined in build example
  proc_train = ds["train"].map(build_example)
  proc_val   = ds["validation"].map(build_example)

  # Drop any truncated/bad rows if they occurred
  proc_train = proc_train.filter(lambda ex: not ex["drop"])
  proc_val   = proc_val.filter(lambda ex: not ex["drop"])

  # Remove the helper column and original text columns
  proc_train = proc_train.remove_columns(["drop", "s1", "s2"])
  proc_val   = proc_val.remove_columns(["drop", "s1", "s2"])

  return proc_train, proc_val