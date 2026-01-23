from datasets import load_dataset
import re
import json
from transformers import AutoTokenizer

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

def compress_data(read_path, save_path, tokenizer):
    '''
    More scalable: loads data properly
    '''
    
    # Load with HuggingFace (more efficient)
    dataset = load_dataset('text', data_files=read_path, split='train')
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_data = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=['text'],
        keep_in_memory=False,
        desc="Tokenizing"
    )

    # Save to disk
    split = tokenized_data.train_test_split(test_size=0.1, seed=42)
    split.save_to_disk(save_path)
    print(f" Saved to {save_path}")


def make_nt_data_chunked_scalable(read_path, save_path, tokenizer, block_size=512):
    """
    Fully scalable version
    """
    
    # Load as dataset (more efficient)
    dataset = load_dataset('text', data_files=read_path, split='train')
    
    print(f"Loaded {len(dataset)} lines")
    
    # Clean and tokenize in one step
    def clean_and_tokenize(examples):
        # Clean each text
        cleaned_texts = [clean_data(text) for text in examples["text"]]
        # Tokenize
        return tokenizer(cleaned_texts, add_special_tokens=False)
    
    tokenized = dataset.map(
        clean_and_tokenize,
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=['text'],
        keep_in_memory=False,  # Important!
        desc="Cleaning and tokenizing"
    )
    
    # Group into chunks
    def group_texts(examples):
        from itertools import chain
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated['input_ids'])
        
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        result = {
            k: [concatenated[k][i:i + block_size] 
                for i in range(0, total_length, block_size)]
            for k in concatenated.keys()
        }
        return result
    
    chunked = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        keep_in_memory=False,  # Add this!
        desc=f"Grouping into {block_size}-token chunks"
    )
    
    print(f"Created {len(chunked)} chunks")
    
    split = chunked.train_test_split(test_size=0.1, seed=42)
    split.save_to_disk(save_path)
    
    return split
#-----------------------------------------------#
#----------- NEXT SENTENCE/UTTERANCE -----------#
#-----------------------------------------------#
def make_jsonl_list(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]

def make_nsp_data(file_path, tokenizer, max_length=512, test_size=0.1):
    """
    Scalable version: processes data efficiently for large datasets
    """
    from datasets import Dataset
    
    # Load pairs
    pairs = make_jsonl_list(file_path)
    dataset = Dataset.from_list(pairs)
    
    print(f"Loaded {len(dataset)} sentence pairs")
    
    # Tokenization function (NO padding here, NO tensors)
    def tokenize_pair(examples):
        """
        Process in batches without padding or converting to tensors
        """
        batch_size = len(examples['s1'])
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for i in range(batch_size):
            # Tokenize s1 and s2 separately
            s1_tokens = tokenizer(
                examples['s1'][i],
                truncation=True,
                max_length=max_length // 2,  # Leave room for s2
                add_special_tokens=False
            )
            
            s2_tokens = tokenizer(
                examples['s2'][i],
                truncation=True,
                max_length=max_length // 2,
                add_special_tokens=False
            )
            
            # Combine: [s1] <eos> [s2] <eos>
            input_ids = (
                s1_tokens['input_ids'] + 
                [tokenizer.eos_token_id] + 
                s2_tokens['input_ids'] + 
                [tokenizer.eos_token_id]
            )
            
            # Create attention mask (all 1s, no padding yet)
            attention_mask = [1] * len(input_ids)
            
            # Create labels: mask s1 and first eos, keep s2
            s1_length = len(s1_tokens['input_ids']) + 1  # +1 for eos
            labels = [-100] * s1_length + s2_tokens['input_ids'] + [tokenizer.eos_token_id]
            
            # Truncate if too long
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        
        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list
        }
    
    # Process in batches (MUCH faster)
    processed = dataset.map(
        tokenize_pair,
        batched=True,
        batch_size=1000,  # Process 1000 examples at a time
        remove_columns=['s1', 's2'],
        num_proc=1,  # Can increase if you have more CPU cores
        keep_in_memory=False,
        desc="Tokenizing pairs"
    )
    
    # Split train/test properly
    split = processed.train_test_split(test_size=test_size, seed=42)
    
    print(f"Train: {len(split['train'])} examples")
    print(f"Test: {len(split['test'])} examples")
    
    return split['train'], split['test']

def preprocess_nsp_and_save(input_file, output_dir, tokenizer, max_length=512):
    """
    Preprocess once and save to disk
    """
    train_dataset, test_dataset = make_nsp_data(
        input_file,
        tokenizer,
        max_length,
        test_size=0.1
    )
    
    # Save processed data
    from datasets import DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    dataset_dict.save_to_disk(output_dir)
    print(f"Saved to {output_dir}")
    
    return dataset_dict


if __name__ == "__main__":
  model_id = "EleutherAI/pythia-70m"
  tokenizer = AutoTokenizer.from_pretrained('./tokenizers/paren_tokenizer')
  
  paren_read_path = './pre-predata/tokenized_paren/tokenized_paren.txt'
  paren_save_path = './pre-predata/tokenized_paren'
  compress_data(read_path=paren_read_path, save_path=paren_save_path, tokenizer=tokenizer)

  # this also saves it
  nt_data = make_nt_data_chunked_scalable('./data/childes.train', './data/nt_dataset', tokenizer)

  preprocess_nsp_and_save(
    './data/nsp_text.jsonl',
    './data/nsp_dataset',
    tokenizer
  )

  preprocess_nsp_and_save(
    './data/nup_text.jsonl',
    './data/nup_dataset',
    tokenizer
  )

  
  

