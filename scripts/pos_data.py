import os
from pathlib import Path
from transformers import AutoTokenizer
import nltk
from datasets import load_dataset, Dataset, DatasetDict
import modify_tokenizer


def process_text(text):
    '''
    take in a chunk of text 
    return the POS tagged version and a count for how many tokens
    '''
    string = ""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    count = len(tagged)

    for pair in tagged:
        # concatenate only the 0th element of each tuple pair and a space
        string = string + pair[1] + " "
    
    return string.strip(), count


def main(run_modify_tok=True):
    '''
    get one standard dataset from online 
    convert it into pos tags
    make sure to get the unique POS tags
    use this to modify the tokenizer and save it appropriately
    '''
    # account for modifying tokenizer
    tok_save_dir = './tokenizers/pos_tokenizer'
    unique_strings = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                    'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
                    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 
                    'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', "''"]
    
    # modify tokenizer
    if run_modify_tok:
        model_id = "EleutherAI/pythia-160m"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("\nModifying tokenizer...")
        modify_tokenizer.main(tokenizer=tokenizer, unique_strings=unique_strings, save_dir=tok_save_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tok_save_dir)
    #-------- PARAMS --------# 
    tokenizer = AutoTokenizer.from_pretrained(tok_save_dir)
    data_save_dir = './data/pos_dataset'
    target_num = 100_000_000
    ratio = 0.85
    train_threshold = int(target_num * ratio)
    max_length = 512

    # load in one chunk of the data
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True
    )


    # Store data in lists
    train_data = []
    test_data = []
    
    running_count = 0
    train_count = 0
    test_count = 0
    
    print(f"Target: {target_num} tokens (train: {train_threshold}, test: {target_num - train_threshold})")
    
    for example in dataset:
        text = example["text"]
        # convert the text into the proper pos tags
        pos_text, _ = process_text(text)
        
        # Tokenize the POS text
        tokenized = tokenizer(
            pos_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )

        count = len(tokenized['input_ids'])
        
        # Determine which split to save to
        if running_count < train_threshold:
            train_data.append({
                "input_ids": tokenized['input_ids'],
                "attention_mask": tokenized['attention_mask']
            })
            train_count += count
        elif running_count < target_num:
            test_data.append({
                "input_ids": tokenized['input_ids'],
                "attention_mask": tokenized['attention_mask']
            })
            test_count += count
        else:
            # We've hit our target
            break
        
        running_count += count
        
        # Progress update
        if running_count % 1_000_000 == 0 or running_count > target_num - 100:
            print(f"Processed {running_count}/{target_num} tokens (train: {train_count}, test: {test_count})")

    # Create Dataset objects
    print("\nCreating datasets...")
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    # Save as Hugging Face dataset
    print(f"Saving dataset to {data_save_dir}...")
    os.makedirs(data_save_dir, exist_ok=True)
    dataset_dict.save_to_disk(data_save_dir)
    
    print(f"Dataset saved! Train examples: {len(train_data)}, Test examples: {len(test_data)}")



if __name__ == "__main__":
    main()