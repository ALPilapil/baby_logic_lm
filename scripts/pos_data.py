import os
from pathlib import Path
from transformers import AutoTokenizer
import nltk
from datasets import load_dataset
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
    
    return string, count


def save_text(text, split='train', data_save_dir='./data/pos_dataset'):
    '''
    save the pos tags to a text file
    
    Args:
        text: POS tagged text string
        split: 'train' or 'test'
        data_save_dir: directory to save the data
    '''
    # Create directory if it doesn't exist
    os.makedirs(data_save_dir, exist_ok=True)
    
    # Define file path based on split
    file_path = os.path.join(data_save_dir, f'{split}.txt')
    
    # Append to file (create if doesn't exist)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def clear_data_files(data_save_dir='./data/pos_dataset'):
    '''
    Clear existing train/test files before starting new data collection
    '''
    os.makedirs(data_save_dir, exist_ok=True)
    
    train_path = os.path.join(data_save_dir, 'train.txt')
    test_path = os.path.join(data_save_dir, 'test.txt')
    
    # Clear files if they exist
    for path in [train_path, test_path]:
        if os.path.exists(path):
            open(path, 'w').close()
            print(f"Cleared {path}")


def main(run_modify_tok=True):
    '''
    get one standard dataset from online 
    convert it into pos tags
    make sure to get the unique POS tags
    use this to modify the tokenizer and save it appropriately
    '''
    #-------- PARAMS --------# 
    model_id = "EleutherAI/pythia-160m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tok_save_dir = './tokenizers/pos_tokenizer'
    data_save_dir = './data/pos_dataset'
    target_num = 100_000_000
    ratio = 0.85
    train_threshold = int(target_num * ratio)

    # Clear existing data files
    clear_data_files(data_save_dir)

    # load in one chunk of the data
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True
    )

    # account for modifying tokenizer
    unique_strings = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                    'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
                    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 
                    'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', "''"]

    # load in chunks of dataset at a time
    running_count = 0
    train_count = 0
    test_count = 0
    
    print(f"Target: {target_num} tokens (train: {train_threshold}, test: {target_num - train_threshold})")
    
    for example in dataset:
        text = example["text"]
        # convert the text into the proper pos tags
        pos_text, count = process_text(text)
        
        # Determine which split to save to
        if running_count < train_threshold:
            # save to training split
            save_text(pos_text, split='train', data_save_dir=data_save_dir)
            train_count += count
        elif running_count < target_num:
            # save to test split
            save_text(pos_text, split='test', data_save_dir=data_save_dir)
            test_count += count
        else:
            # We've hit our target
            break
        
        running_count += count
        
        # Progress update
        if running_count % 100 == 0 or running_count > target_num - 100:
            print(f"Processed {running_count}/{target_num} tokens (train: {train_count}, test: {test_count})")

    # modify tokenizer
    if run_modify_tok:
        print("modifying tokenizer")
        modify_tokenizer.main(tokenizer=tokenizer, unique_strings=unique_strings, save_dir=tok_save_dir)


if __name__ == "__main__":
    main()