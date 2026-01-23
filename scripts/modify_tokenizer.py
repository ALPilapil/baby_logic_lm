''''
Account for the parentheses language
Do this by creating a unique string in special token format, so <f{i}> for example,
Then for each of these unique strings add them to the model tokenizer
save this tokenizer somewhere
'''
from transformers import AutoTokenizer

# load in the data
def load_paren(file_path):
    '''
    Memory-efficient version: extracts unique numbers without loading entire file
    '''
    unique_nums = set()
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                tokens = line.split()
                for token in tokens:
                    try:
                        unique_nums.add(int(token))
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # Return as space-separated string to maintain compatibility
    return ' '.join(map(str, sorted(unique_nums)))


# generate unique strings for each integer
def string_conversion(paren_data):
    '''
    paren data will come in as a string of integers, we need to turn this into a 
    list of ints first, then apply set to it
    '''
    # conversion to list of ints
    list_version = paren_data.split()
    list_ints = list(map(int, list_version))

    # turn it into a set
    unique_nums = list(set(list_ints))

    # turn each unique int into a string in place
    unique_strings = []
    for num in unique_nums:
        string = f"<{num}>"
        unique_strings.append(string)
    
    return unique_strings

def paren_tokenizer(tokenizer=None):
    # load in the data
    paren_data_path = "pre-predata/shuff_dyck/dyck_sequences.txt"
    raw_paren_data = load_paren(paren_data_path)

    # generate unique strings for each integer
    unique_strings = string_conversion(raw_paren_data)

    # add new strings to tokenizer
    num = tokenizer.add_special_tokens({"additional_special_tokens": unique_strings})

    # save the tokenizer somwhere
    save_path = "tokenizers/paren_tokenizer"
    tokenizer.save_pretrained(save_path)

def main(tokenizer, unique_strings, save_dir):
    '''
    input: some tokenizer, unique strings, a place to store it
    output: modified tokenizer
    '''
    num = tokenizer.add_special_tokens({"additional_special_tokens": unique_strings})

    # save tokenizer
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    model_id = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    

