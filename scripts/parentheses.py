''''
Account for the parentheses language
Do this by creating a unique string in special token format, so <f{i}> for example,
Then for each of these unique strings add them to the model tokenizer
pass this tokenizer back
'''
from transformers import AutoTokenizer
import re

# load in the data
def load_paren(file_path):
    content = ''
    try:
        with open(file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return content


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
    raw_paren_data = raw_paren_data[:100]

    # generate unique strings for each integer
    unique_strings = string_conversion(raw_paren_data)

    # add new strings to tokenizer
    num = tokenizer.add_special_tokens({"additional_special_tokens": unique_strings})

    # convert the dataset into strings
    # converted_paren_data = re.sub(r'\b(\d+)\b', r'<\1>', raw_paren_data)
    
    return num

