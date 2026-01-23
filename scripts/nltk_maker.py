import nltk
from nltk.corpus import treebank, brown
from transformers import AutoTokenizer
import modify_tokenizer
# nltk.download('brown')

def gen_pos_corpus(pos_corpus, overwrite=True):
    '''
    generate and save corpus in the proper format
    input: a corpora name
    output: none
    '''
    # initialize new string
    corpus_string = ""
    pos_corpus_list = pos_corpus.tagged_words()
    counter = 0

    for pair in pos_corpus_list:
        # concatenate only the 0th element of each tuple pair and a space
        corpus_string = corpus_string + pair[1] + " "
    
    # if overwrite:
        # overwrite the what's currently in the data
        # intialize a blank directory


    # save this into data


def get_unq_strs(corpus_list):
    '''
    input: list of corpus strings
    output: uniqe list of strings of POS tags
    '''
    unique_list = []

    for corpus in corpus_list:
        unique_list.extend(list({tag for _, tag in corpus.tagged_words()}))

    return unique_list


def main(run_modify_tokenizer=False):
    '''
    get NLTK data and make the same amount of POS data as there 
    is logic data, make sure to standardize everything
    have this able to run once and overwrite whatever is there for 
    easy use
    data should come in flattened tree format so that model learns
    hierarchies

    available copora: https://www.nltk.org/nltk_data/
    documentation: https://www.nltk.org/data.html, https://www.nltk.org/howto/corpus.html, https://www.nltk.org/howto/corpus.html

    '''
    #-------- PARAMS --------# 
    # list the available corpora that get imported manually
    pos_corpora = [brown]
    treebank_files = treebank.fileids() # 10% of the penn
    pos_save_dir = './data/pos_dataset'
    tree_save_dir = './data/tree_dataset'
    model_id = "EleutherAI/pythia-70m"
    new_modifier_dir = "./"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tok_save_dir = './tokenizers/pos_tokenizer'

    #----- MODIFY TOK -----#
    if run_modify_tokenizer:
        # get the model, unique strings, and a save dir
        unique_strings = get_unq_strs(pos_corpora)
        
        # filter to the ones not already in the tokenizer
        tokens_to_add = []
        for string in unique_strings:
            token_id = tokenizer.convert_tokens_to_ids(string)
            if token_id == tokenizer.unk_token_id:
                tokens_to_add.append(string)

        modify_tokenizer.main(tokenizer=tokenizer, unique_strings=unique_strings, save_dir=tok_save_dir)

    #-------- POS --------#
    # iterate through each corpus
    for pos_corpus in pos_corpora:
        gen_pos_corpus(pos_corpus)



if __name__ == "__main__":
    main(run_modify_tokenizer=False)