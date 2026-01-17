import nltk
from nltk.corpus import treebank, brown
# nltk.download('brown')

def gen_pos_corpus(pos_corpus):
    '''
    generate and save corpus in the proper format
    input: a corpora name
    output: none
    '''
    # initialize new string
    corpus_string = ""
    pos_corpus_list = pos_corpus.tagged_words()

    for pair in pos_corpus_list:
        # concatenate only the 0th element of each tuple pair and a space
        corpus_string = corpus_string + pair[1] + " "
    
    # save this into data

    print(corpus_string)

def gen_tree_corpus(tree_corpus_file):
    '''
    same as above just with processing for treebank corpus
    '''
    print(treebank.parsed_sents(tree_corpus_file))


def main():
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


    #-------- TREES -------#
    for file in treebank_files[:1]:
        # each is not of the same length
        gen_tree_corpus(file)
        print("example tree: ")
        print(treebank.parsed_sents(file))
    
    #-------- POS --------#
    print("POS tagged: ")
    # iterate through each corpus
    for pos_corpus in pos_corpora:
        gen_pos_corpus(pos_corpus)

if __name__ == "__main__":
    main()