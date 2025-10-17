from parentheses import load_paren
import re
import sys

def make_paren_data(limit=15000):
    '''
    one time use function 
    '''
    paren_data_path = "pre-predata/shuff_dyck/dyck_sequences.txt"
    raw_paren_data = load_paren(paren_data_path)
    
    if limit is not None:
        raw_paren_data = raw_paren_data[:limit]
    
    converted_paren_data = re.sub(r'\b(\d+)\b', r'<\1>', raw_paren_data)
    with open("pre-predata/tokenized_paren/tokenized_paren.txt", "w") as file:
        file.write(converted_paren_data)
    
    count = limit if limit is not None else "all"
    print(f"Processed {count} items")

if __name__ == "__main__":
    # Use all data if no argument provided
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    make_paren_data(limit)