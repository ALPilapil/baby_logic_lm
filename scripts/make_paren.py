from modify_tokenizer import load_paren
import re
import sys

def make_paren_data(input_path, output_path, limit=None, chunk_size=10000):
    '''
    Memory-efficient version: processes file in chunks
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        limit: Max number of lines to process (None = all)
        chunk_size: Number of lines to process at once
    '''
    lines_processed = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        buffer = []
        
        for line in infile:
            # Convert numbers to <number> format
            converted_line = re.sub(r'\b(\d+)\b', r'<\1>', line)
            buffer.append(converted_line)
            lines_processed += 1
            
            # Write buffer when it reaches chunk_size
            if len(buffer) >= chunk_size:
                outfile.writelines(buffer)
                buffer = []
                print(f"Processed {lines_processed:,} lines...", end='\r')
            
            # Stop if limit reached
            if limit is not None and lines_processed >= limit:
                break
        
        # Write remaining buffer
        if buffer:
            outfile.writelines(buffer)
    
    print(f"\n Processed {lines_processed:,} lines")
    print(f" Saved to {output_path}")

if __name__ == "__main__":
    input_path = "pre-predata/shuff_dyck/dyck_sequences.txt"
    output_path = "pre-predata/tokenized_paren/tokenized_paren.txt"
    
    # Use limit from command line or None for all data
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    make_paren_data(input_path, output_path, limit=limit)