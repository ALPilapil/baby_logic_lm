import re
import json

def clean_data(text, remove_tag=False):
    '''
    given the childes.train data remove the speaker tags and anything in []
    '''
    # Split text into lines for processing
    lines = text.strip().split('\n')
    cleaned_lines = []
        

    if remove_tag:
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

    else:
        for line in lines:
            # Remove speaker tags (pattern: *SPEAKER_TAG: )
            # This handles tags like *CHI:, *MOT:, *COL:, etc.
            # line = re.sub(r'\*[A-Z]+:\s*', '', line)

            # Remove bracketed content including the brackets
            # This handles both single and nested brackets
            line = re.sub(r'\[.*?\]', '', line)

            # Clean up extra whitespace
            line = re.sub(r'\s+', ' ', line).strip()

            # Only keep non-empty lines
            if line:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

def nsp_modify(text):
    '''
    modify the clean text into a list of jsons
    each json should be a pair of s1, s2
    also should clean up the speaker tags
    overlap pairs such that: (s1,s2), (s2,s3), (s3,s4)
    '''
    # Split text into lines for processing
    lines = text.strip().split('\n')
    cleaned_lines = []

    for line in lines:
            
        # Remove speaker tags (patterns like "CHI:", "MOT:", "FAT:", etc.)
        # This regex removes speaker tags at the beginning of lines
        cleaned = re.sub(r'\*[A-Z]+:\s*', '', line)
        
        # Only keep non-empty sentences
        if cleaned:
            cleaned_lines.append(cleaned)
    
    # Create overlapping pairs: (s1,s2), (s2,s3), (s3,s4), etc.
    pairs = []
    for i in range(len(cleaned_lines) - 1):
        pair = {
            "s1": cleaned_lines[i],
            "s2": cleaned_lines[i + 1]
        }
        pairs.append(pair)
    
    return pairs

def nup_modify(text):
    '''
    same as above just with utterance turns instead
    '''
    # Split text into lines for processing
    lines = text.strip().split('\n')
    utterances = []
    
    current_speaker = None
    current_utterance = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Extract speaker tag (e.g., "*MOT:", "*CHI:", "*COL:")
        speaker_match = re.match(r'^\*([A-Z]{2,4}):\s*(.*)', line.strip())
        
        if speaker_match:
            speaker = speaker_match.group(1)
            content = speaker_match.group(2)
            
            # Clean up the content
            # Remove CHILDES annotations like [% comment], [= replacement], etc.
            cleaned_content = re.sub(r'\[[^\]]*\]', '', content)
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
            
            # If speaker has changed, save the previous utterance
            if current_speaker is not None and current_speaker != speaker:
                if current_utterance:  # Only save if there's content
                    utterance_text = ' '.join(current_utterance).strip()
                    if utterance_text:
                        utterances.append(utterance_text)
                current_utterance = []
            
            # Update current speaker and add content if it's not empty
            current_speaker = speaker
            if cleaned_content:
                current_utterance.append(cleaned_content)
        
        # Handle continuation lines (lines without speaker tags)
        else:
            if current_speaker is not None:
                cleaned_line = re.sub(r'\[[^\]]*\]', '', line.strip())
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
                if cleaned_line:
                    current_utterance.append(cleaned_line)
    
    # Don't forget the last utterance
    if current_speaker is not None and current_utterance:
        utterance_text = ' '.join(current_utterance).strip()
        if utterance_text:
            utterances.append(utterance_text)
    
    # Create overlapping pairs: (utterance1, utterance2), (utterance2, utterance3), etc.
    pairs = []
    for i in range(len(utterances) - 1):
        pair = {
            "s1": utterances[i],
            "s2": utterances[i + 1]
        }
        pairs.append(pair)
    
    return pairs

def write_pairs_jsonl(pairs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

def write_text(text, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

def main():
    # params

    # read in the data
    with open('./data/childes.train', 'r') as file:
        raw_next_token_data = file.read()

    raw_next_token_data = raw_next_token_data[:10000]
    clean_text = clean_data(raw_next_token_data)
    nt_text = clean_data(raw_next_token_data, remove_tag=True)
    write_text(nt_text, 'nt_text')

    nsp_text = nsp_modify(clean_text)
    write_pairs_jsonl(nsp_text, 'nsp_text')

    nup_text = nup_modify(clean_text)
    write_pairs_jsonl(nup_text, 'nup_text')

if __name__ == "__main__":
    main()