import torch

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token = tokenizer.eos_token
    
    def __call__(self, examples):
        '''
        input is a list of examples from the dataset
        output is a dict of examples
        '''
        max_length = max(len(f['input_ids']) for f in examples)

        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        pad_token_id = self.tokenizer.pad_token_id
        
        for feature in examples:
            padding_length = max_length - len(feature['input_ids'])
            
            # Pad everything
            batch['input_ids'].append(
                feature['input_ids'] + [pad_token_id] * padding_length
            )
            batch['attention_mask'].append(
                feature['attention_mask'] + [0] * padding_length
            )
            batch['labels'].append(
                feature['labels'] + [-100] * padding_length
            )
        
        # Convert to tensors
        return {
            'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(batch['labels'], dtype=torch.long)
        }
