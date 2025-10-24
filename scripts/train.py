import csv
import os
#from hf
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoXForCausalLM, AutoConfig
from datasets import load_from_disk
# custom scripts
from eval import Evaluation

# General function to use for all 3 models
def train(model,
          tokenizer,
          train_dataset,
          eval_dataset,
          data_collator,
          save_model_path,
          training_args,
          ):
  '''
  general purpose train function that takes a model, tokenizer, training and test dataset
  collator and a path to save the trained model along with the actual model config
  can be used to train all 3 models
  '''

  tokenizer.pad_token = tokenizer.eos_token

  trainer = Trainer(model=model,
                  args = training_args,
                  tokenizer=tokenizer,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  data_collator = data_collator)

  # start training and save it
  trainer.train()
  trainer.save_model(save_model_path)

  # eval results
  eval_results = trainer.evaluate()

  return eval_results

def save_results(evaluation, filename, task):
   '''
   given an evaluation results class object save these to a csv
   '''
   # Prepare the results data
   results = {
        'task_type': task,
        'CEL': evaluation.CEL,
        'perplexity': evaluation.perplexity,
        'CN': evaluation.CN,
        'BLiMP': evaluation.blimp,
        'CoLA': evaluation.cola
    }
    
   # Check if file exists to determine if we need headers
   file_exists = os.path.exists(filename)
    
    # Write to CSV
   with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'task_type', 'CEL', 'perplexity', 'CN', 'BLiMP', 'CoLA']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the results
        writer.writerow(results)

   print(f"Results saved to {filename}")

def main():
  '''
  has parameters for where to save the 3 models, train them, and evaluate them
  all models use the same training arguments and evaluation
  '''
  #------------------ PARAMETERS ------------------#
  # model paths
  pre_model_path = './models/pythia/pre-model'
  nt_model_path = './models/pythia/nt-model'
  nsp_model_path = './models/pythia/nsp-model'
  nup_model_path = './models/pythia/nup-model'
  model_id = "EleutherAI/pythia-70m"
  tokenizer = AutoTokenizer.from_pretrained('./tokenizers/paren_tokenizer')
  output_dir = 'pythia/standard-pythia'

  data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)

  # training arguments
  training_args = TrainingArguments(output_dir=output_dir,
                                    eval_strategy="steps",
                                    eval_steps=500,
                                    num_train_epochs=1,
                                    per_device_train_batch_size=8,
                                    per_device_eval_batch_size=8,
                                    learning_rate=2.5e-4,
                                    lr_scheduler_type='cosine',
                                    warmup_ratio=0.05,
                                    adam_beta1=0.9,
                                    adam_beta2=0.999,
                                    weight_decay=0.01,
                                    logging_strategy="steps",
                                    logging_steps = 500,
                                    save_steps=5000,
                                    save_total_limit=10,
                                    report_to='wandb',
                                  )

  #------------------ train and evaluate ------------------#
  def train_and_evaluate(task_type, max_length=256):
    """
    Train and evaluate a model for different prediction tasks.
    
    Args:
        task_type (str): 'pre-pre', 'next_token', 'next_sentence', or 'next_utterance'
        data_path (str): Path to the training data
        max_length (int): Maximum sequence length for NSP/NUP tasks

    returns the results for evaluation
    """
    if task_type == 'pre_pretrain':
       # Load randomized model for next token prediction
        configuration = AutoConfig.from_pretrained(model_id)
        model = GPTNeoXForCausalLM(configuration) 
        model.resize_token_embeddings(len(tokenizer))
        model.apply(model._init_weights)

        data_path = './pre-predata/tokenized_paren'
        save_path = pre_model_path

    elif task_type == 'next_token':
        model = GPTNeoXForCausalLM.from_pretrained(pre_model_path)
        # Generate NT data
        data_path = './data/nt_dataset'
        save_path = nt_model_path
        
    elif task_type == 'next_sentence':
        # Load pre-trained NT model for next sentence prediction
        model = GPTNeoXForCausalLM.from_pretrained(nt_model_path)
        data_path = './data/nsp_dataset'
        save_path = nsp_model_path
        
    elif task_type == 'next_utterance':
        # Load pre-trained NT model for next utterance prediction
        model = GPTNeoXForCausalLM.from_pretrained(nt_model_path)
        data_path = './data/nup_dataset'
        save_path = nup_model_path
        
    else:
        raise ValueError("task_type must be 'next_token', 'next_sentence', or 'next_utterance'")
  
    # define datasets
    dataset = load_from_disk(data_path)
    train_dataset = dataset['train'].select(range(30))
    eval_dataset = dataset['test'].select(range(30))

    # Train the model
    eval_results = train(model=model,
                        tokenizer=tokenizer,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        save_model_path=save_path,
                        training_args=training_args)
    
    # Load trained model and evaluate
    trained_model = GPTNeoXForCausalLM.from_pretrained(save_path)

    # do not evaluate anything for pre pre training
    if task_type == 'pre-pre':
       return 
    
    evaluation = Evaluation(trained_model, tokenizer, eval_results)
       
    evaluation.eval()

    return evaluation

  # Define what to run here
  pre_train = train_and_evaluate('pre_pretrain')
  next_token = train_and_evaluate('next_token')
  # next_sentence = train_and_evaluate('next_sentence')
  # next_utterance = train_and_evaluate('next_utterance')

  # save the results
  results_path = './training_results'
  save_results(next_token, results_path, 'next_token')
  # save_results(next_sentence, results_path, 'next_sentence')
  # save_results(next_utterance, results_path, 'next_utterance')


if __name__ == "__main__":
  main()
