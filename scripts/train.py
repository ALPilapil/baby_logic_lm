import csv
import os
import torch
import gc
#from hf
from transformers import AutoTokenizer 
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoXForCausalLM, AutoConfig
from datasets import load_from_disk
# custom scripts
from eval import Evaluation
from collator import CustomDataCollator

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
    }
    
   # Check if file exists to determine if we need headers
   file_exists = os.path.exists(filename)
    
    # Write to CSV
   with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['task_type', 'CEL', 'perplexity', 'CN', 'BLiMP', 'CoLA']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the results
        writer.writerow(results)

   print(f"Results saved to {filename}")


class Model():
  def __init__(self, task_type, model_load_path, data_path, tokenizer, model_save_path, data_collator):
    '''
    set all the model parameters  
    '''
    self.task_type = task_type # string of the model type
    self.tokenizer = tokenizer 

    if model_load_path is None: # initialize a fresh model
       model_id = "EleutherAI/pythia-160m"
       configuration = AutoConfig.from_pretrained(model_id)
       model = GPTNeoXForCausalLM(configuration) 
       model.resize_token_embeddings(len(tokenizer))
       model.apply(model._init_weights)
       self.model = model
    else:
       model = GPTNeoXForCausalLM.from_pretrained(model_load_path)
       model.resize_token_embeddings(len(tokenizer))
       self.model = model
    
    self.data_collator = data_collator
    self.data_path = data_path
    self.model_save_path = model_save_path

  def train(self):
    '''
    train the model and save the direct results
    '''
    dataset = load_from_disk(self.data_path)
    train_dataset = dataset['train'].select(range(200))
    eval_dataset = dataset['test'].select(range(60))
    output_dir = 'pythia/standard-pythia'

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
    # run it
    eval_results = train(model=self.model,
                        tokenizer=self.tokenizer,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=self.data_collator,
                        save_model_path=self.model_save_path,
                        training_args=training_args)
    
    self.training_results = eval_results

    
  def evaluate(self, CN=True, blimp=True):
    '''
    run the evaluation on the model saved here
    ''' 
    # load trained model and evaluate
    trained_model = GPTNeoXForCausalLM.from_pretrained(self.model_save_path)
    
    print("running evaluation for: ", self.task_type)

    if not hasattr(self, 'training_results'):
      self.training_results = None
    evaluation = Evaluation(trained_model, self.tokenizer, self.training_results)
    
    evaluation.eval(CN=CN, blimp=blimp)

    self.evaluation_results = evaluation

  def save_eval(self, filename):
    '''
    save it to the csv
    '''
    save_results(self.evaluation_results, filename, self.task_type)

  def cleanup(self):
    '''
    clear model from memory (especially GPU)
    '''
    del self.model
    del self.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
  '''
  has parameters for where to save the 3 models, train them, and evaluate them
  all models use the same training arguments and evaluation
  '''
  #------------------ PARAMETERS ------------------#
  ## model paths, double as loading and saving to
  pos_model_path = './models/pythia/pos-model'
  nt_model_path = './models/pythia/nt-model'
  nsp_model_path = './models/pythia/nsp-model'
  nup_model_path = './models/pythia/nup-model'
  ## tokenizers and collators
  # for pretrianing models
  pos_tokenizer = AutoTokenizer.from_pretrained('./tokenizers/pos_tokenizer')
  pos_tokenizer.pad_token = pos_tokenizer.eos_token
  pos_data_collator = DataCollatorForLanguageModeling(pos_tokenizer, mlm=False)

  # for default and posttraining models
  default_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
  default_tokenizer.pad_token = default_tokenizer.eos_token
  default_data_collator = DataCollatorForLanguageModeling(default_tokenizer, mlm=False)
  convo_data_cllator = CustomDataCollator(default_tokenizer)

  ## define what to run here
  # 1. next word
  # 2. next word + NUP
  # 3. next word + NSP
  # 4. prepre + next word
  # 5. POS + next word

  
  # POS + next word
  ## POS
  pos = Model(task_type='pos', model_load_path=None, data_path='./data/pos_dataset', tokenizer=pos_tokenizer,
                       data_collator=pos_data_collator, model_save_path=pos_model_path)
  pos.train()
  pos.evaluate()
  pos.save_eval(filename='./training_results.csv')
  pos.cleanup()
  ## next word
  next_word = Model(task_type='next_word', model_load_path=pos_model_path, data_path='./data/nt_dataset', tokenizer=default_tokenizer, 
                    data_collator=default_data_collator, model_save_path=nt_model_path)
  next_word.train()
  next_word.evaluate()
  next_word.save_eval(filename='./training_results.csv')
  next_word.cleanup()

  # next word + NUP

  # next word + NSP


if __name__ == "__main__":
  main()