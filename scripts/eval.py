from datasets import DatasetDict
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import math

#--------------- CN helper functions ---------------#
def read_syntax_data(filepath):
    """
    Read syntax evaluation data line by line, preserving the tab-separated format.

    Args:
        filepath (str): Path to the data file

    Returns:
        list: List of tuples (condition, sentence) where condition is the
              syntactic manipulation and sentence is the test sentence
    """
    data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Split on tab
            parts = line.split('\t')

            if len(parts) == 2:
                condition, sentence = parts
                data.append((condition.strip(), sentence.strip()))
            elif len(parts) == 1:
                # Handle lines that might be cut off (like the last line)
                print(f"Warning: Line {line_num} appears incomplete: {line}")
                condition = parts[0].strip()
                data.append((condition, ""))
            elif len(parts) == 3:
               condition = parts[0]
               sentence = parts[2]
               data.append((condition.strip(), sentence.strip()))
            else:
                print(f"Warning: Line {line_num} has unexpected format: {line}, parts: {parts}")

    return data


def read_syntax_data_as_dict(filepath):
    """
    Read syntax evaluation data and group by condition.

    Args:
        filepath (str): Path to the data file

    Returns:
        dict: Dictionary where keys are conditions and values are lists of sentences
    """
    from collections import defaultdict

    grouped_data = defaultdict(list)
    raw_data = read_syntax_data(filepath)

    for condition, sentence in raw_data:
        if sentence:  # Only add non-empty sentences
            grouped_data[condition].append(sentence)

    return dict(grouped_data)

def CN_format(scores_log):
  '''
  format the CN results into a cleaner format, simply track the rank of each number
  in the CN scores
  '''
  nested_lists = scores_log

  # Track counts: position -> rank -> count
  position_rank_counts = defaultdict(lambda: defaultdict(int))

  for sublist in nested_lists:
      # Get indices that would sort the array (smallest to largest)
      rankings = np.argsort(sublist)

      # rankings[0] is position of smallest, rankings[1] is position of 2nd smallest, etc.
      for rank, position in enumerate(rankings):
          position_rank_counts[position][rank + 1] += 1  # rank+1 for 1-indexed ranks

  result = {}
  for position in sorted(position_rank_counts.keys()):
      result[int(position)] = dict(position_rank_counts[position])

  return result

def process_blimp_score(total_score):
  '''
  input is a list of pairs of numbers
  output should be ratio of 1st vs 2nd positions
  '''
  total_amount = len(total_score)
  tally = 0

  for pair in total_score:
    if pair[0] > pair[1]:
      tally += 1

  return tally/total_amount

#------------------------------------------#
#----------- MAIN EVAL FUNCTION -----------#
#------------------------------------------#
class Evaluation():
  def __init__(self, model, tokenizer, eval_results):
    self.model = model
    self.tokenizer = tokenizer
    self.eval_results = eval_results

  def sentence_nll(self, sentence):
    inputs = self.tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        # Hugging Face returns loss = average negative log-likelihood
        nll = outputs.loss.item() * inputs["input_ids"].size(1)  # total NLL
        # avg_nll = outputs.loss.item()                           # per-token NLL
    return nll

  def CN_test(self, file_path):
    '''
    read in the batches of sentences and record the orders of NLL
    the model assigns to each index
    so the model scores type1 as 4th n times, type2 as 1st x times etc.
    '''
    # read in the data
    test_set = read_syntax_data(file_path)[:36]

     # make batches of every 12 lines
    candidates = []
    scores_log = []

    for i, pair in enumerate(test_set, 1):
      candidates.append(pair[1])

      # Process when we have 12 candidates OR at the end
      if i % 12 == 0 or i == len(test_set):
        results = map(self.sentence_nll, candidates)
        scores_log.append(list(results))

        # reset batch
        candidates = []

    result = CN_format(scores_log)
    return result

  #---------------- BLiMP ----------------#
  def run_test(self, file_path):
    """
    input: file path
    output: ratio of good vs bad sentences picked
    """
    total_score = []

    with open(file_path, 'r', encoding='utf-8') as f:
      for line_num, line in enumerate(f, 1):
        # parse each line as JSON
        data = json.loads(line.strip())

        # set the good and bad sentence
        good_sentence = data["sentence_good"]
        bad_sentence = data["sentence_bad"]

        # get score
        score_one = self.sentence_nll(good_sentence)
        score_two = self.sentence_nll(bad_sentence)

        total_score.append((score_one, score_two))

    ratio = process_blimp_score(total_score)
    return ratio

  def run_blimp(self, path):
    """
    each jsonl file in the blimp_tests folder is a test case. 67 total. all we need from each is the ratio
    of good vs bad sentences that the model chooses between. this means its the same process for each file so we just need a
    for loop and a tracker
    input: path to test case folder
    output: a json that tells us the ratio of good to bad sentences for each file
    """
    folder = Path(path)
    results = {}

    # a list of file paths
    test_files_paths = list(folder.glob("*.jsonl"))[:1]

    for file_path in test_files_paths:
      testcase = file_path.stem
      # test case name without the jsonl ending
      result = self.run_test(file_path)
      results[testcase] = result

    return results

  #---------------- CoLA ----------------#
  def run_cola_test(self, file_path):
    '''
    this is essentially a list of good and bad sentences
    we want to get the average rating of the nll test
    '''
    cola_test = pd.read_table(file_path)
    cola_test = cola_test.iloc[:,[1,3]]
    # grabs only the 01 and the actual sentence

    # add columns for ease
    cola_test.columns = ['type', 'sentence']

    # make a list of bad sentences and good sentences
    bad_sentences = cola_test[cola_test['type'] == 0]['sentence'].tolist()
    good_sentences = cola_test[cola_test['type'] == 1]['sentence'].tolist()

    # then run nll on them
    bad_results = np.mean(list(map(self.sentence_nll, bad_sentences)))
    good_results = np.mean(list(map(self.sentence_nll, good_sentences)))

    return {"bad": bad_results, "good": good_results}


  def run_cola(self, path):
    '''
    similar to blimp, folders for every test
    '''
    folder = Path(path)
    results = {}

    # a list of file paths
    test_files_paths = list(folder.glob("*.tsv"))[:1]

    for file_path in test_files_paths:
      testcase = file_path.stem
      # test case name without the jsonl ending
      result = self.run_cola_test(file_path)
      results[testcase] = result

    return results


  def eval(self):
    # perplexity
    perplexity = math.exp(self.eval_results["eval_loss"])
    self.perplexity = perplexity

    # cross entropy loss
    CEL = self.eval_results["eval_loss"]
    self.CEL = CEL

    # CN
    self.CN = self.CN_test('./evals/cn/crain-and-nakayama-breakdown.txt.data') # runs the CN test

    # BLiMP
    self.blimp = self.run_blimp('./evals/blimp_tests')

    # CoLA
    self.cola = self.run_cola('./evals/cola_public/raw')

