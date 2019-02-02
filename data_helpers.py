import re
import itertools
import codecs
from collections import Counter
import pandas as pd
from langconv import *


def clean_str(string):
  return Converter('zh-hans').convert(string)


def load_data_and_labels():
  """
  Loads MR polarity data from files, splits the data into words and generates labels.
  Returns split sentences and labels.
  """
  # Load data from files
  df = pd.DataFrame(columns=['text','class'])
  positive_examples = list(codecs.open("./chinese/pos.txt", "r", "utf-8").readlines())
  positive_examples = [s.strip() for s in positive_examples]
  df_pos = pd.DataFrame({'text':positive_examples, 'class': 'p'})

  negative_examples = list(codecs.open("./chinese/neg.txt", "r", "utf-8").readlines())
  negative_examples = [s.strip() for s in negative_examples]
  df_neg = pd.DataFrame({'text':negative_examples, 'class': 'n'})
  df = pd.concat([df_pos, df_neg])
  #df['text'].apply(clean_str)
  return df