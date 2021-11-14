import math

from pyparsing import Empty
from utils import *

# Question 1
def getPrior(data):
  """
  calculer la probabilité a priori de la classe 1, ainsi que l'intervalle de confiance à 95% pour l'estimation de cette probabilité.
  """
  temp = []
  for item in data['target']:
    if item == 1:
      temp.append(item)
  mu = len(temp)/len(data)
  # x^2 = x => E(x^2) = E(x)
  variance = mu - mu * mu
  sigma = math.sqrt(variance)
  amplitude = 1.96 * sigma / math.sqrt(len(data))

  return {'estimation': mu, 
          'min5pourcent': mu - amplitude, 
          'max5pourcent': mu + amplitude}

# Question 2

class APrioriClassifier(AbstractClassifier):

  def estimClass(self, data):
    if data is None:
      return
    prior = getPrior(data)
    if prior['estimation'] > 0.5:
      return 1
    return 0
    
  def statsOnDF(self, df):
    if df is None:
      return
    VP = VN = FP = FN = 0
    for t in df.itertuples():
      dic=t._asdict()
      if self.__class__.__name__ == 'APrioriClassifier':
        predict = self.estimClass(df)
      else:
        predict = self.estimClass(dic)
      if dic['target'] == 1:
        if predict == 1:
          VP += 1  # target=1 && predict=1
        else:
          FN += 1  # target=1 && predict=0
      else:
        if predict == 1:
          FP += 1  # target=0 && predict=1
        else:
          VN += 1  # target=0 && predict=0
    return {'VP': VP, 'VN': VN, 'FP': FP, 'FN': FN, 
            'Précision': VP / (VP + FP),
            'Rappel': VP / (VP + FN)}

# Question 3

def P2D_general(df, key_numer, key_denom):
  if df is None:
    return
  probs = {} # the result to return
  denom = {}
  for t in df.itertuples():
    dic=t._asdict() # recover the data frame
    # get denom value
    i = dic[key_denom] 
    if i not in probs.keys():
      probs[i] = {} # create the under-dictionary if the key does not exixte
    # get the value of the Numerator field
    j = dic[key_numer]
    # count the value of attr
    probs[i][j] = probs[i][j]+1 if j in probs[i].keys() else 1
    # count the value of denom
    denom[i] = denom[i]+1 if i in denom.keys() else 1
  # claculate the probabilities
  for i in probs.keys():
    for j in probs[i].keys():
      probs[i][j] /= denom[i]
  return probs

def P2D_l(df,attr):
  return P2D_general(df, attr, 'target')
  
def P2D_p(df,attr):
  return P2D_general(df, 'target', attr)


class ML2DClassifier(APrioriClassifier):

  attr = ''
  probs = {}

  def __init__(self, df, attr):
    self.attr = attr
    self.probs = P2D_l(df,attr)

  def estimClass(self,one_line):
    j = one_line[self.attr]
    if self.probs[0][j] > self.probs[1][j]:
      return 0
    return 1

class MAP2DClassifier(APrioriClassifier):

  attr = ''
  probs = {}

  def __init__(self, df, attr):
    self.attr = attr
    self.probs = P2D_p(df,attr)

  def estimClass(self,one_line):
    i = one_line[self.attr]
    if self.probs[i][0] > self.probs[i][1]:
      return 0
    return 1

# Question 4
def nbParams(data, keys=[]):
  # default of keys is all the keys of data
  if keys:
    data = data[keys]
  # count the number of different values of each sample of data
  count_value = count_samples_spaces(data)
  size = 1
  # calculate all the possibilities (number of float)
  for value in count_value.values():
    size *= value
  # 8 Byte/float
  size *= 8 
  # output: print in terminal
  output = "{} variable(s) : {} octets".format(len(count_value), size)
  if size < 1024:
    print(output)
  else:
    print(output + " = " + size_format_str(size))

def count_samples_spaces(data):
  """Calculate the sizes of sample sapce of each sample in data"""
  result = {}
  for key,sample in data.items():
    temp = []
    for value in sample:
      if value not in temp:
        temp.append(value)
    result[key] = len(temp)
  return result
  
def size_format_str(size):
  """Format the value `size`"""
  output = ""
  units = ["To","Go","Mo","Ko","o"]
  while size != 0:
    output = "{} {}  ".format(size%1024,units.pop()) + output
    size //= 1024
  return output.strip()

def nbParamsIndep(data, keys=[]):
  # get only what we want in data
  if keys:
    data = data[keys]
  # get the sizes of sample space of all the sample
  sizes_sample_space = count_samples_spaces(data)
  size = 0
  # calculate all the possibilities (number of float)
  for value in sizes_sample_space.values():
    size += value
  # 8 Byte/float
  size *= 8
  # output
  output = "{} variable(s) : {} octets".format(len(data), size)
  if size < 1024:
    print(output)
  else:
    print(output + " = " + size_format_str(size))

