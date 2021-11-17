import math
from typing import ValuesView
from scipy.stats import chi2_contingency
from scipy.stats import chi2

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
        del dic['Index']
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

def P2D_general_old(df, A, B):
  """Calculate P(A|B) where `A` and `B` are the champs of data frame `df`"""
  if df is None:
    return
  probs = {} # the result to return
  denom = {}
  for t in df.itertuples():
    dic=t._asdict() # recover the data frame
    # get denom value
    i = dic[B] 
    if i not in probs.keys():
      probs[i] = {} # create the under-dictionary if the key does not exixte
    # get the value of the Numerator field
    j = dic[A]
    # count the value of attr
    probs[i][j] = probs[i][j]+1 if j in probs[i].keys() else 1
    # count the value of denom
    denom[i] = denom[i]+1 if i in denom.keys() else 1
  # claculate the probabilities
  for i in probs.keys():
    for j in probs[i].keys():
      probs[i][j] /= denom[i]
  return probs

def P2D_general(df, A, B):
  """Calculate P(A|B) where `A` and `B` are the champs of data frame `df`. And fill with 0 if P(A|B) does not exist"""
  if df is None:
    return
  probs = {} # the result to return
  values_A = []
  values_B = []
  numer = {}
  denom = {}
  for t in df.itertuples():
    dic=t._asdict() # recover the data frame
    # get denom value
    i = dic[B]
    if i not in values_B:
      values_B.append(i)
      numer[i] = {}
    denom[i] = denom[i]+1 if i in denom.keys() else 1
    # get the numer value
    j = dic[A]
    if j not in values_A:
      values_A.append(j)
    numer[i][j] = numer[i][j]+1 if j in numer[i].keys() else 1
  # claculate the probabilities
  for i in values_B:
    probs[i] = {}
    for j in values_A:
      probs[i][j] = numer[i][j] / denom[i] if j in  numer[i].keys() else 0
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
def nbParams(data, keys=None):
  # default of keys is all the keys of data
  if keys is not None:
    data = data[keys]
  # count the number of different values of each sample of data
  count_value = count_samples_spaces(data)
  size = 1
  # calculate all the probabilities (number of float)
  for value in count_value.values():
    size *= value
  # 8 Byte/float
  size *= 8 
  # output: print in terminal
  print_memory_size(len(data.keys()),size)

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

def print_memory_size(nb_var, size):
  output = "{} variable(s) : {} octets".format(nb_var, size)
  if size < 1024:
    print(output)
  else:
    print(output + " = " + size_format_str(size))

def nbParamsIndep(data):
  # get the sizes of sample space of all the sample
  sizes_sample_space = count_samples_spaces(data)
  size = 0
  # calculate all the probabilities (number of float)
  for value in sizes_sample_space.values():
    size += value
  # 8 Byte/float
  size *= 8
  # output
  print_memory_size(len(data.keys()),size)


# Question 5
def drawNaiveBayes(data,root):
  input_arg_str = ''
  keys = list(data.keys())
  keys.remove(root)
  keys.reverse() # 这句话只是单纯为了和老师的输出显示顺序一样，但其实加不加都不重要
  while len(keys) != 0:
    input_arg_str += "{}->{}".format(root, keys.pop())
    if len(keys) != 0:
      input_arg_str += ";"
  return drawGraph(input_arg_str)

def nbParamsNaiveBayes(data,root,keys=None):
  # get only what we want in data
  if keys is None: keys = data.keys()
  if len(keys)==0 : # keys = []
    size = count_samples_spaces(data[[root]])[root] * 8
    # output
    print_memory_size(0,size)
  else: 
    data = data[keys]
    # get the sizes of sample space of all the sample
    sizes_sample_space = count_samples_spaces(data)
    size = 0
    # calculate all the probabilities (number of float)
    for key, value in sizes_sample_space.items():
      if key != root:
        size += value
    size *= sizes_sample_space[root]
    size += sizes_sample_space[root]
    size *= 8 # 8 Byte/float
    # output
    print_memory_size(len(keys),size)


class MLNaiveBayesClassifier(APrioriClassifier):

  probs = {}

  def __init__(self, df):
    self.probs = {}
    for key in df.keys():
      if key != 'target':
        self.probs[key] = P2D_l(df,key)
  
  def getProb(self,champ, a, b):
    """get P(A|B), or 0 if P(A|B) does not exist"""
    if champ in self.probs.keys():
      if b in self.probs[champ].keys() and a in self.probs[champ][b].keys():
        return self.probs[champ][b][a]
      else:
        return 0
    else:
      return 1

  def estimProbas(self, one_line):
    result = self.estimLogProbas(one_line)
    for i in range(len(result)):
      result[i] = math.exp(result[i])
    return result
  
  def estimLogProbas(self, one_line):
    """Beacuse the likelihood is too small, so we use the log value to claculate the likelihood"""
    result = {0:0, 1:0}
    for key,value in one_line.items():
      if key != 'target':
        for i in range(2):
          proba = self.getProb(key, value, i)
          if proba != 0:
            result[i] += math.log(proba)
          else:
            result[i] = -math.inf
    return result

  def estimClass(self,one_line):
    logProbs = self.estimLogProbas(one_line)
    if logProbs[0] >= logProbs[1]:
      return 0
    return 1

class MAPNaiveBayesClassifier(MLNaiveBayesClassifier):

  mu = 0

  def __init__(self, df):
    super().__init__(df)
    self.mu = getPrior(df)['estimation']

  def estimProbas(self, one_line):
    return super().estimProbas(one_line)
  
  def estimLogProbas(self, one_line):
    # get \prod_i P(attr_i | target)
    result =  super().estimLogProbas(one_line)
    # calculate P(target) \cdot \prod_i P(attr_i | target)
    result[0] += math.log(1-self.mu)
    result[1] += math.log(self.mu)
    # normalize
    denom = self.get_log_sum(result.values())
    for i in range(2):
      if result[i] != -math.inf:
        result[i] = result[i] - denom
    return result
  
  def get_log_sum(self, log_probs):
    """Calculate log(\sum P_i) by given log(P_i)"""
    D = - max(log_probs)
    result = 0
    for item in log_probs:
      result += math.exp(item + D)
    return math.log(result) - D


# Question 6
def isIndepFromTarget(df, attr, alpha):
  #count
  table = get_contingency_table(df, attr)
  stat,p,dof,expected = chi2_contingency(table)
  prob = 1-alpha # 选取95%置信度
  critical = chi2.ppf(prob,dof)  # 计算临界阀值
  return abs(stat)<critical

def get_contingency_table(df, attr):
  # count
  values = []
  numbers = {0:{},1:{}}
  res = [[],[]]
  for t in df.itertuples():
    dic=t._asdict() # recover the data frame
    i = dic['target']
    j = dic[attr]
    if j not in values:
      values.append(j)
    numbers[i][j] = numbers[i][j]+1 if j in numbers[i].keys() else 1
  #to list
  for key in numbers[0]:
      res[0].append(numbers[0][key])
      res[1].append(numbers[1][key] if key in numbers[1].keys() else 0)
  # return result          
  return res

class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):

  def __init__(self, df, alpha):
    self.keys_of_indep = self.get_keys(df, alpha)
    super().__init__(df[self.keys_of_indep])

  
  def get_keys(self, df, alpha):
    keys_of_indep = []
    for key in df.keys():
      if key == 'target' or not isIndepFromTarget(df, key, alpha):
        keys_of_indep.append(key)
    return keys_of_indep
  
  def draw(self):
    input_arg_str = ''
    keys = self.keys_of_indep.copy()
    for champ in keys:
      if champ != "target":
        input_arg_str += "{}->{};".format('target', champ)
    # remove last ";"
    input_arg_str = input_arg_str[:-1]
    return drawGraph(input_arg_str)

class ReducedMAPNaiveBayesClassifier(ReducedMLNaiveBayesClassifier, MAPNaiveBayesClassifier):
  def __init__(self, df, alpha):
    ReducedMLNaiveBayesClassifier.__init__(self, df, alpha)
    MAPNaiveBayesClassifier.__init__(self, df[self.keys_of_indep])

