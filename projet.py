import math

def getPrior(data):
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