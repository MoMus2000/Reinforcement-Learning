import numpy as np

"""
Problem is called the multi arm bandit and focuses on exploration vs exploitation trade off.
By choosing to go what seems the best value at the start we minimize are chances to maximize our 
final reward.
Strategy to get close to the best value when our options are limited
for example i have 300 days to explore 3 restraunts and am not aware of the best of the three.
I can use the epsilon greedy strategy to choose the optimal one and deviate from the optimal with
a probability of epsilon.
The final result that I get dependant on the number of iterations should reflect choice closest to 
the best
outcome.
In the experiment below I am using probability values and as a result my goal is to get close to 
the greatest
probability value. I am not aware of the probability values during my experiment.
"""

class Bandit:
	def __init__(self, p):
		self.p = p
		self.N = 0
		self.estimated_probability = 0

	def get_result(self):
		return np.random.random() < self.p

	def update(self, x):
		self.N += 1.0
		self.estimated_probability = (self.estimated_probability*self.N+x)/(self.N+1)

def experiment():
	probs = [0.01, 0.02, 0.4]
	badits = [Bandit(p) for p in probs]
	epsilon = 0.1
	for i in range(0, 100000):
		random = np.random.random()
		if random < epsilon:
			j = np.random.randint(len(badits))
		else:
			j = np.argmax([badit.estimated_probability for badit in badits])
		x = badits[j].get_result()
		badits[j].update(x)

	print(f"best value is {max([badit.estimated_probability for badit in badits])} at {np.argmax([badit.estimated_probability for badit in badits])}")
	for i in range(0, len(badits)):
		print(badits[i].estimated_probability)


experiment()

