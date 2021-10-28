"""
Very simple modification of the purely greedy method, that is selecting the apparent
best outcome each and every time.
We initialize our mean to a large very and trickle downward using a greedy approach
The initial estimated_prob acts as a hyper parameter, High value makes the algorithm explore
because the greedy algorithm believes that the mean is high, if the value is initialized to a smaller value,
it explores lesser.
"""
import numpy as np


class Bandit:
	def __init__(self, p):
		self.p = p
		self.N = 1.0
		self.estimated_prob = 50.2


	def get_result(self):
		return np.random.random() < self.p

	def update(self, x):
		self.N += 1.0
		self.estimated_prob = (self.N*self.estimated_prob+x)/(self.N+1)


def experiment():
	probs = [0.01, 0.02, 0.4]
	badits = [Bandit(p) for p in probs]
	for i in range(0, 100000):
		j = np.argmax([badit.estimated_prob for badit in badits])
		x = badits[j].get_result()
		badits[j].update(x)

	print(f"best value is {max([badit.estimated_prob for badit in badits])} at {np.argmax([badit.estimated_prob for badit in badits])}")
	for i in range(0, len(badits)):
		print(badits[i].estimated_prob)

experiment()