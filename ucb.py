"""
Upper confidence bound method to solve exploration - exploitation dilemma
We use probability to get the upper bound
We make use Hoeffding's inequality

The only changes are that we use a specific formula and initialize the bandits before running the 
algorithm.
"""
import numpy as np

class Bandit:
	def __init__(self, p):
		self.p = p
		self.N = 0
		self.prob_estimate = 0


	def get_result(self):
		return np.random.random() < self.p


	def update(self, x):
		self.N += 1
		self.prob_estimate = (self.N*self.prob_estimate + x)/(self.N+1)
		return

def ucb(mean, n, nj):
	return mean + np.sqrt(2*np.log(n)/nj)


def experiment():
	probs = [0.2, 0.5, 0.69]
	bandits = [Bandit(p) for p in probs]
	total_plays = 0
	
	for i in range(0, len(bandits)):
		x = bandits[i].get_result()
		total_plays += 1
		bandits[i].update(x)

	for j in range(0, 10000):
		k = np.argmax([ucb(bandit.prob_estimate, total_plays, bandit.N) for bandit in bandits])
		x = bandits[k].get_result()
		total_plays += 1
		bandits[k].update(x)


	for l in range(0, len(bandits)):
		print("Actual" , "Calculated", bandits[l].prob_estimate, probs[l])

	best_ind = np.argmax([bandit.prob_estimate for bandit in bandits])
	print(f"best val : {bandits[best_ind].prob_estimate} which is index {best_ind}")

experiment()
