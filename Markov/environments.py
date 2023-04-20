import numpy as np
# The class StochasticEnvironment is a class that simulates a stochastic environment. 
# 
# The class has two attributes: 
# 
# 1. gamma: the probability distribution of the states
# 2. reward_event_mean_matrix: the reward event mean matrix
# 
# The class has two methods: 
# 
# 1. pull: this method takes in an arm and returns a reward of 1 with probability
# reward_event_mean_matrix[arm, state] and 0 otherwise. 
# 2. get_success_prob: this method returns the success probability for Oracle. 
# 
# The class also has a constructor that takes in two arguments: 
# 
# 1. reward_event_mean_matrix: the reward event mean matrix
# 2. gamma: the probability distribution of the states
# 
# The constructor also initializes the state of the environment to a random state according to the
# probability distribution gamma. 
# 
# The constructor also
class StochasticEnvironment():
    def __init__(self, reward_event_mean_matrix,gamma,n):
        """
        The function takes in a reward event mean matrix and a gamma vector and returns a class object
        with the following attributes: gamma, state, success_prob, and reward_event_mean_matrix
        
        :param reward_event_mean_matrix: This is the matrix of the expected reward for each action in
        each state
        :param gamma: the stationary distribution of the Markov chain
        """
        self.n=n
        self.gamma=gamma
        self.state = np.random.choice(self.n,p=gamma) 
        self.success_prob=(reward_event_mean_matrix@gamma).reshape(1,self.n)
        self.reward_event_mean_matrix=reward_event_mean_matrix
        
    
    def pull(self, arm):
        """
        The function takes in an arm and returns a reward of 1 with probability equal to the reward
        event mean of the arm and state, and 0 otherwise
        
        :param arm: the arm to pull
        :return: The reward is being returned.
        """
        self.state=np.random.choice(self.n,size=None,replace=True,p=self.gamma)
        return 1 if np.random.rand() < self.reward_event_mean_matrix[arm, self.state] else 0

        

    # Return the success probability for Oracle and for debugging purposes
    def get_success_prob(self):
        """
        It returns the success probability of the Bernoulli distribution
        :return: The success probability of the agent.
        """
        return self.success_prob.reshape(1, self.n)


# The MarkovianEnvironment class is a class that simulates a Markovian environment. 
# 
# The class has two attributes: 
# 
# 1. transition_matrix: a 100x100 matrix that represents the transition probabilities between states. 
# 2. reward_event_mean_matrix: a 10x100 matrix that represents the probability of success for each arm
# in each state. 
# 
# The class has two methods: 
# 
# 1. pull: takes in an arm and returns a reward of 1 with probability equal to the probability of
# success for that arm in the current state. 
# 2. stationary_distribution: returns the stationary distribution of the Markov chain. 
# 
# The class also has two helper methods: 
# 
# 1. get_stationary_prob: returns the stationary distribution of the Markov chain. 
# 2. get_success_prob: returns the probability of success for each arm.

class MarkovianEnvironment():
    def __init__(self, transition_matrix,reward_event_mean_matrix,n):
        """
        The function takes in the transition matrix and the reward event mean matrix and initializes the
        state of the agent to a random state, calculates the stationary distribution and the success
        probability
        
        :param transition_matrix: The transition matrix of the Markov Chain
        :param reward_event_mean_matrix: This is a matrix of size 100x100. Each row represents a state
        and each column represents an action. The value at each cell is the expected reward for taking
        that action in that state
        """
        self.n=n
        self.transition_matrix = transition_matrix
        self.state = np.random.choice(self.n) 
        self.reward_event_mean_matrix=reward_event_mean_matrix
        self.stationary_prob=self.stationary_distribution()
        self.success_prob=(self.reward_event_mean_matrix@self.stationary_prob).reshape(1,self.n)

    def pull(self, arm):
        """
        The function takes in an arm and returns a reward of 1 with probability equal to the reward
        event mean of the arm and state, and 0 otherwise
        
        :param arm: the arm to pull
        :return: The reward for the arm pulled.
        """
        self.state=np.random.choice(self.n,size=None,replace=True,p=self.transition_matrix[self.state])
        return 1 if np.random.rand() < self.reward_event_mean_matrix[arm, self.state] else 0

        
        
    def stationary_distribution(self):
        """
        The stationary distribution is the eigenvector of the transition matrix corresponding to the
        eigenvalue 1
        :return: The stationary distribution of the Markov chain.
        """
        #cite : https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain
        #note: the matrix is row stochastic.
        #A markov chain transition will correspond to left multiplying by a row vector.
        #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
        evals, evecs = np.linalg.eig(self.transition_matrix.T)
        evec1 = evecs[:,np.isclose(evals, 1)]

        #Since np.isclose will return an array, we've indexed with an array
        #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
        evec1 = evec1[:,0]

        stationary = evec1 / evec1.sum()

        #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
        stationary = stationary.real
        return stationary
    def get_stationary_prob(self):
        """
        This function returns the stationary probability of the Markov chain
        :return: The stationary probability of the Markov Chain.
        """
        return self.stationary_prob
    # Return the success probability for Oracle and for debugging purposes
    def get_success_prob(self):
        """
        It returns the success probability of the Bernoulli distribution.
        :return: The success probability of the agent.
        """
        return self.success_prob
