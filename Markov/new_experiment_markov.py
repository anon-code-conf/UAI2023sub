
#!pip install ray
#Acknoledgement - We thank Saxena et.al. to providing a code base for LinConTS and LinKLUCB on top of which we have built our code.
import os
os.environ["RAY_ALLOW_SLOW_STORAGE"] = "1"
os.environ["RAY_DISABLE_MEMORY_MONITOR"]="1"
from cvxopt import matrix, solvers
import time
import numpy as np
import pandas as pd
import ray
from sklearn.isotonic import IsotonicRegression
import torch
from torch import nn
from environments import MarkovianEnvironment,StochasticEnvironment





def error_matrix_initialization(reward_event_mean_matrix):
    E = 1 - reward_event_mean_matrix
    m=E.shape[0]
    n=E.shape[1]
    return E,m,n

reward_value = np.array([0.11111111, 0.16666667, 0.22222222, 0.33333333, 0.44444444,
                                 0.66666667, 0.88888889, 1.])

a=np.array([.9,.8,.7,.6,.5,.4,.3,.2])
R1 = np.array([a,a,a,a,a,a,a,a]).T
R2 = np.array([[1, 0., 0., 0., 0., 0., 0., 0.],
       [0., 1, 0., 0., 0., 0., 0., 0.],
       [0., 0., 1, 0., 0., 0., 0., 0.],
       [0., 0., 0., 1, 0., 0., 0., 0.],
       [0., 0., 0., 0., 1, 0., 0., 0.],
       [0., 0., 0., 0., 0., 1, 0., 0.],
       [0., 0., 0., 0., 0., 0., 1, 0.],
       [0., 0., 0., 0., 0., 0., 0., 1]])
rank_1_fraction = .9
reward_event_mean_matrix = rank_1_fraction*R1+(1-rank_1_fraction)*R2
#R = generate_reward_event_matrix(convex=convex)

#transition_matrix = generate_transition_matrix()
transition_matrix = np.array([[.5, 0.5, 0, 0., 0., 0.,0,   0     ],
                              [.17, .5, 0.33, 0., 0., 0.,0,0     ],
                              [0., 0.29, .5, 0.21, 0, 0.,0,0     ],
                              [0., 0., .4, 0.5, 0.1,0 ,0,0       ],
                              [0., 0., 0., 0.3, .5, 0.2 ,0,0     ],
                              [0., 0., 0., 0.,0.12, .5, 0.38,0.0 ],
                              [0., 0., 0., 0., 0, 0.05, 0.5, 0.45],
                              [0, 0., 0., 0., 0, 0.4,0.1,0.5   ]
                              ]
                            )
E, m, n = error_matrix_initialization(reward_event_mean_matrix)
gamma = MarkovianEnvironment(transition_matrix=transition_matrix,reward_event_mean_matrix=reward_event_mean_matrix,n=n).get_stationary_prob()
reward_event_mean = reward_event_mean_matrix @ gamma
numbers = np.zeros((m, 2))
best_arm = np.argmax((reward_event_mean_matrix @ gamma) * reward_value)
best_arm_expected_reward = reward_event_mean[best_arm] * reward_value[best_arm]
best_arm_reward_event_mean = reward_event_mean[best_arm]

print('Best Arm: %d, Expected Reward %0.4f, Mean Reward Events: %0.2f' % (best_arm,
                                                                          best_arm_expected_reward,
                                                                          best_arm_reward_event_mean))


print(f'Reward event mean - {reward_event_mean} and {gamma}')
# import sys
# sys.exit("Error message")

@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value


counter_actor = Counter.remote()

nrof_arms = len(reward_event_mean_matrix)



T = 20000
N = 20

channel_event = np.random.choice(n, T, p=gamma)
channel_rolling = np.zeros((T, n))




"""# Base Constrained Bandit"""


# nrof_arms: Number of bandit arms (K)
# reward_value: Reward value for each arm (r_k) if successful
# target_success_prob: Target success probability
# window_size: Window size for sliding window bandit. Events outside the window are discarded

# It's a base class for a bandit algorithm that is constrained to have a certain success probability
class BaseConstrainedBandit():
    def __init__(self,
                 nrof_arms,
                 reward_value,
                 target_success_prob=0.0):

        self.nrof_arms = nrof_arms
        self.reward_value = reward_value

        self.target_success_prob = target_success_prob

        self.t = 0

        self.pull_count = [0 for _ in range(self.nrof_arms)]
        self.success_count = [0 for _ in range(self.nrof_arms)]
        self.gam = [0] * 8
    # Determine which arm to be pulled

    def act(self):  # Implemented in child classes
        pass

    # Update the bandit
    def update(self, arm, success):
        self.t += 1

        self.pull_count[arm] += 1
        self.success_count[arm] += success

    # Calculate the selection probability vector by solving a linear program
    def calculate_selection_probabilities(self, success_prob, tolerance=1e-5):
        c = matrix((-1 * np.array(success_prob) *
                   np.array(self.reward_value)).T)
        neg_success_prob = [-1.0 *
                            r for r in np.array(success_prob).reshape(1, 8)]
        G = matrix(
            np.vstack([neg_success_prob, -1.0 * np.eye(self.nrof_arms)]))
        h = matrix(np.append(-1 * self.target_success_prob,
                   np.zeros((1, self.nrof_arms))))

        A = matrix(np.ones((1, self.nrof_arms)))
        b = matrix([1.0])

        sol = solvers.lp(c, G, h, A, b, solver='glpk', options={
                         'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
        selection_prob = np.reshape(np.array(sol['x']), -1)

        if None in selection_prob:  # Unsolvable optimiation
            return [None]

        # Fix numerical issues
        # Remove precision-related values
        selection_prob[np.abs(selection_prob) < tolerance] = 0.0
        # Recalibrate probability vector to sum to 1
        selection_prob = selection_prob / sum(selection_prob)

        return selection_prob

    # Sample from the probabilistic selection vector
    def sample_prob_selection_vector(self, prob):
        try:
            return np.argwhere(np.random.multinomial(1, prob))[0][0]
        except Exception:
            print('Error thrown by prob sampling. Returning random sample')
            return np.random.randint(0, self.nrof_arms)


"""# Oracle Constrained Bandit
# Implements the stationary optimal policy
"""

# nrof_arms: Number of bandit arms (K)
# reward_value: Reward value for each arm (r_k) if successful
# target_success_prob: Target success probability
# success_prob: Success probability for each bandit arm


class OracleConstrainedBandit(BaseConstrainedBandit):
    def __init__(self,
                 nrof_arms,
                 reward_value,
                 target_success_prob=0.0,
                 env_instance=None):

        super().__init__(nrof_arms, reward_value, target_success_prob)
        self.env = env_instance

        success_prob = self.env.get_success_prob()

        self.selection_prob = self.calculate_selection_probabilities(
            success_prob)

    # Determine which arm to be pulled
    def act(self):
        return self.sample_prob_selection_vector(self.selection_prob)

    # Get selection probabilties (for debugging purposes)
    def get_selection_prob(self):
        return self.selection_prob


(reward_event_mean_matrix @ gamma).shape

"""# LinCon-KL-UCB Bandit"""

eps = 1e-15

# Adopted from
# https://nbviewer.jupyter.org/github/Naereen/notebooks/blob/master/Kullback-Leibler_divergences_in_native_Python__Cython_and_Numba.ipynb#Generic-KL-UCB-indexes,-with-a-bisection-search


def klBern(x, y):
    r""" Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence

    .. math:: \mathrm{KL}(\mathcal{B}(x), \mathcal{B}(y)) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y})."""
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def estimate_kl_ucb(x, d, kl, upperbound, lowerbound=float('-inf'), precision=1e-6, max_iterations=50):
    """ The generic KL-UCB index computation.

    - x: value of the cum reward,
    - d: upper bound on the divergence,
    - kl: the KL divergence to be used (:func:`klBern`, :func:`klGauss`, etc),
    - upperbound, lowerbound=float('-inf'): the known bound of the values x,
    - precision=1e-6: the threshold from where to stop the research,
    - max_iterations: max number of iterations of the loop (safer to bound it to reduce time complexity).

    .. note:: It uses a **bisection search**, and one call to ``kl`` for each step of the bisection search.
    """
    value = max(x, lowerbound)
    u = upperbound
    _count_iteration = 0
    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        m = (value + u) / 2.
        if kl(x, m) > d:
            u = m
        else:
            value = m
    return (value + u) / 2.

# nrof_arms: Number of bandit arms (K)
# reward_value: Reward value for each arm (r_k) if successful
# target_success_prob: Target success probability


class LinConKLUCBBandit(BaseConstrainedBandit):
    def __init__(self,
                 nrof_arms,
                 reward_value,
                 target_success_prob=0.0):

        super().__init__(nrof_arms, reward_value, target_success_prob)

    # Determine which arm to be pulled
    def act(self):
        # Ensure that all arms are pulled at least once
        if self.t < self.nrof_arms:
            return self.t

        # Calculate the current KL-UCB for each arm
        kl_ucb = [self.calculate_kl_ucb(arm) for arm in range(self.nrof_arms)]

        # If not unimodal, select the arm constrained KL-UCB algorithm
        kl_ucb_prob = self.calculate_selection_probabilities(kl_ucb)

        if self.t == T - 1:
            print(kl_ucb_prob)
        if None in kl_ucb_prob:  # Unsolvable optimization
            return np.random.randint(0, self.nrof_arms)
        else:
            return self.sample_prob_selection_vector(kl_ucb_prob)

    # Calculate KL-UCB for the specified arm
    def calculate_kl_ucb(self, arm):
        empirical_mean = self.success_count[arm] / self.pull_count[arm]

        return estimate_kl_ucb(empirical_mean, np.log(self.t) / self.pull_count[arm], klBern, 1)


"""# LinConTS Bandit"""

# nrof_arms: Number of bandit arms (K)
# reward_value: Reward value for each arm (r_k) if successful
# target_success_prob: Target success probability


class LinConTSBandit(BaseConstrainedBandit):
    def __init__(self,
                 nrof_arms,
                 reward_value,
                 target_success_prob=0.0):

        super().__init__(nrof_arms, reward_value, target_success_prob)

    # Determine which arm to be pulled
    def act(self):
        # Ensure that each arm is pulled at least once
        if self.t < self.nrof_arms:
            return self.t

        # Sample a success probability from beta distribution Beta(a, b)
        # where a = 1 + self.success_count[arm]
        # and   b = 1 + self.pull_count[arm] - self.success_count[arm]
        sampled_success_prob = [np.random.beta(1 + self.success_count[arm],
                                               1 + self.pull_count[arm] - self.success_count[arm])
                                for arm in range(self.nrof_arms)]

        # Success probability constraint through linear programming
        ts_prob = self.calculate_selection_probabilities(sampled_success_prob)
        
        if None in ts_prob:  # Unsolvable optimization
            return np.random.randint(0, self.nrof_arms)
        else:
            return self.sample_prob_selection_vector(ts_prob)




"""# LinConErrorInformationTS 2 Bandit"""
""" This uses Dirichlet distribution"""

# nrof_arms: Number of bandit arms (K)
# reward_value: Reward value for each arm (r_k) if successful
# target_success_prob: Target success probability
"""# Full Imformation"""
""" This uses Dirichlet distribution"""

class LinConErrorTSFullInformation(BaseConstrainedBandit):
    def __init__(self,
                 nrof_arms,
                 reward_value,
                 target_success_prob=0.0):

        super().__init__(nrof_arms, reward_value, target_success_prob)
        self.gam = np.zeros(nrof_arms)

    # Determine which arm to be pulled
    def act(self):
        # Ensure that each arm is pulled at least once
        if self.t < self.nrof_arms:
            return self.t
        # Sample a success probability from beta distribution Beta(a, b)
        # where a = 1 + self.success_count[arm]
        # and   b = 1 + self.pull_count[arm] - self.success_count[arm]
        sampled_success_prob = reward_event_mean_matrix @ np.random.dirichlet(
            channel_rolling[self.t] + 1)
        diverging = np.linalg.norm(reward_event_mean - reward_event_mean_matrix @ (
            (channel_rolling[self.t] + 1) / sum(channel_rolling[self.t] + 1)))
        # Success probability constraint through linear programming
        ts_prob = self.calculate_selection_probabilities(sampled_success_prob)
        
        if None in ts_prob:  # Unsolvable optimization
            return np.random.randint(0, self.nrof_arms)
        else:
            return self.sample_prob_selection_vector(ts_prob)


# nrof_arms: Number of bandit arms (K)
# reward_value: Reward value for each arm (r_k) if successful
# target_success_prob: Target success probability
"""# Non Full Imformation"""
""" This uses Dirichlet distribution"""

class LinConErrorTS(BaseConstrainedBandit):
    def __init__(self,
                 nrof_arms,
                 reward_value,
                 target_success_prob=0.0):

        super().__init__(nrof_arms, reward_value, target_success_prob)
        self.gam = np.zeros(nrof_arms)

    def gradient(self, gamma,numbers):
        tmp = np.copy(E)
        arms = len(E)
        channels = len(E[0])
        arm_distribution = np.sum(numbers,axis = 1)/np.sum(numbers)

        for i in range(arms):
            div = np.dot(gamma, E[i])
            for j in range(channels):
                if div == 0:
                   tmp[i, j] = 100*arm_distribution[i]
                elif div==1:
                    tmp [i,j] = 100*arm_distribution[i]
                    #tmp[i, j] = numbers[i, 0] * E[i, j] / \
                    float(div)
                else:
                    #tmp[i,j] = 100
                    tmp[i, j] =numbers[i, 0] * E[i, j] /  float(div) +  numbers[i, 1] * (1- E[i, j]) / (1 - float(div))
       # print(numbers)
        return tmp.sum(axis=0)

    def project(self, v):  # https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf
        """
        The function takes a vector and returns a vector that is the closest point to the original
        vector in the L1 ball
        
        :param v: the vector to be projected
        :return: the projection of the vector v onto the simplex.
        """
        u = np.sort(v)[::-1]  # sort the vector in descending order
        s = 0
        r = len(v)
        for i in range(len(v)):
            s = s + u[i]
            if u[i] + ((1 - s) / float(i + 1)) <= 0:
                r = i
                break
            i = i + 1
        g = (1 - sum(u[:r])) / float(r)
        v = v + g
        v[v < 0] = 0
        return v

    def get_gamma(self,numbers,gamma,te):
        for t in range(100):
            gamma = gamma + (1 / (1000.0 +0* t+10*te)) * self.gradient(gamma,numbers)
            gamma = self.project(gamma)
        return gamma

    # Determine which arm to be pulled
    def act(self,numbers):
        # Ensure that each arm is pulled at least once
        if self.t < self.nrof_arms:
            return self.t
        if  "gamma" not in locals():
           gamma = np.ones(len(E[0])) / float(len(E[0]))
        # Sample a success probability from beta distribution Beta(a, b)
        # where a = 1 + self.success_count[arm]
        # and   b = 1 + self.pull_count[arm] - self.success_count[arm]

        gamma = self.get_gamma(numbers,gamma,self.t)
        gamma = np.random.dirichlet(gamma*self.t + 1)
        sampled_success_prob = reward_event_mean_matrix @ gamma
        # Success probability constraint through linear programming
        ts_prob = self.calculate_selection_probabilities(sampled_success_prob)
        #print(self.t)
        
        if  None in ts_prob:  # Unsolvable optimization
            return np.random.randint(0, self.nrof_arms)
        else:
            return self.sample_prob_selection_vector(ts_prob)

class LinConMLETSIsotonic(BaseConstrainedBandit):
    def __init__(self,
                 nrof_arms,
                 reward_value,
                 target_success_prob=0.0):

        super().__init__(nrof_arms, reward_value, target_success_prob)
        self.gam = np.zeros(nrof_arms)


    def gradient(self, gamma,numbers):
        tmp = np.copy(E)
        arms = len(E)
        channels = len(E[0])
        arm_distribution = np.sum(numbers,axis = 1)/np.sum(numbers)

        for i in range(arms):
            div = np.dot(gamma, E[i])
            for j in range(channels):
                if div == 0:
                   tmp[i, j] = 100*arm_distribution[i]
                elif div==1:
                    tmp [i,j] = 100*arm_distribution[i]
                    #tmp[i, j] = numbers[i, 0] * E[i, j] / \
                    float(div)
                else:
                    #tmp[i,j] = 100
                    tmp[i, j] =numbers[i, 0] * E[i, j] /  float(div) +  numbers[i, 1] * (1- E[i, j]) / (1 - float(div))
       # print(numbers)
        return tmp.sum(axis=0)

    def project(self, v):  # https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf
        u = np.sort(v)[::-1]  # sort the vector in descending order
        s = 0
        r = len(v)
        for i in range(len(v)):
            s = s + u[i]
            if u[i] + ((1 - s) / float(i + 1)) <= 0:
                r = i
                break
            i = i + 1
        g = (1 - sum(u[:r])) / 8.0
        v = v + g
        v[v < 0] = 0
        return v

    def get_gamma(self,numbers,gamma,te):
        channels = len(E[0])
        #gamma = np.ones(channels) / float(channels)
        for t in range(100):
            gamma = gamma + (1 / (1000.0 +0* t+10*te)) * self.gradient(gamma,numbers)
            #print(gamma,self.gradient(gamma,numbers),'llllll')
            gamma = self.project(gamma)

        return gamma

    # Determine which arm to be pulled
    def act(self,numbers):
        # Ensure that each arm is pulled at least once
        if self.t < self.nrof_arms:
            return self.t
        if  "gamma" not in locals():
           gamma = np.ones(len(E[0])) / float(len(E[0]))

        gamma = self.get_gamma(numbers,gamma,self.t)
        
        gamma = np.random.dirichlet(gamma*self.t + 1)
        sampled_success_prob = reward_event_mean_matrix @ gamma
        iso_reg = IsotonicRegression(increasing=False,y_min=0.0,y_max=1.0)
        sampled_success_prob=iso_reg.fit_transform(reward_value, sampled_success_prob)
        ts_prob = self.calculate_selection_probabilities(sampled_success_prob)
        
        if  None in ts_prob:  # Unsolvable optimization
            return np.random.randint(0, self.nrof_arms)
        else:
            return self.sample_prob_selection_vector(ts_prob)


"""# LinConErrorInformationTS Bandit"""

# nrof_arms: Number of bandit arms (K)
# reward_value: Reward value for each arm (r_k) if successful
# target_success_prob: Target success probability

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

error=torch.tensor(E,requires_grad=False)

# It's a neural network with a single hidden layer of 100 neurons, with a LeakyReLU activation function,
# and an output layer with a softmax activation function
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #NLS begin
            nn.Linear(n, 100),
            nn.LeakyReLU(),
            #NLS end
		nn.Linear(100, n)
        )
         

    def forward(self, x):
        relu_stack_output = self.linear_relu_stack(x)
        return torch.nn.Softmax(dim=-1)(relu_stack_output)

# It's a class that trains a neural network to estimate the gamma parameter 
class Model:
    def __init__(self):
        self.model = NeuralNetwork().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.5, momentum=0.99)#torch.optim.Adam(model.parameters())#, lr=learning_rate)
        self.estimated_gamma = np.zeros_like(gamma)
        self.i = 0
        self.model.double()
        global error
        self.error=error.to(device)
        self.model.to(device)
        self.model(error[0])
    def loss_fn(self,x,y):
        criterion = torch.nn.BCELoss(reduction='none')
        return criterion(x,y)
    def train_loop(self,encoder,number):
        #X, y = X.to(device), y.to(device)
        model_output= self.model(error[encoder].double())
        #print(model_output)
        output=self.error@model_output
        frac=number[encoder][0]/np.sum(number[encoder])
        #print(number[:,1]/np.sum(number,axis=1))
        loss=self.loss_fn(output[encoder].double(),torch.tensor(frac,device=device).double())
        #print(loss.item())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        k=self.i+.5
        self.estimated_gamma= k/(k+1)*self.estimated_gamma+1.0/(k+1)* model_output.cpu().detach().numpy()
        #self.estimated_gamma= .99*self.estimated_gamma+.01* model_output.cpu().detach().numpy()
        self.i+=1
        #return estimated_gamma



class LinConErrorTSNeural(BaseConstrainedBandit):
    def __init__(self,
                 nrof_arms,
                 reward_value,
                 target_success_prob=0.0):

        super().__init__(nrof_arms, reward_value, target_success_prob)
        self.gam = np.zeros(nrof_arms)

    
    # Determine which arm to be pulled
    def act(self,neural_model):
        # Ensure that each arm is pulled at least once
        if self.t < self.nrof_arms:
            return self.t
        if  "gamma" not in locals():
           gamma = np.ones(len(E[0])) / float(len(E[0]))
        

        gamma = neural_model.estimated_gamma
        sampled_gamma = np.random.dirichlet(gamma*self.t + 1)
        sampled_success_prob = reward_event_mean_matrix @ sampled_gamma
        #print(sampled_success_prob)
        # Success probability constraint through linear programming
        ts_prob = self.calculate_selection_probabilities(sampled_success_prob)
       
        if  None in ts_prob:  # Unsolvable optimization
            return np.random.randint(0, self.nrof_arms)
        else:
            return self.sample_prob_selection_vector(ts_prob)


# Funtion to run experiment for a given policy and given time horizon
@ray.remote  # Comment out this line to run locally
def run_experiment(policy, T, play_result, reward_value,neural_model = None):
    outcome = np.zeros((T, 2))
    global numbers
    for t in range(T):
        if isinstance(policy, (LinConErrorTS, LinConMLETSIsotonic)):
            arm = policy.act(numbers)
        elif isinstance(policy,LinConErrorTSNeural) :
            arm = policy.act(neural_model=neural_model)
        else:
            arm = policy.act()
        success = play_result[t][arm]
        outcome[t, 0] = success
        numbers[arm, success] += 1
        #print(numbers)
       # print(t,arm,sum(outcome[:t,0])/(t+1.0),np.sum(play_result[:t],axis=0))
        if isinstance(policy,LinConErrorTSNeural):
            neural_model.train_loop(arm,numbers)
        if success:
            outcome[t, 1] = reward_value[arm]
        policy.update(arm, success)
    return outcome
    # return outcome

# Run the policy


def play(policy, T, N, play_result, reward_value, name):
    start = time.time()

    # Uncomment the following line to run locally
   # outcome = [run_experiment(policy, T, play_result[n], reward_value) for n in range(N)]
    if isinstance(policy,LinConErrorTSNeural):
        outcome = ray.get([run_experiment.remote(policy, T, play_result[n], reward_value,Model()) for n in range(N)])
    else:
        outcome = ray.get([run_experiment.remote(policy, T, play_result[n], reward_value) for n in range(N)])

    total_success = np.zeros((N, T))
    total_reward = np.zeros((N, T))
    for n in range(N):
        total_success[n, :] = outcome[n][:, 0]
        total_reward[n, :] = outcome[n][:, 1]

    print(name + ' done! Elasped: %0.2fs' % (time.time() - start))

    return total_success, total_reward


# Generate all events in advance
np.random.seed(42)
#env = Environment(reward_event_mean)
@ray.remote  # Comment out this line to run locally
def generate_data(env,T):
    return [[env.pull(arm) for arm in range(nrof_arms)] for _ in range(T)] 
#env= StochasticEnvironment(reward_event_mean_matrix=reward_event_mean_matrix,gamma=gamma)
env =  MarkovianEnvironment(transition_matrix=transition_matrix,reward_event_mean_matrix=reward_event_mean_matrix,n=n)
constraint = np.mean(reward_event_mean_matrix@gamma)
constraint=np.mean(np.squeeze(env.get_success_prob().T))

#play_result = ray.get([generate_data.remote(StochasticEnvironment(reward_event_mean_matrix=reward_event_mean_matrix,gamma=gamma,n=n),T) for _ in range(N)])
play_result = ray.get([generate_data.remote(MarkovianEnvironment(transition_matrix=transition_matrix,reward_event_mean_matrix=reward_event_mean_matrix,n=n),T) for _ in range(N)])


# Create the policy instances


LinConTSPolicy = LinConTSBandit(nrof_arms, reward_value, constraint)
LinConKLUCBPolicy = LinConKLUCBBandit(nrof_arms, reward_value, constraint)
LinConErrorTSPolicy = LinConErrorTS(nrof_arms,reward_value,constraint)
LinConErrorTSFullInformationPolicy = LinConErrorTSFullInformation(nrof_arms,reward_value,constraint)
LinConErrorTSNeuralPolicy = LinConErrorTSNeural(nrof_arms,reward_value,constraint)
stationary_probabilities = OracleConstrainedBandit(nrof_arms,
                                                   reward_value,
                                                   constraint,
                                                   env_instance=env).get_selection_prob()

print(stationary_probabilities)
stationary_opt_reward = np.sum(
    stationary_probabilities * reward_event_mean * reward_value)
print('Stationary optimal reward: %0.4f' % (stationary_opt_reward))

arm_indices = np.nonzero(stationary_probabilities)[0]

print('Optimal Arms:')
for arm in arm_indices:
    print('Arm: %d, Reward Event Mean: %0.2f, Expected Reward Value: %0.4f, Selection prob: %0.2f' % (arm,
                                                                                                      reward_event_mean[arm],
                                                                                                      reward_event_mean[arm] *
                                                                                                      reward_value[arm],
                                                                                                      stationary_probabilities[arm]))


# Run the policies and collect results


LinConKLUCBPolicy_reward_events, LinConKLUCBPolicy_reward_values = play(LinConKLUCBPolicy,
                                                                T,
                                                                N,
                                                                play_result,
                                                                reward_value,
                                                                'LinConKLUCBPolicy')

LinConTSPolicy_reward_events, LinConTSPolicy_reward_values = play(LinConTSPolicy,
                                                        T,
                                                        N,
                                                        play_result,
                                                        reward_value,
                                                        'LinConTSPolicy')

# LinConErrorTSPolicy_reward_events, LinConErrorTSPolicy_reward_values = play(LinConErrorTSPolicy,
#                                                         T,
#                                                         N,
#                                                         play_result,
#                                                         reward_value,
#                                                         'LinConErrorTSPolicy')

LinConErrorTSNeuralPolicy_reward_events, LinConErrorTSNeuralPolicy_reward_values = play(LinConErrorTSNeuralPolicy,
                                                        T,
                                                        N,
                                                        play_result,
                                                        reward_value,
                                                        'LinConErrorTSNeuralPolicy')

# LinConErrorTSFullInformationPolicy_reward_events, LinConErrorTSFullInformationPolicy_reward_values = play(LinConErrorTSFullInformationPolicy,
#                                                         T,
#                                                         N,
#                                                         play_result,
#                                                         reward_value,
#                                                         'LinConErrorTSNeuralPolicy')





# Save the results
data = {'T': T,
        'N': N,
        'LinConKLUCBPolicy_reward_events': LinConKLUCBPolicy_reward_events,
        'LinConKLUCBPolicy_reward_values': LinConKLUCBPolicy_reward_values,
        'LinConTSPolicy_reward_events': LinConTSPolicy_reward_events,
        'LinConTSPolicy_reward_values': LinConTSPolicy_reward_values,
        # 'LinConErrorTSPolicy_reward_events':LinConErrorTSPolicy_reward_events,
        # 'LinConErrorTSPolicy_reward_values':LinConErrorTSPolicy_reward_values,
        'LinConErrorTSNeuralPolicy_reward_events':LinConErrorTSNeuralPolicy_reward_events,
        'LinConErrorTSNeuralPolicy_reward_values':LinConErrorTSNeuralPolicy_reward_values,
        # 'LinConErrorTSFullInformationPolicy_reward_events':LinConErrorTSPolicy_reward_events,
        # 'LinConErrorTSFullInformationPolicy_reward_values':LinConErrorTSPolicy_reward_values,
        'constraint': constraint,
        'stationary_opt_reward': stationary_opt_reward}
filename = 'channel_selector__eta%0.2f_T%d_N%d.npy' % (constraint, T, N)
filename = f'inv_obj_rank_(1_frac_{str(rank_1_fraction)}.npy'
filename = 'experiment_markov.npy'
np.save(filename, data)

print(f'Saved to {filename}')

