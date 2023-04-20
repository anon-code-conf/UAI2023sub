#Acknoledgement - We thank Saxena et.al. for providing a code base for LinConTS and LinKLUCB plotting on top of which we have built our code.

import numpy as np
from matplotlib import pyplot as plt
results_dir = ''

filename    = 'decreasing_con_rank_one_.1'
file_ext    = '.npy'


data=None
policyCollection = []
policyNames = []

def addPolicy(policyName):
    global policyCollection
    global policyNames
    policyCollection = policyCollection + [f'{x}Policy' for x in policyName]
    policyNames = policyNames + policyName

def worker(filename):
    filename = filename.replace('.','_')
    image_format = '.png'

    T = data['T']
    N = data['N']
    target_success_prob = data['constraint']
    print(filename,data.keys())
    stationay_opt_reward = data['stationary_opt_reward']#0.0139
    cum_constraint = np.tile( target_success_prob * np.arange( 0, T, 1 ), [ N, 1 ] )
    cum_opt_reward = np.tile( stationay_opt_reward * np.arange( 0, T, 1 ), [ N, 1 ] )
    policy_cum_reward_values={}
    policy_cum_regret = {}
    for policy in policyCollection:
        policy_cum_reward_values[policy] = np.cumsum(data[f'{str(policy)}_reward_values'], axis=1)
        policy_cum_regret[policy] = np.maximum(0.0, cum_opt_reward - policy_cum_reward_values[policy])
    policy_cum_violations = {policy: np.maximum(0.0, cum_constraint - np.cumsum(data[f'{str(policy)}_reward_events'], axis=1)) for policy in policyCollection}

    add_ticks(T)
    #plt.ylim([0, 5000])
    plt.rcParams.update({'font.size': 30, 'lines.linewidth' : 3, 'lines.markersize': 20})
    x_ticks = np.arange(0, T)
    global colorMapping
    for policy in policyCollection:
        plt.plot(x_ticks, np.mean( policy_cum_violations[policy], axis=0),color=colorMapping[policy])

    add_legends('Violation', filename, '_VIOLATION', image_format)
    add_ticks(T)
    #plt.ylim([0, 2000])

    x_ticks = np.arange(0, T)
    for policy in policyCollection:
        plt.plot(x_ticks, np.mean( policy_cum_regret[policy], axis=0 ),color=colorMapping[policy] )

    add_legends('Regret', filename, '_REGRET', image_format)
    plt.rcParams.update({'font.size': 30, 'lines.linewidth' : 3,'lines.markersize': 20})

    add_ticks(T)
    #plt.ylim([0, 2000])

    x_ticks = np.arange(0, T)
    for policy in policyCollection:
        plt.plot(x_ticks, np.mean( policy_cum_reward_values[policy], axis=0 ),color=colorMapping[policy] )

    add_legends('Cumulative Reward ', filename, '_REWARD', image_format)

    add_ticks(T)
    x_ticks = np.arange(0, T)
    for policy in policyCollection:
        plt.plot(x_ticks, np.divide( np.mean( policy_cum_reward_values[policy], axis=0), 
                                 np.mean( policy_cum_violations[policy], axis=0)),color=colorMapping[policy] )


    add_legends('Cum. Reward / Violation', filename, '_REWARD_VIO', image_format)



def add_legends(arg0, filename, arg2, image_format):
    plt.legend(policyNames, loc='upper left', fontsize=20)
    plt.xlabel('T')
    plt.ylabel(arg0)
    plt.savefig(results_dir + filename + arg2 + image_format, bbox_inches='tight')



def add_ticks(T):
    plt.figure(figsize=[8, 6])
    plt.grid(False)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(3,3))
    plt.xlim([0, T])



def genPlot(filename,file_ext='.npy'):
    global data
    data = np.load( results_dir + filename + file_ext, allow_pickle=True )[()]
    worker(filename=filename)
    
    
import glob
from pathlib import Path
files = glob.glob('./*markov.npy')
addPolicy(['LinConKLUCB','LinConTS','LinConErrorTSNeural'])
policyNames= ['LinConKLUCB','LinConTS','LinConErrorTS(Neural)','LinConErrorTS(MLE)','LinConErrorTS(FullInformation)']
colorMapping = {'LinConKLUCBPolicy':u'#1f77b4','LinConTSPolicy':u'#ff7f0e','LinConErrorTSPolicy':u'#2ca02c','LinConErrorTSNeuralPolicy':u'#8c564b','LinConErrorTSFullInformationPolicy':u'#d62728'}
#addPolicy(['LinConTS','LinConKLUCB', 'LinConErrorTS','LinConErrorTSNeural','LinConErrorTSFullInformation'])
#addPolicy(['LinConTS','LinConKLUCB','LinConErrorTSNeural'])
for file in files:
    genPlot(Path(file).stem)

