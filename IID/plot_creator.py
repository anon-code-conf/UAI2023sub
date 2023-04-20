# Acknowledgement - We thank Saxena et.al. for providing a code base for LinConTS and LinKLUCB plotting on top of which we have built our code.

import numpy as np
from matplotlib import pyplot as plt
import glob
from pathlib import Path

results_dir = ''

filename = 'decreasing_con_rank_one_.1'
file_ext = '.npy'

data = None
policyCollection = []
policyNames = []


def add_policy(policy_name):
    global policyCollection
    global policyNames
    policyCollection = policyCollection + [f'{x}Policy' for x in policy_name]
    policyNames = policyNames + policy_name


def worker(filename_local):
    filename_local = filename_local.replace('.', '_')
    image_format = '.png'

    t_capital = data['T']
    n_capital = data['N']
    target_success_prob = data['constraint']
    print(filename_local, data.keys())
    stationary_opt_reward = data['stationary_opt_reward']  # 0.0139
    cum_constraint = np.tile(target_success_prob * np.arange(0, t_capital, 1), [n_capital, 1])
    cum_opt_reward = np.tile(stationary_opt_reward * np.arange(0, t_capital, 1), [n_capital, 1])
    policy_cum_reward_values = {}
    policy_cum_regret = {}
    for policy in policyCollection:
        policy_cum_reward_values[policy] = np.cumsum(data[f'{str(policy)}_reward_values'], axis=1)
        policy_cum_regret[policy] = np.maximum(0.0, cum_opt_reward - policy_cum_reward_values[policy])
    policy_cum_violations = {
        policy: np.maximum(0.0, cum_constraint - np.cumsum(data[f'{str(policy)}_reward_events'], axis=1)) for policy in
        policyCollection}

    add_ticks(t_capital)
    # plt.ylim([0, 5000])
    plt.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 20})
    x_ticks = np.arange(0, t_capital)
    global colorMapping
    for policy in policyCollection:
        plt.plot(x_ticks, np.mean(policy_cum_violations[policy], axis=0), color=colorMapping[policy])

    add_legends('Violation', filename_local, '_VIOLATION', image_format)
    add_ticks(t_capital)
    # plt.ylim([0, 2000])

    x_ticks = np.arange(0, t_capital)
    for policy in policyCollection:
        plt.plot(x_ticks, np.mean(policy_cum_regret[policy], axis=0), color=colorMapping[policy])

    add_legends('Regret', filename_local, '_REGRET', image_format)
    plt.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 20})

    add_ticks(t_capital)
    # plt.ylim([0, 2000])

    x_ticks = np.arange(0, t_capital)
    for policy in policyCollection:
        plt.plot(x_ticks, np.mean(policy_cum_reward_values[policy], axis=0), color=colorMapping[policy])

    add_legends('Cumulative Reward ', filename_local, '_REWARD', image_format)

    add_ticks(t_capital)
    x_ticks = np.arange(0, t_capital)
    for policy in policyCollection:
        with np.errstate(divide='ignore'):
            plt.plot(x_ticks, np.divide(np.mean(policy_cum_reward_values[policy], axis=0),
                                        np.mean(policy_cum_violations[policy], axis=0)), color=colorMapping[policy])

    add_legends('Cum. Reward / Violation', filename_local, '_REWARD_VIO', image_format)


def add_legends(arg0, filename_local, arg2, image_format):
    plt.legend(policyNames, loc='upper left', fontsize=20)
    plt.xlabel('T')
    plt.ylabel(arg0)
    plt.savefig(results_dir + filename_local + arg2 + image_format, bbox_inches='tight')


def add_ticks(T):
    plt.figure(figsize=[8, 6])
    plt.grid(False)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
    plt.xlim([0, T])


def gen_plot(filename_local, file_ext_local='.npy'):
    global data
    data = np.load(results_dir + filename_local + file_ext_local, allow_pickle=True)[()]
    worker(filename_local=filename_local)


files = glob.glob('./experiment1.npy')
add_policy(['LinConKLUCB', 'LinConTS', 'LinConErrorTSNeural', 'LinConErrorTS', 'LinConErrorTSFullInformation'])
policyNames = ['LinConKLUCB', 'LinConTS', 'LinConErrorTS(Neural)', 'LinConErrorTS(MLE)',
               'LinConErrorTS(FullInformation)']
colorMapping = {'LinConKLUCBPolicy': u'#1f77b4', 'LinConTSPolicy': u'#ff7f0e', 'LinConErrorTSPolicy': u'#2ca02c',
                'LinConErrorTSNeuralPolicy': u'#8c564b', 'LinConErrorTSFullInformationPolicy': u'#d62728'}
# addPolicy(['LinConTS','LinConKLUCB', 'LinConErrorTS','LinConErrorTSNeural','LinConErrorTSFullInformation'])
# addPolicy(['LinConTS','LinConKLUCB','LinConErrorTSNeural'])
for file in files:
    gen_plot(Path(file).stem)
