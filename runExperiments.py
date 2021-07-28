import copy
import json
import os
import warnings

import pandas as pd
import numpy as np
import time

from histogram import plotHistogram
from longest_path_c import findLongestPath
from samplers.SamplerFactory import get_sampler
from minimize_longest_path import basicMinimise, priorityMinimise, priorityLogMinimise, positionOnly
from minimizer import Minimizer
from Yuval_minimazier import YMinimizer

mean_str = 'mean: m='
var_str = 'variance: m='
kinds = ['Bounded Pareto', ]  # 'Exponential','Geometric', 'Bernoulli']  # , 'Bounded Pareto']
Ns = [10000]
diffs = [0, 0.1, 0.1, 0.3, 0.3, 0.1]
Ms = ['0', '0.1', '0.2', '0.5', '0.8', '0.9']

iterations = [100]
samplers = []
dfs = {x: pd.DataFrame() for x in kinds}

with open("config3.json", "r") as f:
    dists = json.load(f)
for dist in dists:
    samplers.append(get_sampler(dist["kind"], dist["specific_params"]))
try:
    for n, iters in zip(Ns, iterations):
        minimizer = YMinimizer("double_diag_log", n)
        nodes = np.zeros((n, n), dtype=np.float32)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("start time:" + current_time)
        for sampler in samplers:
            weights = sampler.sample(n)
            kind, params = sampler.kind_n_dict()
            sampler_dict = copy.deepcopy(params)
            longest_paths = {'0': np.zeros(iters), '0.1': np.zeros(iters), '0.2': np.zeros(iters),
                             '0.5': np.zeros(iters),
                             '0.8': np.zeros(iters), '0.9': np.zeros(iters)}
            for i in range(iters):
                weights = sampler.sample(n)
                for diff, m in zip(diffs, Ms):
                    minimizer.minimize(weights, diff)
                    findLongestPath(weights, nodes, n)
                    longest_paths[m][i] = nodes[n - 1][n - 1]
            for m in Ms:
                plotHistogram(longest_paths[m], kind, params, n, m)
                mean = longest_paths[m].mean()
                var = longest_paths[m].var()
                sampler_dict['N'] = n
                sampler_dict[mean_str + m] = mean
                sampler_dict[var_str + m] = var

            print("finised " + kind)
            dfs[kind] = dfs[kind].append(sampler_dict, ignore_index=True)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("finish time:" + current_time)
except KeyboardInterrupt:
    pass
for kind in kinds:
    filename = "./csvs/" + kind + ".csv"
    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    dfs[kind].to_csv(filename, mode=append_write)
