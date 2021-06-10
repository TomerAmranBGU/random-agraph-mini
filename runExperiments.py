import copy
import json
from minimizer import Minimizer
import warnings

import pandas as pd
import numpy as np

from histogram import plotHistogram
from longest_path import findLongestPath
from samplers.SamplerFactory import get_sampler
from minimize_longest_path import basicMinimise, priorityMinimise,priorityLogMinimise ,positionOnly
kinds = ['Exponential', 'Geometric', 'Bernoulli', 'Bounded Pareto']
# kinds = ['Exponential', 'Geometric', 'Bernoulli']
# Ns = [10, 100, 1000, 1000]
# kinds = ['Exponential']
Ns = [30]
Ms = [0.1, 0.2, 0.5,0.8 ,0.9]
minimizers_names = [
            'no_minimizer',
            'weights',
            'diag',
            'diag_log',
            'double_diag',
            'double_diag_log',
            'position_only']
iters = 1000
samplers = []
dfs = {x: pd.DataFrame() for x in kinds}
with open("config.json", "r") as f:
    dists = json.load(f)
for dist in dists:
    samplers.append(get_sampler(dist["kind"], dist["specific_params"]))
for m in Ms:
    print('###',m,'###')
    for minimizer_name in minimizers_names:
        minimizer = Minimizer(minimizer_name)
        print()
        print(minimizer_name)
        for n in Ns:
            for sampler in samplers:
                longest_paths = np.zeros(iters)
                for i in range(iters):
                    weights = sampler.sample(n)
                    #tomer
                    weights = minimizer.minimize(weights, m)
                    nodes = findLongestPath(weights)
                    longest_paths[i] = nodes[n - 1][n - 1].longest_path
                kind, params = sampler.kind_n_dict()
                plotHistogram(longest_paths, kind, params, n)
                sampler_dict = copy.deepcopy(params)
                mean = longest_paths.mean()
                var = longest_paths.var()
                sampler_dict['N'] = n
                sampler_dict['mean'] = mean
                sampler_dict['variance'] = var
                dfs[kind] = dfs[kind].append(sampler_dict, ignore_index=True)
                print(kind+" "+"mean: "+str(mean)+",var: "+str(var))

for kind in kinds:
    dfs[kind].to_csv("./" + kind + ".csv")
