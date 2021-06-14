import copy
import json
import warnings

import pandas as pd
import numpy as np

from histogram import plotHistogram
from longest_path import findLongestPath, Node
from samplers.SamplerFactory import get_sampler
from minimize_longest_path import basicMinimise, priorityMinimise, priorityLogMinimise, positionOnly
minimizer=priorityMinimise
kinds = ['Exponential', 'Geometric', 'Bernoulli', 'Bounded Pareto']
Ns = [10,]
Ms = [0.1, 0.2, 0.5, 0.8, 0.9]
iterations = [10000, 5000, 1000, 500]
samplers = []
dfs = {x: pd.DataFrame() for x in kinds}
with open("config.json", "r") as f:
    dists = json.load(f)
for dist in dists:
    samplers.append(get_sampler(dist["kind"], dist["specific_params"]))
for n, iters in zip(Ns, iterations):
    nodes = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            nodes[i][j] = Node(i, j, 0)
    for sampler in samplers:
        longest_paths = np.zeros(iters)
        minimazied_longest_paths = np.zeros(iters)
        for i in range(iters):
            weights = sampler.sample(n)
            nodes = findLongestPath(weights, nodes)
            longest_paths[i] = nodes[n - 1][n - 1].longest_path
        kind, params = sampler.kind_n_dict()
        plotHistogram(longest_paths, kind, params, n)
        sampler_dict = copy.deepcopy(params)
        mean = longest_paths.mean()
        var = longest_paths.var()
        sampler_dict['N'] = n
        sampler_dict['mean'] = mean
        sampler_dict['variance'] = var
        for m in Ms:
            for i in range(iters):
                weights = sampler.sample(n)
                new_weights = minimizer(weights, m)
                nodes = findLongestPath(new_weights, nodes)
                minimazied_longest_paths[i] = nodes[n - 1][n - 1].longest_path
            mean = minimazied_longest_paths.mean()
            var = minimazied_longest_paths.var()
            sampler_dict['mean: m=' + str(m)] = mean
            sampler_dict['variance: m=' + str(m)] = var
        print("finised "+kind)
        dfs[kind] = dfs[kind].append(sampler_dict, ignore_index=True)

for kind in kinds:
    dfs[kind].to_csv("./csvs/" + kind + ".csv")
