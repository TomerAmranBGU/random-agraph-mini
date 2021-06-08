import copy
import json
import warnings

import pandas as pd
import numpy as np

from histogram import plotHistogram
from longest_path import findLongestPath
from samplers.SamplerFactory import get_sampler

kinds = ['Exponential', 'Geometric', 'Bernoulli', 'Bounded Pareto']
Ns = [10, 100, 1000, 1000]
samplers = []
dfs = {x: pd.DataFrame() for x in kinds}
with open("config.json", "r") as f:
    dists = json.load(f)
for dist in dists:
    samplers.append(get_sampler(dist["kind"], dist["specific_params"]))
for n in Ns:
    for sampler in samplers:
        longest_paths = np.zeros(10000)
        for i in range(10000):
            weights = sampler.sample(n)
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
