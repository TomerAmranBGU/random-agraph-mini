import json
import warnings

import pandas as pd
import numpy as np

from longest_path import findLongestPath
from samplers.SamplerFactory import get_sampler

Ns = [10, 100, 1000, 1000]
samplers = []
dfs = {x: pd.DataFrame() for x in ['Exponential', 'Geometric', 'Bernoulli', 'Bounded Pareto']}
with open("config1.json", "r") as f:
    dists = json.load(f)
for dist in dists:
    samplers.append(get_sampler(dist["kind"], dist["specific_params"]))
for n in Ns:
    for sampler in samplers:
        longest_paths = np.zeros(10000)
        for i in range(10000):
            print(i)
            weights = sampler.sample(n)
            nodes = findLongestPath(weights)
            longest_paths[i]=nodes[n-1][n-1].longest_path
        kind, sampler_dict = sampler.kind_n_dict()
        sampler_dict['N'] = n
        sampler_dict['mean'] = longest_paths.mean()
        sampler_dict['variance'] = longest_paths.var()
        dfs[kind] = dfs[kind].append(sampler_dict,ignore_index=True)
        print(kind)
        print(dfs[kind])
