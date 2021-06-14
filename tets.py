import json
import pandas as pd
from samplers.SamplerFactory import get_sampler



sampler = get_sampler("Bounded Pareto",{"alpha":0.5,"h":100})
print(sampler.sample(size=3))
print(sampler.sample1(size=3))
