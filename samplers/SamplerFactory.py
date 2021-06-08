import string
from typing import Dict

from samplers.BoundedPareto import Boundedpareto
from samplers.Sampler import Sampler
from samplers.bernoulli import Bernoulli
from samplers.exp import Expon
from samplers.geo import Geo

kinds = {'Exponential': Expon, 'Geometric': Geo, 'Bernoulli': Bernoulli, 'Bounded Pareto': Boundedpareto}


def get_sampler(kind: string, specific_params: Dict) -> Sampler:
    return kinds[kind](**specific_params)
