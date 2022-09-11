import collections
from omegaconf import OmegaConf


def flatten_omegaconf(d, sep='_'):

    d = OmegaConf.to_container(d)

    obj = collections.OrderedDict()

    def recurse(t, parent_key=''):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i)
                        if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if type(v) in [int, float, str]}

    return
