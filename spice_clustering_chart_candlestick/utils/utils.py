import collections
from omegaconf import OmegaConf


def flatten_omegaconf(conf, sep="_"):

    conf = OmegaConf.to_container(conf)

    dic = collections.OrderedDict()

    def recurse(conf, parent_key=""):
        if isinstance(conf, list):
            for i in range(len(conf)):
                recurse(
                    conf[i], parent_key + sep + str(i)
                    if parent_key else str(i)
                )
        elif isinstance(conf, dict):
            for k, v in conf.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            dic[parent_key] = conf

    recurse(conf)

    dic = {k: v for k, v in dic.items()}

    return dic
