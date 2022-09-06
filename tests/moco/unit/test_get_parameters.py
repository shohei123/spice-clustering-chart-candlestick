from moco.parameters import get_parameters
from rich import print


def test_get_parametets():
    params = get_parameters()
    print(params)