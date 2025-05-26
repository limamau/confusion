from argparse import Namespace
from typing import Union

from . import ve, vp


def get_config(
    args: Namespace,
) -> Union[
    # edm.Config,
    ve.Config,
    vp.Config,
]:
    # if args.config == "edm":
    #     return edm.Config()
    if args.config == "ve":
        return ve.Config()
    elif args.config == "vp":
        return vp.Config()
    else:
        raise ValueError(f"Unknown config: {args.config}")
