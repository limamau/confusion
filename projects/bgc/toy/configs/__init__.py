from argparse import Namespace
from typing import Union

from . import cve, cvp, ve, vp


def get_config(
    args: Namespace,
) -> Union[
    cve.Config,
    cvp.Config,
    ve.Config,
    vp.Config,
]:
    if args.config == "cve":
        return cve.Config()
    elif args.config == "cvp":
        return cvp.Config()
    elif args.config == "ve":
        return ve.Config()
    elif args.config == "vp":
        return vp.Config()
    else:
        raise ValueError(f"Unknown config: {args.config}")
