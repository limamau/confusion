from argparse import Namespace

from . import ve


def get_config(
    args: Namespace,
) -> ve.Config:
    if args.config == "ve":
        return ve.Config()
    else:
        raise ValueError(f"Unknown config: {args.config}")
