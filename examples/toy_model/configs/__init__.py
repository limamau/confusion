from . import vp_mlp


def get_config(args):
    if args.config == "vp_mlp":
        return vp_mlp.get_config()
