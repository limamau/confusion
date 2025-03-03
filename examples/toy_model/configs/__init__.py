from . import ve, vp


def get_config(args):
    if args.config == "ve":
        return ve.get_config()
    elif args.config == "vp":
        return vp.get_config()
