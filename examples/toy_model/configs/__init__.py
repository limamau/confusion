from . import cve, cvp, ve, vp


def get_config(args):
    if args.config == "cve":
        return cve.get_config()
    elif args.config == "cvp":
        return cvp.get_config()
    elif args.config == "ve":
        return ve.Config()
    elif args.config == "vp":
        return vp.get_config()
