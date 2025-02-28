from . import conditional_mixer
from . import unconditional_mixer
from . import conditional_unet
from . import unconditional_unet

def get_config(args, imgs_shape):
    if args.config == "conditional_mixer":
        return conditional_mixer.get_config(imgs_shape)
    elif args.config == "unconditional_mixer":
        return unconditional_mixer.get_config(imgs_shape)
    elif args.config == "conditional_unet":
        return conditional_unet.get_config(imgs_shape)
    elif args.config == "unconditional_unet":
        return unconditional_unet.get_config(imgs_shape)