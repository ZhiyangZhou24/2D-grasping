def get_network(network_name):
    network_name = network_name.lower()
    # Original GR-ConvNet
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == 'grconvnet2':
        from .grconvnet2 import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == 'grconvnet3':
        from .grconvnet3 import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet4':
        from .grconvnet4 import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet3next':
        from .grconvnet3next import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
        # Inverted GR-ConvNet
    elif network_name == 'grconvnet3_sknet':
        from .grconvnet3_sknet import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'grconvnet3_seresunet':
        from .grconvnet3_seresunet import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'grconvnet3_seresunet_slim':
        from .grconvnet3_seresunet import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'grconvnet3_pico_u':
        from .grconvnet3_pico_u import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'grconvnet3_imp':
        from .grconvnet3_imp import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'grconvnet3_imp_pp':
        from .grconvnet3_imp_pp import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'unet':
        from .unet import GenerativeResnet
        return GenerativeResnet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
