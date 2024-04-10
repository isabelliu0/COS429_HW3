import numpy as np

def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model['layers'])
    activations = [None,] * num_layers

    # TODO: FORWARD PROPAGATION CODE

    activations[0] = input

    for i, layer in enumerate(model['layers'], start = 1): #each layer is a dict
        fwd_fn = layer['fwd_fn']
        params = layer.get('params', {})
        hyper_params = layer.get('hyper_params', {})

        a, _, _ = fwd_fn(activations[i-1], params, hyper_params, False)

        if i < num_layers:
            activations[i] = a
        else:
            output = a


    return output, activations
