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

    for i in range(1, num_layers): #each layer is a dict
        fwd_fn = model['layers'][i]['fwd_fn']
        params = model['layers'][i]['params']
        hyper_params = model['layers'][i]['hyper_params']

        a, _, _ = fwd_fn(activations[i-1], params, hyper_params, False)
        activations[i] = a


    output = activations[-1]
    return output, activations
