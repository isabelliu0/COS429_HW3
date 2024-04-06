import numpy as np

def calc_gradient(model, input, layer_acts, dv_output):
    '''
    Calculate the gradient at each layer, to do this you need dv_output
    determined by your loss function and the activations of each layer.
    The loop of this function will look very similar to the code from
    inference, just looping in reverse.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        layer_acts: A list of activations of each layer in model["layers"]
        dv_output: The partial derivative of the loss with respect to each element in the output matrix of the last layer.
    Returns: 
        grads:  A list of gradients of each layer in model["layers"]
    '''
    num_layers = len(model["layers"])
    grads = [None,] * num_layers

    # TODO: Determine the gradient at each layer.
    #       Remember that back-propagation traverses 
    #       the model in the reverse order.

    # Start with the gradient from the output
    dv_next = dv_output

    for i in range(num_layers-1, -1, -1): #reverse order
        fwd_fn = model['layers'][i]['fwd_fn']
        params = model['layers'][i]['params']
        hyper_params = model['layers'][i]['hyper_params']

        _, dv_input, grad = fwd_fn(layer_acts[i], params, hyper_params, True, dv_next)

        grads[i] = grad
        dv_next = dv_input

    return grads