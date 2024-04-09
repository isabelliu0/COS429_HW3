import numpy as np
import scipy.signal

def fn_dropout(input, params, hyper_params, backprop, dv_output=None):
    rate = params['rate']
    # randomly set some values to 0 w.r.t dropout rate
    mask = (np.random.rand(*input.shape) >= rate) / (1 - rate)
    dv_input = None
    grad = {'W': None, 'b': None}

    if backprop:
        # backpropagation: scale the upstream gradient by the mask
        # gradients are only backpropagated through the neurons that
        # were not "dropped out" during the forward pass
        dv_input = dv_output * mask
    else:
        # forward pass: apply the mask to input
        input = input * mask

    return input, dv_input, grad
