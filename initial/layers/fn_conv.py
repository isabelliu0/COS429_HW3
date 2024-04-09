import numpy as np
import scipy.signal

def fn_conv(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [filter_height] x [filter_width] x [filter_depth] x [num_filters] array
            params['b']: layer bias, [num_filters] x 1 array
        hyper_params: Optional, could include information such as stride and padding.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_filters] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """

    in_height, in_width, num_channels, batch_size = input.shape
    _, _, filter_depth, num_filters = params['W'].shape
    out_height = in_height - params['W'].shape[0] + 1
    out_width = in_width - params['W'].shape[1] + 1

    # print("params " + str(params['W'].shape[2]))
    # print("input shape " + str(input.shape[2]))
    # assert params['W'].shape[2] == input.shape[2], 'Filter depth does not match number of input channels'

    # Initialize
    output = np.zeros((out_height, out_width, num_filters, batch_size))
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}
    
    # TODO: FORWARD CODE
    #       Update output with values

    for n in range(batch_size):
        for f in range(num_filters):
            for c in range(num_channels):
                output[:, :, f, n] += scipy.signal.convolve(input[:, :, c, n], params['W'][:, :, c, f], mode='valid')
            output[:, :, f, n] += params['b'][f]



    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)
        
        # TODO: BACKPROP CODE
        #       Update dv_input and grad with values

        for n in range(batch_size):
            for f in range(num_filters):
                for c in range(num_channels):
                    # need to flip the weights back to the original orientation
                    dv_input[:, :, c, n] += scipy.signal.convolve(dv_output[:, :, f, n], np.flip(params['W'][:, :, c, f]), mode='full')
                    grad['W'][:, :, c, f] += scipy.signal.convolve(np.flip(input[:, :, c, n]), dv_output[:, :, f, n], mode='valid')
                grad['b'][f] += np.sum(dv_output[:, :, f, n])
        
        # average gradients across the batch
        grad['W'] /= batch_size
        grad['b'] /= batch_size



    return output, dv_input, grad
