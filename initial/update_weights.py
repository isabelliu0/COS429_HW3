import numpy as np

def update_weights(model, grads, hyper_params):
    '''
    Update the weights of each layer in your model based on the calculated gradients
    Args:
        model: Dictionary holding the model
        grads: A list of gradients of each layer in model["layers"]
        hyper_params: 
            hyper_params['learning_rate']
            hyper_params['weight_decay']: Should be applied to W only.
    Returns: 
        updated_model:  Dictionary holding the updated model
    '''
    num_layers = len(grads)
    a = hyper_params["learning_rate"]
    lmd = hyper_params["weight_decay"]

    # optional: momentum to accelerate
    momentum = hyper_params.get('momentum', 0.9) #0.9 is the default value if momentum not provided

    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients

    # initialize velocity to zero if not present
    if 'velocity' not in updated_model:
        updated_model['velocity'] = [None,] * num_layers
        for i in range(num_layers):
            updated_model['velocity'][i] = {'W': np.zeros_like(updated_model['layers'][i]['params']['W']),
                                            'b': np.zeros_like(updated_model['layers'][i]['params']['b'])}

    for i in range(num_layers):
        updated_model['velocity'][i]['W'] = momentum * updated_model['velocity'][i]['W'] - a * (grads[i]['W'] + lmd * updated_model['layers'][i]['params']['W'])
        updated_model['velocity'][i]['b'] = momentum * updated_model['velocity'][i]['b'] - a * grads[i]['b']
        # reference: momentum formulas based on lecture slides

        updated_model['layers'][i]['params']['W'] += updated_model['velocity'][i]['W']
        updated_model['layers'][i]['params']['b'] += updated_model['velocity'][i]['b']

    return updated_model