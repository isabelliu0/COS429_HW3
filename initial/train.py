import sys
sys.path += ['layers']
import numpy as np
from loss_crossentropy import loss_crossentropy

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = False

# You can modify the imports of this section to indicate
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ['pyc_code']
    from inference_ import inference
    #from inference import inference
    from calc_gradient_ import calc_gradient
    #from calc_gradient import calc_gradient
    from update_weights_ import update_weights
    #from update_weights import update_weights
else:
    # import your own implementation
    from inference import inference
    from calc_gradient import calc_gradient
    from update_weights import update_weights
######################################################

def train(model, input, label, test_input, test_label, params, numIters):
    '''
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [num_inputs]
        label: [num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
    '''
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", .01)
    # Weight decay
    wd = params.get("weight_decay", .0005)
    # Batch size
    batch_size = params.get("batch_size", 128)
    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", 'model.npz')

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr,
                     "weight_decay": wd }

    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,))
    test_loss = np.zeros((numIters,))

    # additional things to make the process better and clearer
    prev_loss = float('inf')
    loss_threshold = 0.0001 # for loss plateau
    accuracy_goal = 0.95


    for i in range(numIters):
        # TODO: One training iteration
        # Steps:
        #   (1) Select a subset of the input to use as a batch
        #   (2) Run inference on the batch
        #   (3) Calculate loss and determine accuracy
        #   (4) Calculate gradients
        #   (5) Update the weights of the model
        # Optionally,
        #   (1) Monitor the progress of training
        #   (2) Save your learnt model, using ``np.savez(save_file, **model)``

        # (1)
        batch_start = (i * batch_size) % num_inputs
        batch_end = batch_start + batch_size
        batch_input = input[:, :, :, batch_start:batch_end]
        batch_label = label[batch_start:min(batch_end, num_inputs)]

        # (2)
        output, activations = inference(model, batch_input)

        # (3)
        loss[i], dv_output = loss_crossentropy(output, batch_label, None, True)
        accuracy = np.mean((output.argmax(axis=0) == batch_label))

        # display training progress
        if i % 100 == 0:
            print(f'Iteration {i}: Training Loss = {loss[i]:.3f}, Training Accuracy = {accuracy:.3f}')

        # display test loss every 100 iterations
        if i % 100 == 0:
            test_output, _ = inference(model, test_input)
            test_loss[i], _ = loss_crossentropy(test_output, test_label, None, False)
            test_accuracy = np.mean((test_output.argmax(axis=0) == test_label))
              
            print(f'Iteration {i}: Test Loss = {test_loss[i]:.3f}, Test Accuracy = {test_accuracy:.3f}')
        
        # check for loss plateau
        if abs(prev_loss - loss[i]) < loss_threshold:
            print('Loss plateaued, adjusting learning rate...')
            update_params["learning_rate"] /= 2
        
        # check for goal accuracy
        if accuracy >= accuracy_goal:
            print('Goal accuracy reached, stopping training...')
            break
        
        # (4)
        grads = calc_gradient(model, batch_input, activations, dv_output)

        # (5)
        model = update_weights(model, grads, update_params)

        # save learnt model
        if i % 1000 == 0:
            np.savez(save_file, **model)

        prev_loss = loss[i] #update previous loss


    return model, loss, test_loss