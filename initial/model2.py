from data_utils import get_CIFAR10_data

import numpy as np
from init_layers import init_layers
from init_model import init_model
from train import train
import matplotlib.pyplot as plt
from inference import inference

import sys
sys.path += ['layers']
from loss_crossentropy import loss_crossentropy

def main():
    train_data, train_labels, test_data, test_labels = get_CIFAR10_data()
    print("Train data shape:", train_data.shape)


    # input: 32 x 32 x 3 image from cifar-10
    # conv1: 32 x 32 x 3 --> 30 x 30 x 12 (filter depth = 3)
    # relu: no change to dimensions
    # pool1: 30 x 30 x 12 --> 15 x 15 x 12
    # conv2: 15 x 15 x 12 --> 13 x 13 x 16 (filter depth = 12)
    # relu: no change
    # pool2: 13 x 13 x 16 --> 6 x 6 x 16
    # flatten: 3D to 1D, --> 576
    # linear1: 576 --> 128
    # relu: no change
    # linear2: 128 --> 10
    # softmax: no change

    l = [init_layers('conv', {'filter_size': 3,
                              'filter_depth': 3,
                              'num_filters': 12}),
         init_layers('relu', {}),
         init_layers('pool', {'filter_size': 2,
                              'stride': 2}),
         init_layers('conv', {'filter_size': 3, 
                              'filter_depth': 12,
                              'num_filters': 16}),
         init_layers('dropout', {'rate': 0.5}), # add dropout layer
         init_layers('relu', {}),
         init_layers('pool', {'filter_size': 2,
                              'stride': 2}),
         init_layers('flatten', {}),
         init_layers('linear', {'num_in': 576,
                                'num_out': 128}),
         init_layers('relu', {}),
         init_layers('linear', {'num_in': 128,
                                'num_out': 10}),
         init_layers('softmax', {})]

    model = init_model(l, [32, 32, 3], 10, True)

    hyper_params = {"learning_rate": 0.01, "weight_decay": 0.0005, "batch_size": 128}

    model, train_loss, test_loss = train(model, train_data, train_labels, test_data, test_labels, hyper_params, numIters=1000)

    #plot the training loss
    plt.plot(train_loss)
    plt.title("Training Loss")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Train Loss")

    plt.savefig("train_loss_model3.png")

     #plot the training loss
    plt.plot(test_loss)
    plt.title("Testing Loss")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Test Loss")

    plt.savefig("test_loss_model3_.png")

    #run the final model on test data to see accuracy
    output, _ = inference(model, test_data)

    test_loss, _ = loss_crossentropy(output, test_labels, None, True)
    test_accuracy = np.mean((output.argmax(axis=0) == test_labels))

    print(f'Test Loss = {test_loss:.3f}, Test Accuracy = {test_accuracy:.3f}')

    np.savez('trained_model3.npz', **model)

if __name__ == '__main__':
    main()
