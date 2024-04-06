from data_utils import get_CIFAR10_data

import sys
sys.path += ['layers']
import numpy as np
from init_layers import init_layers
from init_model import init_model
from train import train

def main():
    train_data, train_labels, test_data, test_labels = get_CIFAR10_data()

    # input: 32 x 32 x 3 image from cifar-10
    # conv1: 32 x 32 x 3 --> 30 x 30 x 32 (filter depth = 3)
    # relu: no change to dimensions
    # pool1: 30 x 30 x 32 --> 15 x 15 x 32
    # conv2: 15 x 15 x 32 --> 13 x 13 x 64 (filter depth = 32)
    # relu: no change
    # pool2: 13 x 13 x 64 --> 6 x 6 x 64
    # flatten: 3D to 1D, --> 2304
    # linear1: 2304 --> 512
    # relu: no change
    # linear2: 512 --> 10
    # softmax: no change

    l = [init_layers('conv', {'filter_size': 3,
                              'filter_depth': 3,
                              'num_filters': 32}),
         init_layers('relu', {}),
         init_layers('pool', {'filter_size': 2,
                              'stride': 2}),
         init_layers('conv', {'filter_size': 3, 
                              'filter_depth': 32,
                              'num_filters': 64}),
         init_layers('relu', {}),
         init_layers('pool', {'filter_size': 2,
                              'stride': 2}),
         init_layers('flatten', {}),
         init_layers('linear', {'num_in': 2304,
                                'num_out': 512}),
         init_layers('relu', {}),
         init_layers('linear', {'num_in': 512,
                                'num_out': 10}),
         init_layers('softmax', {})]

    model = init_model(l, [32, 32, 3], 10, True)

    hyper_params = {"learning_rate": 0.01, "weight_decay": 0.0005, "batch_size": 128}

    model, loss = train(model, train_data, train_labels, hyper_params, numIters=10000)

    np.savez('trained_model.npz', **model)

if __name__ == '__main__':
    main()
