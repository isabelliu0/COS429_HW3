import numpy as np
from inference import inference
from data_utils import get_CIFAR10_data

import sys
sys.path += ['layers']
from loss_crossentropy import loss_crossentropy

_, _, test_data, test_labels = get_CIFAR10_data()

model = np.load('trained_model3.npz', allow_pickle=True)
model = dict(model)

#run it on test data to see accuracy
output, _ = inference(model, test_data)

test_loss, _ = loss_crossentropy(output, test_labels, None, True)
test_accuracy = np.mean((output.argmax(axis=0) == test_labels))

print(f'Test Loss = {test_loss:.3f}, Test Accuracy = {test_accuracy:.3f}')