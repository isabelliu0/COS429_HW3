from data_utils import get_CIFAR10_data

# import sys
# sys.path += ['layers']
import numpy as np
# from init_layers import init_layers
# from init_model import init_model
# from train import train
import matplotlib.pyplot as plt
# from inference import inference
# from loss_crossentropy import loss_crossentropy

test_loss = [2.911, 2.825, 1.905, 2.074, 1.975, 1.962, 1.989, 2.046, 2.015, 1.926, 1.993]

#plot the training loss
plt.figure()
plt.plot(np.arange(0, len(test_loss)*100, 100), test_loss)
plt.title("Test Loss")
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")

plt.savefig("test_loss3.png")