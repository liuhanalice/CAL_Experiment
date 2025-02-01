import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import os
import torch

from dataset import *
from src.multi_GP import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GP parameters")
    
    # Add a positional argument for the number
    # parser.add_argument("InD_Dataset", type=str, help="The name of the InD dataset.")
    parser.add_argument("train_batch_size", type=int, help="train_batch_size", default=128)
    parser.add_argument("test_batch_size", type=int, help="test_batch_size", default=128)
    parser.add_argument("f_size", type=int, default=32, help="extracted_feature_size")

    args = parser.parse_args()
    
    num_classes = 10
    train_set, test_set, trloader, tsloader = MNIST_dataset(batch_size = args.train_batch_size, test_batch_size = args.test_batch_size)

    # mkdir directory to save
    parent_dir = os.getcwd()
    directory = 'out/f_' + str(args.f_size)
    path = os.path.join(parent_dir, directory)
    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory", directory, "is created")
    
    model_directory = 'model'
    model_path = os.path.join(parent_dir, model_directory)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("The new directory", model_directory, "is created")

    
    # Get all labels of training data for GP
    train_labels = train_set.targets
    test_labels = test_set.targets
    

    # Train MNIST
    epochs = 20
    net = ConvFeatNet(out_size=args.f_size)
    train(network=net, trloader=trloader, epochs=epochs, verbal=True)
    torch.save(net.state_dict(), os.path.join(model_directory, "MNIST_" + str(args.f_size) + "_net.pt"))

    train_features, train_scores, train_acc = scores(net, trloader, is_test=False)
    test_features, test_scores, test_acc= scores(net, tsloader, is_test=True) 
    print("Train accuracy: ", train_acc)
    print("Test accuracy: ", test_acc)
    
    # select first 20000 training data and first 5000 test data for GP
    in_train_features, in_train_scores = train_features[0: 20000], train_scores[0:20000]
    in_train_labels = train_labels[0:20000]
    in_train_data = np.concatenate((in_train_features.cpu().numpy(), in_train_scores.cpu().numpy()),1)
    in_train_data = pd.DataFrame(in_train_data)
    in_train_data['label'] = in_train_labels
    in_train_data.to_csv(directory + '/train.csv', index=False)
    print("train data stored")
    
    in_test_features, in_test_scores = test_features[0:5000], test_scores[0:5000]
    in_test_labels = test_labels[0:5000]
    in_test_data = np.concatenate((in_test_features.cpu().numpy(), in_test_scores.cpu().numpy()),1)
    in_test_data = pd.DataFrame(in_test_data)
    in_test_data['label'] = in_test_labels
    in_test_data.to_csv(directory + '/test.csv', index=False)
    print("test data stored")



print("\nEND")