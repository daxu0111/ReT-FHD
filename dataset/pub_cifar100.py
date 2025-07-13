

import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
import copy

random.seed(1)
np.random.seed(1)




def generate_cifar100(dir_path, datasize, classes=np.arange(10)):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    

    full_dataset = torchvision.datasets.CIFAR100(root=dir_path, train=True, transform=transform, download=True)
    

    selected_cifar10_classes = [96,93,80,42,84,68,44,43,97,99]


    cifar10_dataset = create_cifar10_dataset(full_dataset, selected_cifar10_classes)
    

    partial_dataset,total_samples = generate_partial_data(cifar10_dataset, classes, datasize)
    print(f"Total samples in the partial dataset: {total_samples}")

    save_partial_dataset(dir_path, partial_dataset)


def generate_partial_data(dataset, classes, datasize=1000):
    targets = dataset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    
    data_indices = []
    total_samples = 0

    for c in classes:
        idx_c = list(np.where(targets == c)[0])
        sample_size = min(len(idx_c), datasize // len(classes))  # Adjust sample size for each class
        if sample_size > 0:
            selected_indices = np.random.choice(idx_c, size=sample_size, replace=False)
            data_indices.extend(selected_indices)
            total_samples += sample_size

    if len(data_indices) == 0:
        raise ValueError("No data found for the specified classes")

    partial_dataset = copy.deepcopy(dataset)
    partial_dataset.data = partial_dataset.data[data_indices]
    partial_dataset.targets = np.array(partial_dataset.targets)[data_indices]
    
    return partial_dataset, total_samples 
def save_partial_dataset(dir_path, dataset):
    images = dataset.data  # No need to convert, as it's already a numpy array
    labels = np.array(dataset.targets)  # Convert targets to numpy array if not already
    np.savez_compressed(os.path.join(dir_path, '0.npz'), x=images, y=labels)


def create_cifar10_dataset(dataset, selected_classes):
    filtered_data = []
    filtered_targets = []

    for i in range(len(dataset.targets)):
        if dataset.targets[i] in selected_classes:
            filtered_data.append(dataset.data[i])
            filtered_targets.append(selected_classes.index(dataset.targets[i]))

    cifar10_dataset = copy.deepcopy(dataset)
    cifar10_dataset.data = np.array(filtered_data)
    cifar10_dataset.targets = filtered_targets
    
    return cifar10_dataset


dir_path = "dataset/cifar100_public/"
datasize = 1000
classes = np.arange(10)
generate_cifar100(dir_path, datasize, classes)

