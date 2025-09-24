"""
Data loading and preprocessing module for Human Activity Recognition
"""
import numpy as np
import torch
from pandas import read_csv
from numpy import dstack
from torch.utils.data import DataLoader, TensorDataset


def load_file(filepath):
    """Load a single file as a numpy array"""
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


def load_group(filenames, prefix=''):
    """Load a list of files and return as a 3d numpy array"""
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


def load_dataset_group(group, prefix=''):
    """Load a dataset group, such as train or test"""
    filepath = prefix + group + '/Inertial Signals/'
    print('File Path:', filepath)
    
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y


def prepare_data(data_path='UCIDataset/', batch_size=128):
    """
    Load and prepare the dataset for training and testing
    
    Args:
        data_path (str): Path to the UCI HAR Dataset
        batch_size (int): Batch size for data loaders
    
    Returns:
        tuple: (train_loader, test_loader, data_info)
    """
    # Load all train and test data
    X_train, Y_train = load_dataset_group('train', data_path)
    X_test, Y_test = load_dataset_group('test', data_path)
    
    # Zero-offset class values (convert from 1-6 to 0-5)
    Y_train = Y_train - 1
    Y_test = Y_test - 1
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    Y_train = torch.LongTensor(Y_train.flatten())
    Y_test = torch.LongTensor(Y_test.flatten())
    
    print('X_train.shape:', X_train.shape)
    print('Y_train.shape:', Y_train.shape)
    print('X_test.shape:', X_test.shape)
    print('Y_test.shape:', Y_test.shape)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Data information
    data_info = {
        'n_timesteps': X_train.shape[1],
        'n_features': X_train.shape[2],
        'n_outputs': 6,  # 6 classes for activity recognition
        'n_train_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0]
    }
    
    return train_loader, test_loader, data_info


def get_class_names():
    """Get the class names for human activities"""
    return ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']