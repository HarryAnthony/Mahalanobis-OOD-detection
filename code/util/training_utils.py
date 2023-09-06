"""
Helper functions for training a neural network.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wide_resnet import Wide_ResNet
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import resnet18, efficientnet_v2_s, vgg11, vgg16_bn, resnet34, resnet50
from csv import writer

def get_network_architecture(args,num_classes,suffix):
    """
    Returns the network architecture and the file name for the network.

    Parameters
    ----------
    args : argparse
        The arguments that are passed to the program.
    num_classes : int
        The number of classes in the dataset.
    suffix : str
        The suffix for the file name.

    Returns
    -------
    net : torch.nn.Module
        The network architecture.
    file_name : str
        The file name for the network.

    """
    #Load the required neural network architecture
    if args.net_type == 'wide-resnet':
        net = Wide_ResNet(args.depth, args.widen_factor,args.dropout, num_classes)
        file_name = str(args.net_type)+'-'+str(args.depth)+'x'+str(args.widen_factor)+'_'+str(suffix)
    elif args.net_type == 'ResNet18':
        net = resnet18(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'ResNet34':
        net = resnet34(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'ResNet50':
        net = resnet50(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'efficientnet':
        net = efficientnet_v2_s(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'vgg11':
        net = vgg11(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'vgg16_bn':
        net = vgg16_bn(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    else:
        raise Exception("Network architecture  not avaliable, pick from list [wide-resnet,ResNet18,efficientnet,vgg11,vgg16]")
    
    net = add_dropout_network_architechture(net,net_info = {'Model architecture': args.net_type,
                                                            'Dropout': args.dropout,
                                                            'Act_func_Dropout': args.act_func_dropout,
                                                            'num_classes': num_classes})
 
    return net, file_name


def add_dropout_network_architechture(net,net_info):
    """
    Add dropout to the network architecture.

    Parameters
    ----------
    net : torch.nn.Module
        The net to add dropout to.
    net_info : dict
        A dictionary containing the net information.

    Returns
    -------
    net : torch.nn.Module
        The net with dropout added.
    """
    if net_info['Act_func_Dropout'] > 0:
        net = append_dropout(net,rate=net_info['Act_func_Dropout'])
    if net_info['Dropout'] > 0:    
        if net_info['Model architecture'] == 'ResNet18':
            num_ftrs = net.fc.in_features  # Get the number of input features
            net.fc = nn.Sequential(
                nn.Dropout(p=net_info['Dropout']),
                nn.Linear(num_ftrs, net_info['num_classes'])  # Modify the output size as needed
                )
    return net


def append_dropout(model, rate=0.2):
    """
    Append dropout to the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to append dropout to.
    rate : float, optional
        The dropout rate. The default is 0.2.

    Returns
    -------
    model : torch.nn.Module
        The model with dropout appended.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, rate)  # Recursively apply to child modules
        if isinstance(module, nn.ReLU):
            # Insert a Dropout2d layer after the ReLU
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
            setattr(model, name, new)
    return model


def seed_used_before(seed_num,save_dir):
    """
    Checks to see if the seed has been used before. If it has been used before
    then the function returns True. If it has not been used before then the
    function returns False.

    Parameters
    ----------
    seed_num : int
        The seed number to check.
    save_dir : str
        The directory to check for previous experiments.

    Returns
    -------
    bool
        True if the seed has been used before, False if it has not been used before.
    """

    save_dir =os.listdir(save_dir) #Get list of previous experiments
    #Check the last few characters match the seed_num
    used_before = np.array([x[-4-len(str(seed_num)):-4] == str(seed_num) and
     x[-5-len(str(seed_num))].isdigit() is False for x in save_dir]).sum()
    return used_before > 0


def select_experiment_seed(seed_num,save_dir,allow_repeats=False):
    """
    Selects a seed for the experiment. If seed_num is not 'random' then the seed
    is checked to see if it has been used before. If it has been used before and 
    allow_repeats is False, then an exception is raised. If seed_num is 'random' 
    then a random seed is chosen and checked to see if it has been used before. 
    If it has been used before then a new random seed is chosen until a seed that 
    has not been used before is found.

    Parameters
    ----------
    seed_num : str or int
        Seed number to use for the experiment. If 'random' then a random seed is
        chosen.
    save_dir : str
        Directory to save the experiment results.
    allow_repeats : bool, optional
        If True then the same seed can be used multiple times. The default is False.

    Returns
    -------
    int
        Seed number to use for the experiment.

    """
    if seed_num != 'random':
        if seed_num.isdigit() == False:
            raise Exception('Seed must be an integer')
        if seed_used_before(seed_num,save_dir) and allow_repeats==False:
            raise Exception('Seed '+str(seed_num)+' has already been used. Chose another seed or allow repeats.')
        return int(seed_num)
    elif allow_repeats==False:
        seed_num = np.random.randint(0,100000)
        while seed_used_before(seed_num,save_dir):
                seed_num = np.random.randint(0,100000)
    return int(seed_num)


def get_class_weights(df):
    """
    get class weights for imbalanced dataset

    Parameters
    ----------
    df : pandas dataframe
        dataframe containing the class labels.

    Returns
    -------
    np.array
        array of class weights.

    """
    class_sample_count = np.array([len(np.where(df['class'] == t)[0]) for t in np.unique(df['class'])])
    weight = 1. - (class_sample_count / len(df))
    return np.array([weight[t] for t in df['class']])


def record_model(model_record_filename:str,list:list):
    """
    record model details in a csv file

    Parameters
    ----------
    model_record_filename : str
        filename of the csv file to record the model details.
    list : list
        list of model details to be recorded.
    """
    with open(model_record_filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(list)
        f_object.close()


def initialise_network(net,initialisation_method='he'):
    """
    Initialise the weights of the network.

    Parameters
    ----------
    net : torch.nn.Module
        The network to initialise.
    initialisation_method : str, optional
        Name of the initialisation method to use. The default is 'he'.

    Returns
    -------
    net : torch.nn.Module
        The initialised network.
    """
    if initialisation_method in ['glorot', 'he', 'lecun']:
        for module in net.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                if initialisation_method == 'glorot':
                    torch.nn.init.xavier_uniform_(module.weight)
                elif initialisation_method == 'he':
                    torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                elif initialisation_method == 'lecun':
                    torch.nn.init.normal_(module.weight, mean=0, std=1 / (module.weight.shape[1] ** 0.5))
    return net


def get_criterion(criterion_name='CrossEntropyLoss', label_smoothing=0.0):
    """
    Get the criterion (loss function) for the model.

    Parameters
    ----------
    criterion_name : str, optional
        The name of the criterion. The default is 'CrossEntropyLoss'.
    label_smoothing : float, optional
        The label smoothing value. The default is 0.

    Returns
    -------
    criterion : torch.nn.modules.loss
        The criterion (loss function) for the model.
    """
    criterion_dict = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        'BCELoss': nn.BCELoss(),
        'MSELoss': nn.MSELoss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
        'NLLLoss': nn.NLLLoss(),
        'smoothL1Loss': nn.SmoothL1Loss()
    }

    if criterion_name in criterion_dict:
        criterion = criterion_dict[criterion_name]
    else:
        raise Exception('Criterion %s unknown.' % criterion_name)

    return criterion


def get_optimiser_scheduler(net, args, cf, trainloader, num_epochs):
    
    momentum = cf.momentum
    weight_decay = cf.weight_decay
    # Set the optimiser
    if args.optimiser == 'SGD':
        optimiser = optim.SGD(net.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)
    elif args.optimiser == 'Adam':
        optimiser = optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
    elif args.optimiser == 'AdamW':
        optimiser = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=weight_decay)
    else:
        raise Exception('Optimiser %s unknown. Choose from SGD, AdamW, and Adam.' % args.optimiser)

    # Set the scheduler
    if args.scheduler == 'MultiStepLR':
        lr_milestones = cf.lr_milestones
        lr_gamma = cf.lr_gamma
        scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=lr_milestones, gamma=lr_gamma)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.2, patience=30)
    elif args.scheduler == 'OneCycleLR':
        max_lr = float(np.max([float(args.max_lr), args.lr * 10]))
        steps_per_epoch = len(trainloader)
        div_factor = 10
        scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=num_epochs, div_factor=div_factor)
    elif args.scheduler == 'ConstantLR':
        scheduler = lr_scheduler.ConstantLR(optimiser, total_iters=num_epochs)
    else:
        raise Exception('Scheduler %s unknown. Choose from MultiStepLR, OneCycleLR, and ReduceLROnPlateau' % args.scheduler)

    return optimiser, scheduler


def set_activation_function(net, activation_function='ReLU'):
    """
    Set the activation function of the net.

    Parameters
    ----------
    net : torch.nn.Module
        The net to set the activation function of.
    activation_function : str
        The name of the activation function.

    Returns
    -------
    net : torch.nn.Module
        The net with the activation function set.
    """
    #Dictionary of activation functions
    activation_mapping = {
        'LeakyReLU': nn.LeakyReLU,
        'SiLU': nn.SiLU,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
        'CELU': nn.CELU
    }

    #Raise error if activation function is invalid
    if activation_function != 'ReLU' and activation_function not in activation_mapping.keys():
        raise ValueError(f'Invalid activation function: {activation_function}')

    #If activation function is ReLU, do nothing
    if activation_function == 'ReLU':
        return net
    
    activation_func = activation_mapping.get(activation_function)
    #Otherwise, convert the activation function
    net = convert_activation(net,activation_func)
    return net


def convert_activation(net,activation_func):
    """
    Convert the activation function of the net.

    Parameters
    ----------
    model : torch.nn.Module
        The net to convert the activation function of.
    activation_mapping : dict
        The dictionary of activation functions.
    activation_function : str
        The name of the activation function.

    """
    for child_name, child in net.named_children():
        if isinstance(child, nn.ReLU):
            setattr(net, child_name, activation_func())
        else:
            convert_activation(child)
    return net

