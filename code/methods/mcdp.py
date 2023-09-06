import torch
import torch.nn as nn
import numpy as np 
from torch.autograd import Variable
from util.general_utils import print_progress
from util.evaluate_network_utils import softmax

def enable_dropout(net, dropout_rate=0.3,two_dim_dropout_rate='same',**kwargs):
    """ 
    Function to enable the dropout layers during test-time
     
    Parameters
    ----------
    net: torch.nn.Module
        The model to enable dropout layers for
    dropout_rate: float
        The dropout rate to set the dropout layers to
    two_dim_dropout_rate: float or str
        The dropout rate to set the two dimensional dropout layers to. If the string 'same' is passed, the two_dim_dropout_rate is set to the dropout_rate

    Returns
    -------
    torch.nn.Module
        The model with the dropout layers enabled
    """
    if isinstance(dropout_rate, float) == False:
        raise ValueError('dropout_rate must be a float')
    if two_dim_dropout_rate == 'same':
        two_dim_dropout_rate = dropout_rate
    elif isinstance(two_dim_dropout_rate, float) == False:
        raise ValueError('two_dim_dropout_rate must be a float or the string "same"')

    for module in net.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate  # Set the new dropout rate
            module.train()  # Enable dropout during test-time by setting to train mode
        elif isinstance(module, nn.Dropout2d):
            module.p = two_dim_dropout_rate
            module.train()

        # Check for nn.Sequential and iterate through its submodules
        elif isinstance(module, nn.Sequential):
            for sub_module in module.children():
                if isinstance(sub_module, nn.Module):
                    enable_dropout(sub_module, dropout_rate)  # Recursively enable dropout

    return net


def evaluate(net, idloader, oodloader, use_cuda=True, samples=100, dropout_rate=0.3,verbose=True,**kwargs):
    """
    Evaluate Monte Carlo dropout on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset
    use_cuda: bool
        Whether to use cuda. Default: True
    samples: int
        The number of samples to use for Monte Carlo dropout. Default: 100
    dropout_rate: float
        The dropout rate to set the dropout layers to. Default: 0.3
    verbose: bool
        Whether to print progress. Default: True

    Returns
    -------
    list
        A confidence list containing two lists. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset
    """
    net.eval()
    net.training = False
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = enable_dropout(net, dropout_rate=dropout_rate,**kwargs) #Enable dropout layers during test-time

    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, targets) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            with torch.no_grad():
                out = [softmax(net(inputs)) for _ in range(samples)]
            out_stack = np.stack(out, axis=2) 
            softmax_score = np.mean(out_stack, axis=2)
            confidence[OOD].extend(np.max(softmax_score,axis=1).tolist())

    return confidence 



def train():
    pass

