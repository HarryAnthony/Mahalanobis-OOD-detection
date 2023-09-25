import torch
import numpy as np 
from torch.autograd import Variable
import ast
from source.util.general_utils import print_progress
from source.util.evaluate_network_utils import softmax, load_net


def get_ensemble_members(net,seed_list,net_dict,**kwargs):
    """
    Get the ensemble members for deep ensemble OOD detection.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    seed_list: list
        The list of seeds to use for the ensemble members, not including the seed used for net.
    net_dict: dict
        The dictionary containing the network parameters

    Returns
    -------
    list
        A list containing the ensemble members
    """
    nets = [net] #Array of networks to be used in the ensemble

    #Load the ensemble members
    seed_list = ast.literal_eval(seed_list)
    for seed in list(seed_list):
        if isinstance(seed,int) == False and seed.isdigit() == False:
            raise ValueError('Seed must be an integer')
        net_ensemble_member, net_dict_ensemble_member, _ = load_net(int(seed),use_cuda=False,verbose=False)

        assert (net_dict['classes_in'] == net_dict_ensemble_member['classes_in']) or (
            net_dict['classes_out'] == net_dict_ensemble_member['classes_out']) or (
            net_dict['save_dir'] == net_dict_ensemble_member['save_dir']), 'Ensemble members must be trained on same dataset'
        
        net_ensemble_member.eval()
        net_ensemble_member.training = False
        nets.append(net_ensemble_member)


def evaluate(net, idloader, oodloader, use_cuda=True,verbose=True,net_dict=None,seed_list=[],**kwargs):
    """
    Evaluate deep ensemble on the ID and OOD datasets.

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
    verbose: bool
        Whether to print progress. Default: True
    net_dict: dict
        The dictionary containing the network parameters
    seed_list: list
        The list of seeds to use for the ensemble members, not including the seed used for net.

    Returns
    -------
    list
        A confidence list containing two lists. The first list contains the confidence scores for the ID dataset
        and the second list contains the confidence scores for the OOD dataset.
    """
    net.cpu().eval()
    net.training = False
    nets = get_ensemble_members(net,seed_list,net_dict,**kwargs)
        
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                out = [softmax(ensemble_member.cuda()(inputs)) for ensemble_member in nets]
            out_stack = np.stack(out, axis=2)

            softmax_score = np.mean(out_stack, axis=2) 

            confidence[OOD].extend(np.max(softmax_score,axis=1).tolist())
    return confidence


def train():
    pass

