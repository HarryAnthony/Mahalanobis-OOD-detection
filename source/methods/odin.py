import torch
import torch.nn as nn
import numpy as np 
from torch.autograd import Variable
from source.util.general_utils import print_progress
from source.util.evaluate_network_utils import get_softmax_score

def perturb_x(outputs,inputs,softmax_score,criterion,noiseMagnitude,use_cuda=True,**kwargs):
        """
        Perturb the inputs x to improve the confidence in the prediction. 

        parameters
        ----------
        outputs: torch.Tensor
            The output of the network
        inputs: torch.Tensor
            The input to the network
        softmax_score: torch.Tensor
            The output of the network after softmax
        criterion: torch.nn.CrossEntropyLoss
            The loss function
        noiseMagnitude: float
            The magnitude of the noise to be added to the input

        returns
        -------
        torch.Tensor
            The perturbed input
        """
        # calculate which perturbation is necessary
        maxIndexTemp = np.argmax(softmax_score, axis=1)
        labels = Variable(torch.LongTensor(maxIndexTemp))
        if use_cuda:
            labels = labels.cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0) #Returns boolean if grad > 0
        gradient = (gradient.float() - 0.5) * 2  # Turn into -1 and +1
        return torch.add(inputs.data, gradient, alpha=-noiseMagnitude)


def evaluate(net, idloader, oodloader, use_cuda=True, noiseMagnitude = 0, temper = 1,criterion=nn.CrossEntropyLoss(),verbose=True,**kwargs):
    """
    Evaluate ODIN on the ID and OOD datasets.

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
    noiseMagnitude: float
        The magnitude of the noise to be added to the input. Default: 0
    temper: float
        The temperature to use for scaling the softmax output. Default: 1
    criterion: torch.nn.CrossEntropyLoss
        The loss function to backpropogate with. Default: nn.CrossEntropyLoss().
    verbose: bool
        Whether to print progress. Default: True

    Returns
    -------
    list
        A confidence list containing two lists. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset.
    """
    net.eval()
    net.training = False
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
            #Print the progress bar
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            outputs, inputs, softmax_score = get_softmax_score(inputs,net,use_cuda=use_cuda,required_grad=True,temper=temper)

            #Preprocess inputs to seperate OOD and ID
            inputs_perturbed = perturb_x(outputs/temper,inputs,softmax_score,criterion,noiseMagnitude,use_cuda=use_cuda,**kwargs)

            with torch.no_grad():
                #Classify perturbed inputs
                softmax_score_perturbed = get_softmax_score(inputs_perturbed,net,use_cuda=use_cuda,required_grad=False,softmax_only=True,temper=temper)

            confidence[OOD].extend(np.max(softmax_score_perturbed,axis=1).tolist())

    return confidence


def train():
    pass

