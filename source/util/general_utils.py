"""
Helper functions for general use.
"""
from torch.autograd import Variable
import os

def select_cuda_device(cuda_device):
    """
    Selects the cuda device to use. If cuda_device is 'all' then all available cuda devices are used.

    Parameters
    ----------
    cuda_device : str
        The cuda device to use. Can be 'all', 'none' or a number.

    Returns
    -------
    bool
        True if cuda is used, False otherwise.
    """
    if cuda_device not in ['all','none'] and cuda_device.isdigit() == False:
        raise Exception('Invalid cuda device entered. Use cuda device number, all or none')
    if cuda_device == 'none':
        return False
    if cuda_device != 'all':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"]= cuda_device
    return True


def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", verbose=True):
    """
    Call in a loop to create terminal progress bar. Adapted from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console.

    Parameters
    ----------
    iteration : int
        Current iteration.
    total : int
        Total iterations.
    prefix : str, optional
        Prefix string. The default is ''.
    suffix : str, optional
        Suffix string. The default is ''.
    decimals : int, optional
        Positive number of decimals in percent complete. The default is 1.
    length : int, optional
        Character length of bar. The default is 100.
    fill : str, optional
        Bar fill character. The default is '█'.
    printEnd : str, optional
        End character (e.g. "\r"). The default is "\r".
    verbose : bool, optional
        Whether to print the progress bar. The default is True.

    Returns
    -------
    None if verbose is False, otherwise prints the progress bar.

    """
    if verbose:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)

        # Print New Line on Complete
        if iteration == total:
            space_length = length + len(prefix) + len(suffix) + 11 
            spaces = ' ' * max(space_length, 0)
            print(spaces, end=printEnd, flush=True)
        #if iteration == total: 
        #    print('')
    else:
        return None


def variable_use_cuda(var,use_cuda):
    """
    Moves a variable to cuda if use_cuda is True.

    Parameters
    ----------
    var : torch.autograd.Variable
        The variable to move to cuda.
    use_cuda : bool
        A boolean indicating whether to use cuda.

    Returns
    -------
    torch.autograd.Variable
        The variable on cuda if use_cuda is True, otherwise the variable is returned unchanged.
    """
    if use_cuda:
        var = var.cuda()
    return Variable(var)


class DefaultDict(dict):
    """
    A dictionary that returns a default value if the key is not in the dictionary.
    """
    def __init__(self, default_value, *args, **kwargs):
        self.default_value = default_value
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            return self.default_value