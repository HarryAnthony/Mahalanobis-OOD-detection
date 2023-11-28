import torch
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from source.util.general_utils import print_progress
from source.util.mahal_utils import  estimate_mean_precision, calc_mahalanobis_score


def evaluate(net, idloader, oodloader, module, OOD_dict={'name': ['Mahalanobis']},**kwargs):
    """
    Evaluate Mahalanobis distance score on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader 
        The dataloader for the OOD dataset
    module: int, list, tuple, numpy array, or str
        The index of the module after which to extract embeddings and apply Mahalanobis distance score. 
        If it is an array then Mahalanobis distance score is applied on several modules.
        If 'all' is passed, Mahalanobis distance score is applied to all modules of the network.
    OOD_dict: dict
        The dictionary containing details of OOD detection method. Is used to store tha name of the module(s) after which Mahalanobis distance score is applied.

    Returns
    -------
    list
        A confidence list containing pairs of lists for each module. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset.
    """
    confidence = []

    #Get the modules to extract embeddings from
    if isinstance(module, int):
        if module == -1:
            modules = len(get_graph_node_names(net)[0])-1
        modules = [module]
        kwargs['feature_combination'] = False
    elif isinstance(module,str) and module=='all':
        model_modules = get_graph_node_names(net)[0]
        modules = np.arange(0,len(model_modules))
    elif isinstance(module, (list,type(np.array([])),tuple)) == False:
        raise ValueError('module must be an integer, list, tuple, or numpy array, or "all" for all modules')
    else:
        modules = module
    
    confidence = evaluate_mahalanobis_distance(net, idloader, oodloader, modules=modules, OOD_dict=OOD_dict, **kwargs)
    OOD_dict['name'].pop(0) #Remove the original name of the method

    return confidence


def evaluate_MBM(net, idloader, oodloader,OOD_dict={'name': ['Mahalanobis']},net_type='ResNet18',MBM_type='MBM',verbose=True,**kwargs):
    """
    Evaluate Multi-Branch Mahalanobis (MBM) score on the ID and OOD datasets for a model with multiple branches.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset
    OOD_dict: dict
        The dictionary containing details of OOD detection method. Is used to store tha name of the branch(s) after which Mahalanobis distance score is applied.
    net_type: str
        The type of network. Must be one of 'ResNet18', 'ResNet18_with_dropout', or 'VGG16_bn'. Default: 'ResNet18'
    MBM_type: str
        The type of MBM. Must be one of 'MBM' or 'MBM_act_func_only'. Default: 'MBM'
    
    Returns
    -------
    list
        A confidence list containing pairs of lists for each branch. The first list contains the confidence scores for the ID dataset
        and the second list contains the confidence scores for the OOD dataset.
    """
    confidence = []

    if net_type not in mahalanobis_module_dict.keys():
        raise ValueError('net_type must be one of '+str(list(mahalanobis_module_dict.keys())))
    if MBM_type not in mahalanobis_module_dict[net_type].keys():
        raise ValueError('MBM_type must be one of '+str(list(mahalanobis_module_dict[net_type].keys())))

    for branch in mahalanobis_module_dict[net_type][MBM_type].keys():
        if verbose==True:
            print('\nEvaluating '+str(branch))
        modules = mahalanobis_module_dict[net_type][MBM_type][branch]['modules']
        modules_to_skip = mahalanobis_module_dict[net_type][MBM_type][branch]['modules_to_skip']
        confidence.append(evaluate_mahalanobis_distance(net, idloader, oodloader, modules=modules, modules_to_skip=modules_to_skip,OOD_dict=OOD_dict,**kwargs))
    
    OOD_dict['name'] = []
    for branch in mahalanobis_module_dict[net_type][MBM_type].keys():
        OOD_dict['name'].append(str(MBM_type)+' ('+str(branch)+')')

    return confidence


def evaluate_mahalanobis_distance(net,idloader,oodloader,modules,modules_to_skip=[],use_cuda=True, verbose=True, 
             alpha=None,feature_combination=True,trainloader=None,OOD_dict={'name': ['Mahalanobis']},**kwargs):
    """
    Evaluate Mahalanobis distance score on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset
    modules: list, tuple, numpy array
        A list of modules after which to extract embeddings and apply Mahalanobis distance score.
    modules_to_skip: list, tuple, numpy array
        A list of modules to skip which are in the list modules. Default: []
    use_cuda: bool
        Whether to use cuda. Default: True
    verbose: bool
        Whether to print progress. Default: True
    alpha: list
        A list of alpha values to use for combining the Mahalanobis distance scores from different modules. 
        If None, the Mahalanobis distance scores are standardised using the training data. Default: None.
    feature_combination: bool
        Whether to combine the Mahalanobis distance scores from the different modules into one score, or to
        output the score at each module seperately. Default: True
    trainloader: torch.utils.data.DataLoader
        The dataloader for the training dataset. Required if alpha is None and feature_combination is True. Default: None.
    OOD_dict: dict
        The dictionary containing details of OOD detection method. Is used to store tha name of the module(s) after which Mahalanobis distance score is applied.

    Returns
    -------
    list
        A confidence list containing pairs of lists for each module. The first list contains the confidence scores for the ID dataset
        and the second list contains the confidence scores for the OOD dataset.
    """
    net.eval()
    if use_cuda:
        net.cuda()

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    node_names = get_graph_node_names(net)
    try:
        module_names = [node_names[0][k] for k in modules if k not in modules_to_skip]
    except:
        raise ValueError('modules {} is out of range. The model has {} modules.'.format(modules,len(node_names[0])))
    
    feature_extractor = create_feature_extractor(net, return_nodes=module_names)
    #Calculate the statistics of the training data (mean and covariance)
    training_data_statistics = estimate_mean_precision(module_names=module_names,feature_extractor=feature_extractor,trainloader=trainloader,**kwargs)

    #Calculate the Mahalanobis distance metric for the ID and OOD data
    conf_list = [[[] for _ in range(len(module_names))] for _ in range(2)]
    for OOD, loader in enumerate([idloader, oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')
        l = len(loader)
        print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
        for batch_idx, (inputs, target) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
            conf = calc_mahalanobis_score(inputs=inputs,feature_extractor=feature_extractor, modules=module_names,
                                                         training_data_statistics=training_data_statistics,**kwargs)
            for i in range(len(conf)):
                conf_list[OOD][i].extend((-conf[i]).cpu().tolist())

    if feature_combination == True: #If combination of distances from different modules is used
        if alpha is None: #If alpha is not specified, use the training data to determine the standardisation values
            training_conf = [[] for _ in range(len(module_names))]
            OOD_dict['name'].append('Mahalanobis (feature combination of modules '+str(format_modules(modules))+')') 
            print('Standardising Mahalanobis distances with training data')  
            l = len(trainloader)
            kwargs.pop('preprocess')    
            for batch_idx, (inputs, target) in enumerate(trainloader):
                print_progress(batch_idx + 1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
                training_conf_batch = calc_mahalanobis_score(inputs=inputs,feature_extractor=feature_extractor, modules=module_names, 
                                                             training_data_statistics=training_data_statistics,preprocess=False,**kwargs)
                for i in range(len(training_conf_batch)):
                    training_conf[i].extend((-training_conf_batch[i]).cpu().tolist())

            #Standardise the Mahalanobis distances
            training_data_mean = np.array([np.mean(training_conf[i]) for i in range(len(training_conf))])
            training_data_std = np.array([np.std(training_conf[i]) for i in range(len(training_conf))])
            conf = [np.zeros(len(conf_list[0][0])),np.zeros(len(conf_list[1][0]))]
            for k in range(len(module_names)):
                for OOD in range(2):
                    conf_list[OOD][k] = np.array(conf_list[OOD][k])
                    conf_list[OOD][k] = (conf_list[OOD][k] - training_data_mean[k])/training_data_std[k]
                    conf[OOD] += conf_list[OOD][k].tolist()
            return conf
        else: #If alpha is specified, use the alpha values to combine the distances from different modules
            assert isinstance(alpha,list) == True, 'Alpha must be a list'
            assert len(alpha) == len(module_names), 'Alpha must be a list of the same length as the number of modules'
            OOD_dict['name'].append('Mahalanobis (feature combination (modules '+str(format_modules(modules))+') with alpha='+str(alpha)+')')  
            conf = [np.zeros(len(conf_list[0][0])),np.zeros(len(conf_list[1][0]))]
            for k in range(len(module_names)):
                for OOD in range(2):
                    conf_list[OOD][k] = np.array(conf_list[OOD][k])
                    conf[OOD] += conf_list[OOD][k].tolist() * alpha[k]
            return conf
        
    else: #Else output the Mahalanobis distance values for each module separately
        for module in module_names:
            OOD_dict['name'].append('Mahalanobis (module '+str(module)+')')
        return list(map(list, zip(*conf_list)))


def format_modules(modules):
    """
    Format the list of modules to apply mahalanobis distance to a consise format.

    Parameters
    ----------
    modules : list
        List of modules to apply mahalanobis distance to.

    Returns
    -------
    str
        Formatted string of modules.
    """
    ranges = []
    skipped_modules = []
    start = modules[0]
    end = modules[0]

    for module in modules[1:]:
        if module == end + 1:
            end = module
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            for skipped_module in range(end+1, module):
                skipped_modules.append(skipped_module)
            start = end = module

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    formatted_modules = ', '.join(ranges)
    modules_skipped_string = (f"{modules[0]}-{modules[-1]}, skipped {','.join(map(str,skipped_modules))}")
    if len(formatted_modules) < len(modules_skipped_string):
        return formatted_modules
    else:
        return modules_skipped_string


#Dictionary of modules to apply MBM to for each network, please add to this dictionary if you want to apply MBM to a new network
mahalanobis_module_dict = {'ResNet18': {'MBM': {'Branch 1': {'modules': np.arange(4,19), 'modules_to_skip': []},
                                         'Branch 2': {'modules': np.arange(19,35), 'modules_to_skip': []},
                                         'Branch 3': {'modules': np.arange(35,51), 'modules_to_skip': []},
                                         'Branch 4': {'modules': np.arange(51,69), 'modules_to_skip': []}},
                                'MBM_act_func_only': {'Branch 1': {'modules': np.arange(7,19), 'modules_to_skip': [8,9,10,12,13,15,16,17]},
                                                    'Branch 2': {'modules': np.arange(21,19), 'modules_to_skip': [22,23,24,25,26,28,29,31,32,33]},
                                                    'Branch 3': {'modules': np.arange(37,19), 'modules_to_skip': [38,39,40,41,42,44,45,47,48,49]},
                                                    'Branch 3':{'modules': np.arange(53,19), 'modules_to_skip': [54,55,56,57,58,60,61,63,64,65,67,68]}}},

                            'ResNet18_with_dropout': {'MBM': {'Branch 1': {'modules': np.arange(6,24), 'modules_to_skip': [9,14,18,23]},
                                                              'Branch 2': {'modules': np.arange(25,44), 'modules_to_skip': [27,30,31,34,38,43]},
                                                                'Branch 3': {'modules': np.arange(45,64), 'modules_to_skip': [47,50,51,54,58,63]},
                                                                'Branch 4': {'modules': np.arange(65,88), 'modules_to_skip': [67,70,71,74,78,83,86]}},
                                                    'MBM_act_func_only': {'Branch 1': {'modules': [8,13,17,22], 'modules_to_skip': []},
                                                    'Branch 2': {'modules': [26,33,37,42], 'modules_to_skip': []},
                                                    'Branch 3': {'modules': [46,53,57,62], 'modules_to_skip': []},
                                                    'Branch 4':{'modules': [66,73,77,82], 'modules_to_skip': []}}},

                    'vgg16_bn': {'MBM': {'Branch 1': {'modules': np.arange(7,14), 'modules_to_skip': []},
                                            'Branch 2': {'modules': np.arange(14,23), 'modules_to_skip': []},
                                            'Branch 3': {'modules': np.arange(24,34), 'modules_to_skip': []},
                                            'Branch 4': {'modules': np.arange(34,44), 'modules_to_skip': []}},
                                'MBM_act_func_only': {'Branch 1': {'modules': np.arange(10,14), 'modules_to_skip': [11, 12]},
                                                    'Branch 2': {'modules': np.arange(17,24), 'modules_to_skip': [18,19,21,22]},
                                                    'Branch 3': {'modules': np.arange(27,34), 'modules_to_skip': [28,29,31,32]},
                                                    'Branch 4':{'modules': np.arange(37,44), 'modules_to_skip': [38,39,41,42]}}},
                                                    
                    'vgg16_bn_with_dropout': {'MBM': {'Branch 1': {'modules': np.arange(9,18), 'modules_to_skip': []},
                                                      'Branch 2': {'modules': np.arange(18,31), 'modules_to_skip': []},
                                                        'Branch 3': {'modules': np.arange(31,44), 'modules_to_skip': []},
                                                        'Branch 4': {'modules': np.arange(44,57), 'modules_to_skip': []}},
                                        'MBM_act_func_only': {'Branch 1': {'modules': [12,16], 'modules_to_skip': []},
                                                            'Branch 2': {'modules': [21,25,29], 'modules_to_skip': []},
                                                            'Branch 3': {'modules': [34,38,42], 'modules_to_skip': []},
                                                            'Branch 4':{'modules': [47,51,55], 'modules_to_skip': []}}}}

def train():
    pass
