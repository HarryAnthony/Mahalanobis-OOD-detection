"""
Helper functions for evaluating the network and its OOD detection performance.
"""
import pandas as pd
import torch
import torch.nn as nn
import numpy as np 
from torch.autograd import Variable
import sklearn.metrics as skm
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import ast
from itertools import combinations
from torch.nn import functional as F
from torchvision.models import resnet18, vgg11, vgg16_bn, resnet34, resnet50
from source.models.wide_resnet import Wide_ResNet
from source.util.training_utils import set_activation_function, add_dropout_network_architechture
from source.util.general_utils import print_progress


def check_net_exists(seed, verbose=True, get_output=False):
    """
    Check if the net exists in the database of nets.

    Parameters
    ----------
    seed : int
        The seed of the net to check.
    verbose : bool, optional
        Whether to print the net information. The default is True.
    get_output : bool, optional
        Whether to return the net information. The default is False.

    Raises
    ------
    Exception
        If the net does not exist in the database.

    Returns
    -------
    None if get_output is False, otherwise returns a dictionary containing the net information.
    """
    model_list = pd.read_csv('outputs/saved_models/model_list.csv')  # Get list of nets

    matching_models = model_list[model_list['Seed'] == int(seed)]
    if len(matching_models) == 0:
        raise Exception('Experiment seed is not in the list of known experiments')
    model = matching_models.iloc[0]

    if verbose:
        print('Model database: {}\nModel setting: {}\nModel type: {}\nModel widen factor x depth: {} x {}\nDropout: {}\n'.format(
            model['Database'], model['Setting'], model['Model_type'], model['Widen_factor'],
            model['Depth'], model['Dropout']))

    if get_output:
        net_info = {
            'Model pathname': model['Model_name'],
            'Model database': model['Database'],
            'Model setting': 'setting'+str(int(model['Setting'])),
            'Model architecture': model['Model_type'],
            'Model widen factor': model['Widen_factor'],
            'Model depth': model['Depth'],
            'Model activation': model['Activation_function'],
            'Dropout': model['Dropout'],
            'Act_func_Dropout': model['act_func_dropout'],
            'DUQ': model['DUQ'],
            'Requires split': model['requires_split'],
            'Dataset seed': model['Dataset_seed'],
            'class_selections': model['class_selections'],
            'demographic_selections': model['demographic_selections'],
            'dataset_selections': model['dataset_selections'],
            'train_val_test_split_criteria': model['train_val_test_split_criteria'],
            'num_classes': model['num_classes']
        }

        return net_info
    return None


def load_net(seed,verbose=True,use_cuda=True):
    """
    Load the net from the database of nets.

    Parameters
    ----------
    seed : int
        The seed of the net to load.
    verbose : bool, optional
        Whether to print the net information. The default is True.
    use_cuda : bool, optional
        Whether to use cuda. The default is True.
    ensemble : bool, optional
        Whether the net is being used for an ensemble. The default is False.

    Raises
    ------
    Exception
        If the net does not exist in the database.

    Returns
    -------
    net : torch.nn.Module
        The loaded net.
    net_dict : dict
        A dictionary containing the net information.
    cf : config file
        The configuration file for the dataset.

    """
    net_info = check_net_exists(seed,verbose=verbose,get_output=True)

    net_dict = {'Requires split': net_info['Requires split'],
                'setting': net_info['Model setting']}

    #Get configuration for given datasets
    if net_info['Model database'] == 'CheXpert' or net_info['Model database'] == 'cheXpert':
        from source.config import chexpert as cf_chexpert
        cf = cf_chexpert
    elif net_info['Model database'] == 'cifar10':
        from source.config import cifar10 as cf_cifar10
        cf = cf_cifar10

    #Get the classes in and out
    if net_info['Requires split']:
        net_dict['class_selections'] = turn_str_into_dict(net_info['class_selections'])
        net_dict['demographic_selections'] = turn_str_into_dict(net_info['demographic_selections'])
        net_dict['dataset_selections'] = turn_str_into_dict(net_info['dataset_selections'])
        net_dict['train_val_test_split_criteria'] = turn_str_into_dict(net_info['train_val_test_split_criteria'])
        net_dict['classes_ID'] = net_dict['class_selections']['classes_ID']
        net_dict['classes_OOD'] = net_dict['class_selections']['classes_OOD']
    else:
        net_dict['classes_ID'] = cf.classes
        net_dict['classes_OOD'] = []

    net_dict['num_classes'] = net_info['num_classes']

    if verbose:
        print('| Preparing '+net_info['Model database']+' test with the following classes: ')
        print(f"| Classes ID: {net_dict['classes_ID']}")
        print(f"| Classes OOD: {net_dict['classes_OOD']}\n")

    #Select the network architecture
    model_architecture_dict = {
    'wide-resnet': (Wide_ResNet, ['Model depth', 'Model widen factor', 'Dropout']),
    'ResNet18': (resnet18, []),
    'ResNet34': (resnet34, []),
    'ResNet50': (resnet50, []),
    'vgg11': (vgg11, []),
    'vgg16_bn': (vgg16_bn, [])
    }
    model_architecture = net_info['Model architecture']
    model_func, model_args = model_architecture_dict.get(model_architecture, (None, None))
    if model_func is None:
        raise ValueError('Invalid model architecture')
    
    #Load the network architecture
    kwargs = {'num_classes': net_dict['num_classes']}
    for arg in model_args:
        kwargs[arg] = int(net_info[arg])
    net = model_func(**kwargs)
    net = add_dropout_network_architechture(net,net_info)
    net_dict['file_name'] = f"{str(net_info['Model architecture'])}-{int(net_info['Model depth'])}x{net_info['Model widen factor']}_{str(net_info['Model database'])}-{int(seed)}"
    net_dict['save_dir'] = os.path.join("outputs", f"{net_info['Model database'].lower()}_{net_info['Model setting']}")
    net_dict['pathname'] = net_info['Model pathname']

    # Model setup
    assert os.path.isdir('outputs/saved_models'), 'Error: No saved_models directory found!'
    if use_cuda:
        checkpoint = torch.load('outputs/saved_models/'+net_info['Model database'].lower()+'/'+net_info['Model pathname']+'.pth')
    else:
        checkpoint = torch.load('outputs/saved_models/'+net_info['Model database'].lower()+'/'+net_info['Model pathname']+'.pth',map_location='cpu')
    #Apply parameters and activation function to the network
    params = {}
    for k_old in checkpoint.keys():
        k_new = k_old.replace('module.', '')
        params[k_new] = checkpoint[k_old]
    net.load_state_dict(params)
    net = set_activation_function(net,net_info['Model activation'])

    net_dict['act_func_dropout_rate'] = net_info['Act_func_Dropout']
    net_dict['net_type'] = net_info['Model architecture']

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True

    return net, net_dict, cf


def turn_str_into_dict(string):
    """
    Turn a string into a dictionary.

    Parameters
    ----------
    string : str
        The string to convert.

    Returns
    -------
    convert_dict : dict
        The converted dictionary.
    """
    string = string.replace('np.nan', '\"null\"') #Replace np.nan with null to prevent errors
    convert_dict = ast.literal_eval(string)

    if 'replace_values_dict' in convert_dict.keys():
        if 'null' in convert_dict['replace_values_dict'].keys():
            convert_dict['replace_values_dict'][np.nan] = convert_dict['replace_values_dict']['null']
            convert_dict['replace_values_dict'].pop('null')
 
    return convert_dict
                

def evaluate_ood_detection_method(method,net,idloader,oodloader,return_metrics=False,**kwargs):
    """
    Evaluate the OOD detection performance of a net for a given method.

    Parameters
    ----------
    method : str
        The name of the OOD detection method.
    net : torch.nn.Module
        The net to evaluate.
    idloader : torch.utils.data.DataLoader
        The dataloader for the in-distribution dataset.
    oodloader : torch.utils.data.DataLoader
        The dataloader for the OOD dataset.
    return_metrics : bool, optional
        Whether to return the AUROC and AUCPR. The default is False.

    Raises
    ------
    ValueError
        If the method is invalid.

    Returns
    -------
    AUROC : float
        The AUROC (if return_metrics is True).
    AUCPR : float
        The AUCPR (if return metrics is True).
    """
    from source.methods import mcp, odin, mcdp, deepensemble, mahalanobis

    OOD_detection_dict = {'MCP': {'function': mcp.evaluate, 'name': ['MCP']},
                           'ODIN': {'function': odin.evaluate, 'name': ['ODIN']},
                           'MCDP': {'function': mcdp.evaluate, 'name': ['MCDP']},
                           'deepensemble': {'function': deepensemble.evaluate, 'name': ['Deep_ensemble']},
                            'mahalanobis' : {'function': mahalanobis.evaluate, 'name': ['Mahalanobis']},
                            'MBM':  {'function': mahalanobis.evaluate_MBM, 'name': ['MBM']}}
    
    if method not in OOD_detection_dict.keys():
        raise ValueError(f'Invalid OOD detection method: {method}')
    kwargs['OOD_dict'] = OOD_detection_dict[method]
    
    if return_metrics == True:
        AUROC, AUCPR = ood_evaluation(OOD_detection_dict[method], net, idloader, oodloader, return_metrics=True,**kwargs)
        return AUROC, AUCPR
    ood_evaluation(OOD_detection_dict[method], net, idloader, oodloader, **kwargs)
    

def ood_evaluation(ood_detection_method, net, idloader, oodloader, verbose=True, save_results=False, save_dir=None, return_metrics=False, plot_metric=False, filename='', **kwargs):
    """
    Evaluate the OOD detection performance of a net for a given method.

    Parameters
    ----------
    ood_detection_method : dict
        The dictionary of the OOD detection method and the methods name.
    net : torch.nn.Module
        The net to evaluate.
    idloader : torch.utils.data.DataLoader
        The dataloader for the in-distribution dataset.
    oodloader : torch.utils.data.DataLoader
        The dataloader for the OOD dataset.
    verbose : bool, optional
        Whether to print the AUROC and AUCPR. The default is True.
    save_results : bool, optional
        Whether to save the results in textfiles. The default is False.
    save_dir : str, optional
        The directory to save the results, requires save_results to be True. The default is None.
    return_metrics : bool, optional
        Whether to return the AUROC and AUCPR. The default is False.
    plot_metric : bool, optional
        Whether to plot a visualisation of the OOD detection metric, requires save_results to be True. The default is False.
    filename : str, optional
        The filename to save the results as, requires save_results to be True. The default is ''.

    Returns
    -------
    AUROC : float
        The AUROC (if return_metrics is True).
    AUCPR : float
        The AUCPR (if return metrics is True).
    """
    confidence_id_ood = ood_detection_method['function'](net, idloader, oodloader, **kwargs)

    ood_detection_method['name'] = kwargs['OOD_dict']['name'] if ood_detection_method['name'] != kwargs['OOD_dict']['name'] else ood_detection_method['name']
    confidence_id_ood = [confidence_id_ood] if isinstance(confidence_id_ood[0][0],(float,int)) == True else confidence_id_ood
    OOD_detection_method_scores = []

    for idx, (id, ood) in enumerate(confidence_id_ood):
        AUROC, AUCPR = get_AUROC_AUCPR(id, ood)
        OOD_detection_method_scores.append([ood_detection_method['name'][idx],AUROC, AUCPR])
        if verbose:
            print(f"| AUROC for {ood_detection_method['name'][idx]}: {AUROC}\n| AUCPR for {ood_detection_method['name'][idx]}: {AUCPR}")
        if save_results:
            if save_dir == None:
                raise Exception('Must specify save_dir')
            else:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                f1_path = os.path.join(save_dir, "confidence_ID_%s%s.txt" % (ood_detection_method['name'][idx],filename))
                f2_path = os.path.join(save_dir, "confidence_OOD_%s%s.txt" % (ood_detection_method['name'][idx],filename))
                np.savetxt(f1_path, id)
                np.savetxt(f2_path, ood)

                if plot_metric == True:
                    plot_metrics(id,ood,save_dir=save_dir+'/OOD_scoring_function_'+str(ood_detection_method['name'][idx])+str(filename)+'.pdf',title=str(ood_detection_method['name'][idx])+str(filename),show_plot=False)
    
    if save_results:    
        metrics_filename = "Metrics%s.txt" % (filename) if len(ood_detection_method['name'])!=1 else "Metrics_%s%s.txt" % (ood_detection_method['name'][0],filename)
        f4_path = os.path.join(save_dir, metrics_filename)
        with open(f4_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['OOD detection method', 'AUROC', 'AUCPR'])
            for (OOD_method_name, AUROC, AUCPR) in OOD_detection_method_scores:
                writer.writerow([OOD_method_name, AUROC, AUCPR])

    if return_metrics:
        return OOD_detection_method_scores[0][1], OOD_detection_method_scores[0][2]


def get_softmax_score(inputs,net,use_cuda=True,required_grad=False,softmax_only=False,temper=1,**kwargs):
    """
    Classify inputs using a given neural network and output the softmax scores.

    Parameters
    ----------
    inputs : torch.Tensor
        The inputs to classify.
    use_cuda : bool
        Whether to use cuda.
    net : torch.nn.Module
        The neural network to use.
    required_grad : bool, optional
        Whether to require gradients for the inputs. The default is False.
    softmax_only : bool, optional
        Whether to only output the softmax scores. The default is False.
    temper : float, optional
        The temperature for the softmax. The default is 1.

    Returns
    -------
    outputs : torch.Tensor
        The outputs of the neural network.
    inputs : torch.Tensor
        The inputs to the neural network.
    softmax_score : torch.Tensor
        The softmax outputs of the neural network.
    """
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs, requires_grad=required_grad)
    outputs = net(inputs)
    softmax_score = softmax(outputs,temper=temper)  #Convert outputs into softmax
    if softmax_only == True:
        return softmax_score
    return outputs, inputs, softmax_score


def get_softmax_score_report_accuracy(inputs,targets,use_cuda,net,correct,total,logits_list,labels_list,correct_list,predicted_list,required_correct_list=False,**kwargs):
    """
    Classify inputs with a given neural network, output the softmax scores and report the accuracy of the classifier.

    Parameters
    ----------
    inputs : torch.Tensor
        The inputs to classify.
    targets : torch.Tensor
        The targets of the inputs.
    use_cuda : bool
        Whether to use cuda.
    net : torch.nn.Module
        The neural network to use.
    correct : int
        The number of correct predictions.
    total : int
        The total number of predictions.
    logits_list : list
        The list of logits.
    labels_list : list
        The list of labels.
    correct_list : list
        The list of correct predictions.
    predicted_list : list
        The list of predicted labels.
    required_grad : bool, optional
        Whether requireS gradients for the inputs. The default is False.
    required_correct_list : bool, optional
        Whether requireS the correct list. The default is False.
    temper : float, optional
        The temperature for the softmax. The default is 1.

    Returns
    -------
    outputs : torch.Tensor
        The outputs of the neural network.
    inputs : torch.Tensor
        The inputs to the neural network.
    nnOutputs : torch.Tensor
        The softmax outputs of the neural network.
    hidden : torch.Tensor
        The hidden layer outputs of the neural network.
    total : int
        The total number of predictions.
    """
    outputs, inputs, softmax_score = get_softmax_score(inputs,net,use_cuda=use_cuda,**kwargs)

    if use_cuda:
        targets = targets.cuda()
    targets = Variable(targets)

    with torch.no_grad():
            logits_list.append(outputs.data)
            labels_list.append(targets.data)

    if required_correct_list:
        #Compare classifier outputs to targets to get accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_list.extend(predicted.eq(targets.data).cpu().tolist())
        predicted_list.extend(predicted.cpu().tolist())
        return outputs, inputs, softmax_score, total, correct, logits_list, labels_list,correct_list,predicted_list

    return outputs, inputs, softmax_score, logits_list, labels_list


def calculate_accuracy(logits_list,labels_list,correct,total,correct_list,confidence_list,ece_criterion,verbose=True):
    """
    Calculate the accuracy and AUC of an ood_method.

    Parameters
    ----------
    logits_list : list
        The list of logits.
    labels_list : list
        The list of labels.
    correct : int
        The number of correct predictions.
    total : int
        The total number of predictions.
    correct_list : list
        The list of correct predictions.
    confidence_list : list
        The list of confidence scores.
    ece_criterion : torch.nn.Module
        The ECE criterion.
    verbose : bool, optional
        Whether to print the accuracy. The default is False.

    Returns
    -------
    acc : float
        The accuracy of the classifier.
    acc_list : float
        The accuracy of the classifier.
    auroc_classification : float
        The AUROC of the classifier.
    """
    #Calculate the accuracy
    with torch.no_grad():
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        ece = ece_criterion(logits, labels)
    acc = 100.*correct/total
    acc_list = (sum(correct_list)/len(correct_list))

    pred_probs_total = combine_arrays([confidence_list,correct_list])
    pred_probs_total_sort = np.array(pred_probs_total)[np.array(pred_probs_total)[:, 0].argsort()]
    confidence_list = np.array([pred_probs_total_sort[i][0] for i in  range(len(pred_probs_total))])
    correct_list = np.array([pred_probs_total_sort[i][1] for i in range(len(pred_probs_total))])

    # calculate AUROC for classifcation accuracy
    fpr, tpr, _ = skm.roc_curve(y_true = correct_list, y_score = confidence_list, pos_label = 1) #positive class is 1; negative class is 0
    auroc_classification = skm.auc(fpr, tpr)
    if verbose:
        print("| Test Result\tAcc@1: %.2f%%" %(acc))
        print(f'| ECE: {ece.item()}')
        print(f'| Acc list: {acc_list}')
        print(f'| AUROC classification: {auroc_classification}')

    return acc, auroc_classification, ece.item()


def softmax(outputs, temper=1):
    """
    Calculate the softmax using the outputs of a neural network.

    Parameters
    ----------
    outputs : torch.Tensor
        The outputs of a neural network.
    temper : float, optional
        The temperature for the softmax. The default is 1.

    Returns
    -------
    nnOutputs : torch.Tensor
        The softmax outputs of the neural network.
    """
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs/temper) / np.sum(np.exp(nnOutputs/temper), axis=1, keepdims=True)
    return nnOutputs


def evaluate_accuracy(net,loader,verbose=True,use_cuda=True,save_results=False,save_dir='',plot_metric=False,filename='ID'):
    ece_criterion = ECELoss()
    if use_cuda:
        ece_criterion.cuda()
    net.eval()
    net.training = False
    correct, total = 0, 0
    total = 0
    logits_list = []
    labels_list = []
    confidence_list = np.array([])
    correct_list = []
    predicted_list = []

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    l = len(loader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
    for batch_idx, (inputs, targets) in enumerate(loader):
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        with torch.no_grad():
            _, inputs, softmax_score, total, correct, logits_list, labels_list,correct_list,predicted_list = get_softmax_score_report_accuracy(inputs,
                            targets,use_cuda,net,correct,total,logits_list,labels_list,correct_list,predicted_list,required_grad=False,required_correct_list=True)
        confidence_list = np.concatenate([confidence_list,np.max(softmax_score,axis=1)])

    acc, auroc, ece = calculate_accuracy(logits_list,labels_list,correct,total,correct_list,confidence_list,ece_criterion)

    if save_results:
        f1_path = os.path.join(save_dir,'ID_task_accuracy'+str(filename)+'.txt')
        with open(f1_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID task accuracy', 'AUROC', 'ECE'])
            for (acc_val, auroc_val, ece_val) in [(acc, auroc, ece)]:
                writer.writerow([acc_val, auroc_val, ece_val])

        true_confidences = [confidence for i, confidence in enumerate(confidence_list) if correct_list[i]]
        false_confidences = [confidence for i, confidence in enumerate(confidence_list) if not correct_list[i]]
        correct_bool_list = [True if i in correct_list else False for i in range(len(confidence_list))]

        f2_path = os.path.join(save_dir,'ID_task_confidence'+str(filename)+'.txt')
        with open(f2_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Confidence', 'Correct'])
            for (confidence_val, correct_val) in [(confidence_list, correct_bool_list)]:
                writer.writerow([confidence_val, correct_val])

        if plot_metric:
            plot_softmax_confidence(true_confidences,false_confidences,save_dir=save_dir+'/ID_task_accuracy_'+str(filename)+'.pdf',title='ID task accuracy for '+str(filename)+' data',show_plot=False)


def get_metrics(path_id_confidence, path_ood_confidence, verbose=True, normalized=True):
    """ 
    Returns most common metrics (AUC, FPR, TPR) for comparing OOD vs ID inputs.
    Assumes that values are probabilities/confidences between 0 and 1 as default. 
    If not, please set normalized to False.

    Parameters
    ----------
    path_id_confidence : str
        The path to the text file containing the confidence scores of the in-distribution data.
    path_ood_confidence : str
        The path to the text file containing the confidence scores of the out-of-distribution data.
    verbose : bool, optional
        Whether to print the metrics. The default is True.
    normalized : bool, optional
        Whether the confidence scores are normalized. The default is True.

    Returns
    -------
    auroc : float
        The AUROC of the classifier.
    aucpr : float
        The AUCPR of the classifier.
    fpr : float
        The FPR of the classifier.
    tpr : float
        The TPR of the classifier.
    """
    id = np.loadtxt(path_id_confidence)
    ood = np.loadtxt(path_ood_confidence)
    if verbose:
        print('Mean confidence OOD: {}, Median: {}, Length: {}'.format(np.mean(ood), np.median(ood), len(ood)))
        print('Mean confidence ID: {}, Median: {}, Length: {}'.format(np.mean(id), np.median(id), len(id)))
    id_l = np.ones(len(id))
    ood_l = np.zeros(len(ood))
    true_labels = np.concatenate((id_l, ood_l))
    pred_probs = np.concatenate((id, ood))
    assert(len(true_labels) == len(pred_probs))
    if not normalized:
        # use unity based normalization to also catch negative values
        pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs))
    pred_probs_total = combine_arrays([pred_probs,true_labels])
    pred_probs_total_sort = np.array(pred_probs_total)[np.array(pred_probs_total)[:, 0].argsort()]
    pred_probs = np.array([pred_probs_total_sort[i][0] for i in  range(len(pred_probs_total))])
    true_labels = np.array([pred_probs_total_sort[i][1] for i in range(len(pred_probs_total))])
    fpr, tpr, _ = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1) #positive class is 1; negative class is 0
    auroc = skm.auc(fpr, tpr)
    precision, recall, _ = skm.precision_recall_curve(true_labels, pred_probs)
    aucpr = skm.auc(recall, precision)
    if verbose:
        print('AUROC: {}'.format(auroc))
    return auroc, aucpr, fpr, tpr


def combine_arrays(input_arrays):
    """
    Combines elements of arrays into a single array

    Parameters
    ----------
    input_arrays : list
        A list of arrays to be combined.

    Returns
    -------
    output_array : list
        A list of arrays containing the combined elements of the input arrays.
    """
    if any(len(input_arrays[0])!= len(i) for i in input_arrays):
        raise Exception("Lists must all be the same length")

    output_array = []
    for i in range(0,len(input_arrays[0])):
        output_array.append([])
        for j in range(0,len(input_arrays)):
            output_array[i].append(input_arrays[j][i])
    
    return output_array


def plot_softmax_confidence(correct_predictions, incorrect_predictions, normalized=True, title='ODD metric', save_dir=0, show_plot=False):
    """
    Plots the histograms of the ood metric for comparing OOD and ID inputs.

    Parameters
    ----------
    path_id_confidence : str
        The path to the text file containing the confidence scores of the in-distribution data.
    path_ood_confidence : str
        The path to the text file containing the confidence scores of the out-of-distribution data.
    normalized : bool, optional
        Whether the confidence scores are normalized. The default is True.
    title : str, optional
        The title of the plot. The default is 'ODD metric'.
    save_dir : str, optional
        The directory to save the plot. The default is 0.
    """
    max_val = np.max([np.max(correct_predictions),np.max(incorrect_predictions)])
    min_val = np.min([np.min(correct_predictions),np.min(incorrect_predictions)])
    plt.figure()
    set_style(fontsize=12)
    for _ , (out_scores,color,name) in enumerate([[correct_predictions,'mediumseagreen','Correct prediction'],[incorrect_predictions,'darkmagenta','Incorrect prediction']]):
        vals,bins = np.histogram(out_scores,bins = 51,density=True)
        bin_centers = (bins[1:]+bins[:-1])/2.0
        plt.plot(bin_centers,vals,linewidth=2,color=color,marker="",label=name)
        plt.fill_between(bin_centers,vals,[0]*len(vals),color=color,alpha=0.3)
    plt.xlim(min_val-(bins[1]-bins[0]),max_val+(bins[1]-bins[0]))
    plt.ylim(0)
    plt.legend()
    plt.title(str(title))
    plt.xlabel('Normalised maximum class softmax probability')
    plt.ylabel('Frequency')
    plt.grid()

    if save_dir != 0:
        plt.savefig(save_dir)

    if show_plot == True:
        plt.show()
    plt.close()


def plot_metrics(id, ood, normalized=True, title='ODD metric', save_dir=0, show_plot=False):
    """
    Plots the histograms of the ood metric for comparing OOD and ID inputs.

    Parameters
    ----------
    path_id_confidence : str
        The path to the text file containing the confidence scores of the in-distribution data.
    path_ood_confidence : str
        The path to the text file containing the confidence scores of the out-of-distribution data.
    normalized : bool, optional
        Whether the confidence scores are normalized. The default is True.
    title : str, optional
        The title of the plot. The default is 'ODD metric'.
    save_dir : str, optional
        The directory to save the plot. The default is 0.
    """
    max_val = np.max([np.max(id),np.max(ood)])
    min_val = np.min([np.min(id),np.min(ood)])

    id_norm = (id - min_val)/(max_val - min_val)
    ood_norm = (ood - min_val)/(max_val - min_val)

    id_l = np.ones(len(id))
    ood_l = np.zeros(len(ood))
    true_labels = np.concatenate((id_l, ood_l))
    pred_probs = np.concatenate((id, ood))
    assert(len(true_labels) == len(pred_probs))
    if not normalized:
        # use unity based normalization to also catch negative values
        pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs))
    _, _, thresholds = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1) #positive class is 1; negative class is 0

    plt.figure()
    set_style(fontsize=12)
    for _ , (out_scores,color,name) in enumerate([[id_norm,'RoyalBlue','In distribution'],[ood_norm,'orange','Out of distribution']]):
        vals,bins = np.histogram(out_scores,bins = 51,density=True)
        bin_centers = (bins[1:]+bins[:-1])/2.0
        plt.plot(bin_centers,vals,linewidth=2,color=color,marker="",label=name)
        plt.fill_between(bin_centers,vals,[0]*len(vals),color=color,alpha=0.3)
    plt.xlim(0,1)
    plt.ylim(0)
    plt.legend()
    plt.title(str(title))
    plt.xlabel('Metric')
    plt.ylabel('Frequency')

    plt.axvline(x=thresholds[-1],color='black',linestyle='--')
    plt.grid()

    if save_dir != 0:
        plt.savefig(save_dir)

    if show_plot == True:
        plt.show()
    plt.close()


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    

def get_AUROC_AUCPR(id,ood,return_fpr_tpr=False):
    """
    Calculates the AUROC, AUCPR, FPR and TPR of an ID and OOD dataset.

    Parameters
    ----------
    id : numpy array
        The confidence scores of the in-distribution data.
    ood : numpy array
        The confidence scores of the out-of-distribution data.

    Returns
    -------
    auroc : float
        The AUROC score.
    aucpr : float
        The AUCPR score.
    fpr : float
        The false positive rate.
    tpr : float
        The true positive rate.
    """
    id_l = np.ones(len(id))
    ood_l = np.zeros(len(ood))
    true_labels = np.concatenate((id_l, ood_l))
    pred_probs = np.concatenate((id, ood))
    assert(len(true_labels) == len(pred_probs))
        # use unity based normalization to also catch negative values
    pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs))
    fpr, tpr, _ = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1) #positive class is 1; negative class is 0
    auroc = skm.auc(fpr, tpr)
    precision, recall, _ = skm.precision_recall_curve(true_labels, pred_probs)
    aucpr = skm.auc(recall, precision)
    
    if return_fpr_tpr:
        return auroc, aucpr, fpr, tpr
    return auroc, aucpr


def set_style(fontsize=20):
    """
    Sets the style of the plots.

    Parameters
    ----------
    fontsize : int, optional
        The fontsize of the plots. The default is 20.
    """
    mpl.rcParams['font.family'] = 'sans-serif'
    #mpl.rcParams['font.sans-serif'] = 'Lato'
    plt.rcParams.update({'font.size': fontsize})


def ensure_class_overlap(OOD_dataset,classes_ID,classes_OOD):
    """
    Ensures that there is class overlap between the ID and OOD datasets.

    Parameters
    ----------
    OOD_dataset : dict
        The OOD dataset.
    classes_ID : list
        The list of classes in the ID dataset.
    classes_OOD : list
        The list of classes in the OOD dataset.

    Returns
    -------
    OOD_dataset : dict
        The OOD dataset with class overlap.
    """
    assert any(item in classes_OOD for item in classes_ID), 'There must be class overlap between ID and OOD'
    allowed_classes = [index for index, item_OOD in enumerate(classes_OOD) if item_OOD in classes_ID]
    OOD_dataset['test_df'] = OOD_dataset['test_df'][OOD_dataset['test_df']['class'].isin(allowed_classes)]
    class_mapping = {index_OOD: classes_ID.index(item_OOD) for index_OOD, item_OOD in enumerate(classes_OOD) if item_OOD in classes_ID}
    OOD_dataset['test_df']['class'] = OOD_dataset['test_df']['class'].map(class_mapping)
    return OOD_dataset


def expand_classes(classes_ID,class_sel_dict):
    """
    Expands the classes to include all possible combinations of classes.

    Parameters
    ----------
    classes_ID : list
        The list of classes in the ID dataset.
    class_sel_dict : dict
        The dictionary of the class selection.

    Returns
    -------
    classes_ID : list
        The list of classes in the ID dataset.
    """
    if classes_ID == []:
        classes_ID = class_sel_dict['classes_ID']

    if ('atleast_one_positive_class' in class_sel_dict.keys() and class_sel_dict['atleast_one_positive_class'] == False
        ) or ('allow_multiple_positive_classes' in class_sel_dict.keys() and class_sel_dict['allow_multiple_positive_classes'] == True):
        result = ['Neg ' + ', '.join(classes_ID)] # Initialize the result list with 'no classes'
        # Generate all possible combinations of classes
        for r in range(1, len(classes_ID) + 1):
            for combo in combinations(classes_ID, r):
                # Create a string representation of the combination
                combo_str = ' '.join(['Pos ' + c if c in combo else 'Neg ' + c for c in classes_ID])
                result.append(combo_str)
        classes_ID = result

    return classes_ID