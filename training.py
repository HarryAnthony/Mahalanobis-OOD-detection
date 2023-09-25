import argparse
import os
from types import SimpleNamespace
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from source.util.general_utils import select_cuda_device
from source.util.processing_data_utils import get_dataset_config, get_weighted_dataloader, get_dataloader, get_dataset_selections
from source.util.training_utils import select_experiment_seed, get_class_weights, record_model, get_network_architecture, initialise_network, get_criterion, get_optimiser_scheduler, set_activation_function
from source.util.evaluate_network_utils import load_net
from source.util.Select_dataset import Dataset_selection_methods
from source.util.Train_DNN import Train_DNN


parser = argparse.ArgumentParser(description='Training a DNN')
parser.add_argument('--setting', default='setting1', type=str,
                    help='dataset setting for CheXpert, either setting1, setting2 or setting3')
parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
parser.add_argument('--net_type', default='ResNet18',
                    type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10,
                    type=int, help='width of model')
parser.add_argument('--dropout', default=0.4, type=float, help='dropout_rate')
parser.add_argument('--act_func_dropout', default=0.2, type=float,
                    help='2D dropout to be applied in a CNN.')
parser.add_argument('--cuda_device', default='all', type=str,
                    help='Select device to run code, could be device number or all')
parser.add_argument('--seed', default='random', type=str,
                    help='Select experiment seed')
parser.add_argument('--dataset_seed', default='same', type=str,
                    help='Select seed for seperating the dataset into train, validation and test sets. Default is the same as the experiment')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size (typically order of 2), default: 8')
parser.add_argument('--dataset', default='chexpert', type=str,
                    help='dataset = [cifar10/cifar100/chexpert]')
parser.add_argument('--allow_repeats',default=False,type=bool, help='Allow experiments of the same seed to be repeated. Default: False')
parser.add_argument('--verbose',default=True,type=bool, help='verbose')
parser.add_argument('--optimiser', '-O', default='SGD',
                    help='optimiser method for the model (default: SGD).')
parser.add_argument('--scheduler', '-Sc', default='MultiStepLR',
                    help='scheduler for the model (default: MultiStepLR).')
parser.add_argument('--save_model', default='last_epoch', type=str,
                    help='When to save model parameters.')
parser.add_argument('--max_lr', default=1e-2,
                    help='Maxmimum lr which can be reached when using OneCycleLR.')
parser.add_argument('--act_func', '-Af', default='ReLU',
                    help='The activation function used for the DNN. (default: ReLU)')
parser.add_argument('--class_selections', '-c_sel', default={'classes_ID': ['Cardiomegaly'], 'classes_OOD': [], 'atleast_one_positive_class': False,'replace_values_dict':{}}, type=dict,
                    help='The class selections to be used if the args.setting is not known.')
parser.add_argument('--demographic_selections', '-d_sel', default={}, type=dict,
                    help='The demographic selections to be used if the args.setting is not known.')
parser.add_argument('--dataset_selections', '-dataset_s', default={}, type=dict,
                    help='The dataset specific selections to be used if the args.setting is not known (default is for CheXpert).')
parser.add_argument('--train_val_test_split_criteria', '-split_sel', default={'valSize': 0.2, 'testSize': 0}, type=dict,
                    help='The dataset splitting criteria to be used if the args.setting is not known.')
parser.add_argument('--fold', default=0, type=int,
                    help='The fold to train with when using k-fold cross validation.')
parser.add_argument('--label_smoothing', default=0.0, type=float,
                    help='Float for label smoothing.')
parser.add_argument('--save_path', default='outputs/saved_models', type=str,
                    help='Path to save the model.')
parser.add_argument('--resume_training', default=False, type=bool,
                    help='Resume training from a saved model.')
parser.add_argument('--valSize', default=0.2, type=float,
                    help='Validation set size.')
parser.add_argument('--testSize', default=0, type=float,
                    help='Test set size.')
parser.add_argument('--gradient_clipping', default=True, type=bool,
                    help='Whether to use gradient clipping. Default: True')
args = parser.parse_args()


#Select a GPU to use
use_cuda = select_cuda_device(args.cuda_device)
pin_memory = use_cuda
device_count = torch.cuda.device_count()


#Get configurations for a given dataset
dataset_name = args.dataset
cf, load_dataset = get_dataset_config(args.dataset)
savepath = args.save_path
num_epochs = cf.num_epochs
batch_size = args.batch_size
resize = cf.image_size
requires_split = 0 


# setup checkpoint and experiment tracking
if not os.path.isdir(savepath):
    os.mkdir(savepath)
save_point = savepath + '/'+str(args.dataset)+os.sep
if not os.path.isdir(save_point):
    os.mkdir(save_point)


#Select a seed for the experiment
args.allow_repeats = True if args.resume_training == True else args.allow_repeats
seed = select_experiment_seed(args.seed,savepath+'/'+str(args.dataset),allow_repeats=args.allow_repeats)
if args.dataset_seed == 'same':
    dataset_seed = int(seed)
else:
    if isinstance(args.dataset_seed,int) == False and args.dataset_seed.isdigit() == False:
        raise ValueError('Seed must be an integer')
    dataset_seed = int(args.dataset_seed)


#Load the dataset
if load_dataset == 1: #Process the dataset which does not need splitting

    classes_ID = cf.classes
    classes_OOD = []

    #Split data into test, validation and training sets
    df_train_full = cf.train_ID
    train_idx, val_idx = train_test_split(list(range(len(df_train_full))), test_size=args.valSize,stratify=df_train_full.targets,random_state=int(dataset_seed))

    #Get dataloader for test and validation sets
    df_train = torch.utils.data.Subset(df_train_full, train_idx)
    df_validation = torch.utils.data.Subset(df_train_full, val_idx)
    df_validation.transform = cf.transform_test
    args_train = {'resize': resize, 'batch_size': batch_size, 'shuffle': True, 'pin_memory': use_cuda, 'device_count': device_count}
    trainloader = get_dataloader(args=SimpleNamespace(**args_train), dataset=df_train)

else: #Process the dataset which needs splitting

    classes_ID = cf.classes_ID[args.setting] if args.setting in cf.dataset_selection_settings.keys() else args.class_selections['classes_ID']
    classes_OOD = cf.classes_OOD[args.setting] if args.setting in cf.dataset_selection_settings.keys() else args.class_selections['classes_OOD']
    requires_split = 1

    # load dataframe 
    path_train = os.path.join(cf.root, 'train.csv')
    data_temp = pd.read_csv(path_train)
    path_train_valid = os.path.join(cf.root, 'valid.csv')
    data_valid = pd.read_csv(path_train_valid)
    data = pd.concat([data_temp,data_valid])

    #Applies selections on the data to get desired training and validation datasets
    Dataset_selection = Dataset_selection_methods(data,cf,mode='train')
    class_selections, demographic_selections, dataset_selections, train_val_test_split_criteria = get_dataset_selections(cf,args,dataset_seed)
    dataset = Dataset_selection.apply_selections(class_selections=class_selections,
                                                 demographic_selections=demographic_selections,
                                                 dataset_selections=dataset_selections,
                                                 train_val_test_split_criteria=train_val_test_split_criteria)
    
    
    assert 'train_df' in dataset.keys(), 'train_df not in dataset, check mode of Dataset_selection_methods is set to "train"'
    assert 'validation_df' in dataset.keys(), 'validation_df not in dataset, check mode of Dataset_selection_methods is set to "train"'
    
    #Get dataloader for test and validation sets, which are class balanced
    df_train = cf.Database_class(cf.loader_root, dataset['train_df'], cf.transform_train[args.setting])
    df_validation = cf.Database_class(cf.loader_root, dataset['validation_df'], cf.transform_test[args.setting])
    # trainloader uses a weighted sampler as training data is class imbalanced
    train_weights = get_class_weights(dataset['train_df']) #Get weights for each class 
    args_train = {'resize': resize, 'batch_size': batch_size, 'shuffle': True, 'root': cf.loader_root, 'pin_memory': use_cuda, 'device_count': device_count}
    trainloader = get_weighted_dataloader(args=SimpleNamespace(**args_train), dataset=df_train, weights=train_weights)

args_validation = {'resize': resize, 'batch_size': batch_size, 'shuffle': False, 'pin_memory': use_cuda, 'device_count': device_count}
validationloader = get_dataloader(args=SimpleNamespace(**args_validation), dataset=df_validation)

 
# Displays information about the experiment
if args.verbose == True:
    print('\nExperiment details:')
    print('| Running experiment seed: {}'.format(seed))
    print(f'| Classes ID: {classes_ID}')
    print(f'| Length of train dataset: {len(df_train)}')
    print(f'| Length of validation dataset: {len(df_validation)}')

num_classes = int(dataset['train_df']['class'].max()) + 1
assert num_classes > 1, 'There must be more than one class to train the neural network'


if args.resume_training != True: #Get the network architecture and initialise the network
    net, file_name = get_network_architecture(args,num_classes,cf.df_name)
    net = set_activation_function(net,activation_function=args.act_func)
    net = initialise_network(net,initialisation_method=cf.initialisation_method)
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

else: #Load a saved model
    if args.seed == 'random':
        raise ValueError('Cannot resume training with a random seed')
    net, net_dict, cf_load_net = load_net(args.seed,use_cuda=use_cuda)
    file_name = net_dict['pathname']+'_resume_training'
    assert cf_load_net.df_name == cf.df_name, 'Dataset in the saved model does not match the current experiment'
    assert net_dict['num_classes'] == num_classes, 'Number of classes in the saved model does not match the number of classes in the current experiment'
    
criterion = get_criterion(criterion_name=cf.criterion,label_smoothing=args.label_smoothing)
optimiser, scheduler = get_optimiser_scheduler(net,args,cf,trainloader,num_epochs)


#Displays information about model training
if args.verbose == True:
    print('\nTraining model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimiser = ' + str(args.optimiser))
    print('| Scheduler = ' + str(args.scheduler))
    print('| Batch size = ' + str(batch_size))


#Record the model details
Model_details = [0,file_name+'-'+str(seed), seed, cf.df_name,args.setting[-1],args.net_type,args.depth,args.widen_factor,args.dropout,0,0,
                 requires_split,dataset_seed,args.act_func,class_selections,demographic_selections,dataset_selections,train_val_test_split_criteria,num_classes,args.act_func_dropout,
                 args.lr,args.optimiser,args.scheduler,args.max_lr,args.label_smoothing,args.batch_size,cf.initialisation_method,args.save_path,cf.criterion,args.save_model]
record_model('outputs/saved_models/model_list.csv',Model_details)


#Train the model
training_dict = {'net': net,
                 'trainloader': trainloader,
                 'validationloader': validationloader,
                 'optimiser': optimiser,
                 'scheduler': scheduler,
                 'criterion': criterion,
                 'scheduler_name': args.scheduler,
                 'num_epochs': num_epochs,
                 'use_cuda': use_cuda,
                 'verbose': args.verbose,
                 'save_model_mode': args.save_model,
                 'save_point': save_point,
                 'filename': file_name,
                 'seed': seed,
                 }
train_model = Train_DNN(training_dict)
train_model(gradient_clipping=args.gradient_clipping)