import argparse
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from types import SimpleNamespace
import copy
import ast
from source.util.general_utils import select_cuda_device
from source.util.processing_data_utils import get_dataloader, get_weighted_dataloader, get_ood_dataset, get_dataset_selections
from source.util.training_utils import get_class_weights
from source.util.evaluate_network_utils import load_net, evaluate_ood_detection_method, evaluate_accuracy, ensure_class_overlap, expand_classes
from source.util.Select_dataset import Dataset_selection_methods
from make_synthetic_artefacts import RandomErasing_square, RandomErasing_triangle, RandomErasing_polygon, RandomErasing_ring, RandomErasing_text, add_Gaussian_noise, modify_transforms


parser = argparse.ArgumentParser(description='Evaluate OOD detection method')
parser.add_argument('--method', '-m', type=str, default='MCP',
                    help='Method for OOD Detection, one of [MCP (default), ODIN, MCDP, Mahalanobis, DeepEnsemble]')
parser.add_argument('--cuda_device', default='none', type=str,
                    help='Select device to run code, could be device number or all')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size (typically order of 2), default: 32')
parser.add_argument('--verbose',default=True,type=bool, help='verbose')
parser.add_argument('--seed', default='82868', type=str,
                    help='Select experiment seed')
parser.add_argument('--ood_class_selections', '-c_sel', default={'classes_ID': ['Fracture'], 'classes_OOD': ['Cardiomegaly','Pneumothorax']}, type=dict, 
                    help='The class selections to be used if the args.ood_setting is not known.')
parser.add_argument('--ood_demographic_selections', '-d_sel', default={}, type=dict,
                    help='The demographic selections to be used if the args.ood_setting is not known.')
parser.add_argument('--ood_dataset_selections', '-dataset_s', default={'seperate_patient_IDs': True}, type=dict,
                    help='The dataset specific selections to be used if the args.ood_setting is not known (default is for CheXpert).')
parser.add_argument('--ood_train_val_test_split_criteria', '-split_sel', default={'valSize': 0, 'testSize': 1}, type=dict,
                    help='The dataset splitting criteria to be used if the args.ood_setting is not known.')
#Used for selecting the OOD type
parser.add_argument('--ood_type', default='different_class',
                    help='Select ood task')
#Arguments for synthetic OOD
parser.add_argument('--synth_artefact', default='square', type=str,
                    help='Form of synthetic artefact.')
parser.add_argument('--synth_scale', default=(0.1,0.1), type=tuple,
                    help='Percentage of image size to scale synthetic artefact. Default: (0.1,0.1)')
parser.add_argument('--synth_ratio', default=(1,1), type=tuple,
                    help='Ratio of synthetic artefact. Default: (1,1)')
parser.add_argument('--synth_value', default=0, type=str,
                    help='Value of synthetic artefact (float) or can be [random_gaussian_noise, random_uniform_noise, foreign_texture, image_replace, image_replace_no_overlap] Default: 0')
parser.add_argument('--synth_setting', default='random', type=str,
                    help='Location of synthetic artefact in image, can be [random, centred, near_centre, periphery, corners, near_corners, near_periphery]. Default: random')
parser.add_argument('--synth_noise_mean', default='img_mean', type=str,
                    help='Mean of the Gaussian noise used if synth_value is random_gaussian_noise. Default: img_mean')
parser.add_argument('--synth_noise_std', default='img_std', type=str,
                    help='Standard deviation of the Gaussian noise used if synth_value is random_gaussian_noise. Default: img_std')
parser.add_argument('--synth_coarseness', default=1, type=float,
                    help='Coarseness of the noise or foriegn texture of the synthetic artefact (>=1). Default: 1')
parser.add_argument('--synth_rotation_angle', default=0, type=str,
                    help='Rotation angle of synthetic artefact (float), can be "random". Default: 0')
parser.add_argument('--synth_foreign_texture', default=torch.tensor(np.kron([[1,0]*5,[0,1]*5]*5, np.ones((10, 10))),dtype=torch.float32),
                    help='Foreign texture of synthetic artefact, should be 2D or 3D')
parser.add_argument('--synth_gaussian_filter_sigma', default=0, type=float,
                    help='Standard deviation of Gaussian filter to be applied to the erased area to smooth the edges of the erased area. Default is 0.')
parser.add_argument('--synth_make_transparent', default=False, type=bool,
                    help='Whether to make the synthetic artefact transparent. Default is False.')
parser.add_argument('--synth_transparency_power', default=5, type=float,
                    help='Power to raise the transparency mask to. Default is 1.')
parser.add_argument('--synth_triangle_type', default='equilateral', type=str,
                    help='Type of triangle to use for synthetic artefact, can be [equilateral, isosceles, right]. Default is equilateral.')
parser.add_argument('--synth_polygon_coordinates', default=np.array([(112, 50), (118, 80), (142, 80), (124, 98), (140, 122),(112, 110), (84, 122), (100, 98), (82, 80), (106, 80)]),
                    help='Coordinates of the polygon to be used for synthetic artefact. Default is random.')
parser.add_argument('--synth_ellipse_parameter', default=1, type=float,
                    help='Ellipcticity of the ring (<=1). Default is 1.')
parser.add_argument('--synth_ring_width', default=20, type=float,
                    help='Width of the ring. Default is 20.')
parser.add_argument('--synth_text', default='OOD', type=str,
                    help='Text to be used for synthetic artefact. Default is OOD.')
parser.add_argument('--synth_font_family', default='sans-serif', type=str,
                    help='Font family to be used for synthetic artefact. Default is sans-serif.')
#Arguments for selecting the OOD dataset
parser.add_argument('--ood_dataset', default='SHVM', type=str,
                    help='Name of dataset to be used for ood.')
#Arguments for OOD detection methods
parser.add_argument('--filename', default='practise', type=str,
                    help='Name of dataset to be used for ood.')
parser.add_argument('--temperature', '-T', default='1', type=float,
                    help='Temperature (used for methods like ODIN)')
parser.add_argument('--noiseMagnitude', '-eps', default='0', type=float,
                    help='Perturbation epislon (used for methods like ODIN)')
parser.add_argument('--MCDP_samples', default='10', type=int,
                    help='Samples used for method MCDP')
parser.add_argument('--deep_ensemble_seed_list','-DESL', default='[]', type=str,
                    help='List of seeds to be used for deep ensemble')
parser.add_argument('--save_results', default=False, type=bool,
                    help='Boolean whether to save results for OOD detection')
parser.add_argument('--plot_metric', default=False, type=bool,
                    help='Boolean whether to plot metric of OOD detection metrics')
parser.add_argument('--return_metrics', default=False, type=bool,
                    help='Whether to return AUROC and AUCPR')
parser.add_argument('--evaluate_ID_accuracy', default=False, type=bool,
                    help='Whether to measure the accuracy of the ID test dataset')
parser.add_argument('--evaluate_OOD_accuracy', default=False, type=bool,
                    help='Whether to measure the accuracy of the OOD test dataset (requires overlap with OOD dataset)')
parser.add_argument('--mahalanobis_module', default=-1,
                    help='Module of DNN to apply Mahalanobis distance')
parser.add_argument('--mahalanobis_feature_combination', default=False, type=bool,
                    help='Combine the distances from different modules into one distance')
parser.add_argument('--mahalanobis_alpha', default=None, 
                    help='List of weights to be used for combining the distances from different modules. Should be same length as mahalanobis modules')
parser.add_argument('--mahalanobis_RMD', default=False, type=bool,
                    help='Whether to use relative Mahalanobis (True) or Mahalanobis (False). Default is False.')
parser.add_argument('--mahalanobis_preprocess', default=False, type=bool,
                    help='Whether to preprocess an image with FGSM before calculating Mahalanobis distance. Default is False.')
parser.add_argument('--MBM_type', default='MBM', 
                    help='The type of Multi-Branch Mahalanobis to use, should be MBM or MBM_act_func_only. Default is MBM.')
args = parser.parse_args()

#Check that a valid ood_type is selected
valid_ood_type_options = ['different_dataset','synthetic','different_class']
if isinstance(args.ood_type,list):
    for item in args.ood_type:
        if item not in valid_ood_type_options:
            raise Exception('OOD type should be in [different_dataset,synthetic,different_class]')
elif isinstance(args.ood_type,str):
    if args.ood_type not in ['different_dataset','synthetic','different_class']:
        raise Exception('OOD type should be in [different_dataset,synthetic,different_class]')
else:
    raise Exception('OOD type should be in [different_dataset,synthetic,different_class]') 


#Select a GPU to use
use_cuda = select_cuda_device(args.cuda_device)
pin_memory = use_cuda
device_count = torch.cuda.device_count()


#Load net with its associated parameters
net, net_dict, cf = load_net(args.seed,use_cuda=use_cuda)
classes_ID = net_dict['classes_ID']
classes_OOD = net_dict['classes_OOD']
args.setting = net_dict['setting']
num_classes = net_dict['num_classes']
file_name = net_dict['file_name']
dataset_seed = net_dict['train_val_test_split_criteria']['dataset_seed']
save_dir = net_dict['save_dir']
requires_split = net_dict['Requires split']


#Load parameters specific to this database
batch_size = args.batch_size 
resize = cf.image_size # image size
kwargs_method = {}


if requires_split == 0: #If dataset does not need to be split
    df_ID_test = cf.ID_dataset
    if args.method in ['mahalanobis','MBM']:
        df_ID_train = cf.train_ID
else: #Else split dataset by classes in and classes out
    root = cf.root
    loader_root = cf.loader_root
    # load dataframe 
    path_train = os.path.join(cf.root, 'train.csv')
    data_temp = pd.read_csv(path_train)
    path_train_valid = os.path.join(cf.root, 'valid.csv')
    data_valid = pd.read_csv(path_train_valid)
    data = pd.concat([data_temp,data_valid])

    ID_Dataset_selection = Dataset_selection_methods(data,cf,mode='test')
    ID_dataset = ID_Dataset_selection.apply_selections(class_selections=net_dict['class_selections'],
                                                 demographic_selections=net_dict['demographic_selections'],
                                                 dataset_selections=net_dict['dataset_selections'],
                                                 train_val_test_split_criteria=net_dict['train_val_test_split_criteria'])
    
    df_ID_test = cf.Database_class(cf.loader_root, ID_dataset['test_df'], cf.transform_test[args.setting])
    if args.method in ['mahalanobis','MBM']:
        # get dataloader for training data 
        args_train = {'resize': resize, 'batch_size': batch_size, 'shuffle': False, 'root': loader_root, 'pin_memory': use_cuda, 'device_count': device_count}
        train_weights = get_class_weights(ID_dataset['train_df'])
        df_ID_train = cf.Database_class(cf.loader_root, ID_dataset['train_df'], cf.transform_test[args.setting])
        trainloader = get_weighted_dataloader(args=SimpleNamespace(**args_train), dataset=df_ID_train, weights=train_weights)
        kwargs_method = {'trainloader': trainloader}

OOD_dataset_split = 0
if 'different_dataset' in args.ood_type:
    df_OOD, classes_OOD, OOD_dataset_split = get_ood_dataset(args.ood_dataset)
if 'different_class' in args.ood_type or OOD_dataset_split==1:
    OOD_Dataset_selection = Dataset_selection_methods(data,cf,mode='test')
    OOD_class_selections, OOD_demographic_selections, OOD_dataset_selections, OOD_train_val_test_split_criteria = get_dataset_selections(cf,args,dataset_seed,get_ood_data=True)
    OOD_dataset = OOD_Dataset_selection.apply_selections(class_selections=OOD_class_selections,
                                                 demographic_selections=OOD_demographic_selections,
                                                 dataset_selections=OOD_dataset_selections,
                                                 train_val_test_split_criteria=OOD_train_val_test_split_criteria)
    
    if args.evaluate_OOD_accuracy == True: #Ensures there is class overlap in OOD and ID datasets to test accuracy
        OOD_dataset = ensure_class_overlap(OOD_dataset,classes_ID,classes_OOD)
    df_OOD = cf.Database_class(cf.loader_root, OOD_dataset['test_df'], cf.transform_test[args.setting])

if args.ood_type == 'synthetic':
        df_OOD = copy.deepcopy(df_ID_test)


if 'synthetic' in args.ood_type: 
    transform_keys = ['scale', 'ratio', 'value', 'setting', 'noise_mean',
    'noise_std', 'coarseness', 'rotation_angle', 'foreign_texture',
    'gaussian_filter_sigma', 'make_transparent', 'transparency_power',
    'triangle_type', 'polygon_coordinates', 'ellipse_parameter',
    'ring_width', 'text', 'font_family']

    transform_kwargs = {key: getattr(args, f'synth_{key}') for key in transform_keys}
    transform_kwargs['p'] = 1

    kernel_size = tuple(np.array(args.synth_scale)*cf.image_size)

    # Dictionary to map synth_artefact values to functions
    artefact_to_transform = {
        'square': RandomErasing_square(**transform_kwargs),
        'triangle': RandomErasing_triangle(**transform_kwargs),
        'polygon': RandomErasing_polygon(**transform_kwargs),
        'ring': RandomErasing_ring(**transform_kwargs),
        'text': RandomErasing_text(**transform_kwargs),
        'Gaussian_noise': add_Gaussian_noise(**transform_kwargs),
        'Gaussian_blur': T.GaussianBlur(kernel_size=tuple(map(lambda x: round(x)+1 if round(x)%2==0 else round(x), kernel_size)), 
                                        sigma=(args.synth_gaussian_filter_sigma if args.synth_gaussian_filter_sigma != 0 else 1)),
        'invert': T.RandomInvert(p=1),
    }

    # Check if args.synth_artefact is in the dictionary, then apply the corresponding transform
    if args.synth_artefact in artefact_to_transform:
        transform_fn = artefact_to_transform[args.synth_artefact]
        df_OOD.transform = modify_transforms(transform_fn, df_OOD.transform, where_to_insert='insert_after', insert_transform=T.Normalize(mean=[0], std=[1]))
    else:
        raise(Exception(f'Artefact {args.synth_artefact} not recognised. Available options are {list(artefact_to_transform.keys())}'))
    

#Expand list of classes to include all classes in the dataset
classes_ID = expand_classes(classes_ID, net_dict['class_selections'])
if args.ood_type == 'synthetic':
    classes_OOD = classes_ID
else:
    classes_OOD = expand_classes(classes_OOD, OOD_class_selections)

args_dataloader = {'resize': resize, 'batch_size': batch_size, 'shuffle': False, 'pin_memory': use_cuda, 'device_count': device_count}
ID_loader = get_dataloader(args=SimpleNamespace(**args_dataloader), dataset=df_ID_test)
OOD_loader = get_dataloader(args=SimpleNamespace(**args_dataloader), dataset=df_OOD)


#Evaluate accuracy of ID and OOD datasets on the ID classification task
if args.evaluate_OOD_accuracy == True:
    evaluate_accuracy(net,OOD_loader,use_cuda=use_cuda,save_results=args.save_results,plot_metric=args.plot_metric,save_dir='outputs/experiment_outputs',filename='OOD')
if args.evaluate_ID_accuracy == True:
    evaluate_accuracy(net,ID_loader,use_cuda=use_cuda,save_results=args.save_results,plot_metric=args.plot_metric,save_dir='outputs/experiment_outputs')

kwargs_test = {'use_cuda':use_cuda,'verbose':args.verbose}

if args.save_results == True:
    kwargs_test['save_results'] = True
    kwargs_test['save_dir'] = 'outputs/experiment_outputs'
    if args.plot_metric == True:
        kwargs_test['plot_metric'] = True
if args.filename != 'practise':
    kwargs_test['filename'] = '_'+str(args.filename)
#Set parameters required for each OOD detection method
if args.method == 'ODIN':
    kwargs_test['temper'] = args.temperature
    kwargs_test['noiseMagnitude'] = args.noiseMagnitude
if args.method == 'MCDP':
    kwargs_test['samples'] = args.MCDP_samples
    kwargs_test['two_dim_dropout_rate'] = net_dict['act_func_dropout_rate']
if args.method == 'deepensemble':
    if args.deep_ensemble_seed_list == '[]':
        raise Exception('Please specify a list of seeds to be used for deep ensemble')
    kwargs_test['net_dict'] = net_dict
    kwargs_test['seed_list'] = args.deep_ensemble_seed_list
if args.method == 'mahalanobis' or args.method == 'MBM':
    kwargs_test['trainloader'] = trainloader
    kwargs_test['module'] = ast.literal_eval(args.mahalanobis_module)
    kwargs_test['num_classes'] = num_classes
    kwargs_test['feature_combination'] = True if args.method == 'MBM' or args.mahalanobis_feature_combination == True else False
    kwargs_test['alpha'] = args.mahalanobis_alpha
    kwargs_test['preprocess'] = args.mahalanobis_preprocess
    kwargs_test['RMD'] = args.mahalanobis_RMD
if args.method == 'MBM':
    kwargs_test['net_type'] = net_dict['net_type']+'_with_dropout' if net_dict['act_func_dropout_rate'] > 0 else net_dict['net_type']
    kwargs_test['MBM_type'] = args.MBM_type


np.random.seed(int(args.seed))
torch.manual_seed(int(args.seed))

if args.return_metrics == True:
    AUROC, AUCPR = evaluate_ood_detection_method(args.method,net,ID_loader,OOD_loader,return_metrics=True,**kwargs_test)
else:
    evaluate_ood_detection_method(args.method,net,ID_loader,OOD_loader,**kwargs_test)
