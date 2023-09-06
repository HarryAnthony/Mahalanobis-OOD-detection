"""
Helper functions for processing and loading data.
"""
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torchvision

def balance_df(df,random_state=42):
    """
    Create a dataframe where the number of entries of each class is balanced.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to balance.
    random_state : int, optional
        The random state to use for sampling, by default 42

    Returns
    -------
    pd.DataFrame
        The balanced dataframe.
    """
    df = df.groupby('class')
    return df.apply(lambda x: x.sample(df.size().min(),random_state=random_state).reset_index(drop=True))


def rows_not_in_df(df_larger,df_smaller):
    """
    Creates a dataframe of rows that are included in a larger df and not the smaller df.

    Parameters
    ----------
    df_larger : pd.DataFrame
        The larger dataframe.
    df_smaller : pd.DataFrame
        The smaller dataframe.

    Returns
    -------
    pd.DataFrame
        The dataframe of rows that are included in a larger df and not the smaller df.
    """
    df_larger.reset_index(drop = True, inplace = True)
    df_smaller.reset_index(drop = True, inplace = True)

    df_all = df_larger.merge(df_smaller.drop_duplicates(), on=df_larger.columns.tolist(), 
                   how='left', indicator=True)
    return df_all[df_all['_merge'] == 'left_only']


def get_dataset_config(database):
    """
    Get the configuration of the dataset.

    Parameters
    ----------
    database : str
        The name of the dataset.
    Boolean
        Whether the dataset requires splitting into train, validation and test sets.
    """
    if database == 'chexpert':
        from config import chexpert as cf_chexpert
        return cf_chexpert, 0
    elif database == 'cifar10':
        from config import cifar10 as cf_cifar10
        return cf_cifar10, 1
    elif database == 'MIMIC':
        from config import MIMIC as cf_MIMIC
        return cf_MIMIC, 0
    else:
        raise Exception('Database configuration unknown.')
      

def get_dataset_selections(cf,args,dataset_seed,get_ood_data=False):
    """
    Load the dataset selections for a given setting. If the setting is not known, the selections in the args are used instead.

    Parameters
    ----------
    cf : str
        The dataset configuration file
    args : argparse.Namespace
        The arguments to use for the experiment.
    dataset_seed : int
        The seed to use for splitting the dataset into train, validation and test sets.
    get_ood_data : bool, optional
        Whether to get out of distribution data, by default False

    Returns
    -------
    class_selections : list
        The selections for the classes to use in the experiment.
    demographic_selections : list
        The selections for the demographics to use in the experiment.
    dataset_selections : list
        The dataset specific selections to use in the experiment.
    train_val_test_split_criteria : dict
        The criteria to use for splitting the dataset into train, validation and test sets.
    """
    setting = args.setting
    if get_ood_data==True:
        dataset_selection_settings = cf.OOD_selection_settings
    else:
        dataset_selection_settings = cf.dataset_selection_settings

    if setting in dataset_selection_settings: #Load dataset selections for a given setting
        class_selections = dataset_selection_settings[setting]['class_selections']
        demographic_selections = dataset_selection_settings[setting]['demographic_selections']
        dataset_selections = dataset_selection_settings[setting]['dataset_selections']
        train_val_test_split_criteria = dataset_selection_settings[setting]['train_val_test_split_criteria']
    else: #If setting is not known, use the selections in the args instead
        if get_ood_data:
            class_selections = args.ood_class_selections
            demographic_selections = args.ood_demographic_selections
            dataset_selections = args.ood_dataset_selections
            train_val_test_split_criteria = args.ood_train_val_test_split_criteria
        else:
            class_selections = args.class_selections
            demographic_selections = args.demographic_selections
            dataset_selections = args.dataset_selections
            train_val_test_split_criteria = args.train_val_test_split_criteria

    train_val_test_split_criteria['dataset_seed'] = float(dataset_seed)
    if 'k_fold_split' in train_val_test_split_criteria.keys():
        train_val_test_split_criteria['fold'] = args.fold

    return class_selections, demographic_selections, dataset_selections, train_val_test_split_criteria


def worker_init_fn(worker_id):
    """
    Worker init function for the dataloader.

    Parameters
    ----------
    worker_id : int
        The worker id.
    """                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataloader(args, dataset, drop_last_batch=False):
    """
    Get the dataloader for the dataset which has balanced classes.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments to use for the experiment.
    dataset : torch.utils.data.Dataset
        The dataset to use for the dataloader.
    drop_last_batch : bool, optional
        Whether to drop the last batch, by default False

    Returns
    -------
    torch.utils.data.DataLoader
        The dataloader for the dataset.
    """
    if args.pin_memory:
        return DataLoader(dataset, args.batch_size, shuffle=args.shuffle,pin_memory=True, num_workers=args.device_count*4, prefetch_factor = args.device_count, drop_last=drop_last_batch, persistent_workers=True, worker_init_fn=worker_init_fn)
    return  DataLoader(dataset, args.batch_size, shuffle=args.shuffle,pin_memory=False, drop_last=drop_last_batch, persistent_workers=False, worker_init_fn=worker_init_fn)


def get_weighted_dataloader(args, dataset, weights, drop_last_batch=False):
    """
    Get a weighted dataloader for the dataset, used for a dataset with unbalanced classes,
    such that the batches are sampled according to the weights.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments to use for the experiment.
    dataset : torch.utils.data.Dataset
        The dataset to use for the dataloader.
    weights : list
        The weights for each sample in the dataset.
    drop_last_batch : bool, optional
        Whether to drop the last batch, by default False

    Returns
    -------
    torch.utils.data.DataLoader
        The dataloader for the dataset.
    """
    samples_weight = torch.FloatTensor(weights)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    if args.pin_memory:
        return DataLoader(dataset, args.batch_size, shuffle=False, sampler=sampler, pin_memory=True, num_workers=args.device_count*4, prefetch_factor = args.device_count, drop_last=drop_last_batch, persistent_workers=True, worker_init_fn=worker_init_fn)
    return  DataLoader(dataset, args.batch_size, shuffle=False, sampler=sampler, pin_memory=False, drop_last=drop_last_batch, persistent_workers=False, worker_init_fn=worker_init_fn)


def get_ood_dataset(dataset_name):
    """
    Get a dataset unseen by the model to use to out-of-distribution detection. The dataset must
    be in the list [shvm, lsun, cifar10, cifar100, imagenet, tinyimagenet].
    
    Parameters
    ----------
    dataset_name : str
        The name of the OOD dataset.
        
    Returns
    -------
    torch.utils.data.Dataset
        The OOD dataset.
    list
        The classes of the OOD dataset.
    """
    if dataset_name not in ['SHVM', 'CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet']:
        raise ValueError('Dataset not supported. The dataset must be in the list [SHVM, LSUN, cifar10, cifar100, imagenet, tinyimagenet].')

    if dataset_name == 'SHVM':   
        from config import SHVM as cf_SHVM
        ood_cf = cf_SHVM
        ood_dataset = torchvision.datasets.SVHN(root='../data/SHVM', split='test', download=True, transform=ood_cf.transform_test)
        classes = cf_SHVM.classes
        ood_dataset_split = 0
    elif dataset_name == 'CIFAR10':
        from config import cifar10 as cf_cifar10
        ood_cf = cf_cifar10
        ood_dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=ood_cf.transform_test)
        classes = cf_cifar10.classes
        ood_dataset_split = 0
    elif dataset_name == 'CIFAR100':
        from config import cifar100 as cf_CIFAR100
        ood_cf = cf_CIFAR100
        ood_dataset = torchvision.datasets.CIFAR100(root='../data/cifar100', train=False, download=True, transform=ood_cf.transform_test)
        classes = cf_CIFAR100.classes
        ood_dataset_split = 0
    
    return ood_dataset, classes, ood_dataset_split