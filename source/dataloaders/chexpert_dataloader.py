from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import  Dataset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

class ChexpertSmall(Dataset):
    def __init__(self, folder_dir, dataframe, transform):
        """
        Class for the CheXpert small dataset (# adapted from https://github.com/kamenbliznashki/chexpert/blob/2bf52b1b70c3212a4c2fd4feacad0fd198fe8952/dataset.py#L17)
        
        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        transform: torchvision.transforms
            transform to apply to images
        """
        self.image_paths = [] # List of image paths
        self.targets = [] # List of image labels
        self.patient_ids = []
        
        self.transform = transform

        for row in dataframe.to_dict('records'):
            self.patient_ids.append(row['Path']) 
            image_path = str(folder_dir) + str(row['Path'])
            self.image_paths.append(image_path)
            self.targets.append(row['class'])

        assert len(self.targets) == len(self.image_paths), 'Label count does not match the image count.'

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        image_path = self.image_paths[index]
        image_data = Image.open(image_path) 

        if self.transform is not None:
            image = self.transform(image_data)

        return image, self.targets[index] 


#Dataset specific selection functions

def select_pacemaker_images(dataset,criteria=['remove all images without pacemaker']):
    """
    A function for selecting images with a pacemaker and controlling how they are used in the dataset.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images with a pacemaker. These include:
        'remove all images without pacemaker': Remove all images without a pacemaker
        'set_to_train': Put all images with a pacemaker into the training set
        'set_to_val': Put all images with a pacemaker into the validation set
        'set_to_test': Put all images with a pacemaker into the test set
        'make_pacemaker_binary_classifier': Make a binary classifier with all images with a pacemaker as class 1 and the remaining as class 0
        'make_pacemaker_class': Make a new class for pacemaker without changing the other classes

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    pacemaker_list = np.loadtxt("data/CheXpert/manual_annotations/pacemaker.txt",dtype=str)
    pacemaker_list = ['CheXpert-v1.0-small/'+str(element) for element in pacemaker_list ]
    #Get all images with pacemaker
    pacemaker_data =  dataset['total_df'][dataset['total_df']['Path'].isin(pacemaker_list)]

    #Remove all images that do not contain a pacemaker image (useful for making an ID or OOD dataset)
    if 'remove all images without pacemaker' in criteria:
        dataset['total_df'] = pacemaker_data 
    
    #Put all images with a pacemaker into the training, validation or test set (enables more control over the dataset images)
    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], pacemaker_data]) if 'train_df' in dataset else pacemaker_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], pacemaker_data]) if 'validation_df' in dataset else pacemaker_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], pacemaker_data]) if 'test_df' in dataset else pacemaker_data
        dataset['total_df'] = dataset['total_df'].drop(pacemaker_data.index)

    #Set all images with a pacemaker to class 1 and the remaining to class 0 (for making a binary classifier)
    if 'make_pacemaker_binary_classifier' in criteria:
        dataset['total_df']['class'] = dataset['total_df']['Path'].apply(lambda x: 1 if x in pacemaker_list else 0)
    elif 'make_pacemaker_class' in criteria: #Make a new class for pacemaker
        new_class_int = int(max(dataset['total_df']['class']) + 1)
        dataset['total_df']['class'] = dataset['total_df']['Path'].map({x: new_class_int for x in pacemaker_list})

    return dataset


def select_no_support_device_images(dataset,criteria=['remove all images with support device']):
    """
    A function for selecting images with no support device and controlling how they are used in the dataset.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images with a no support device. These include:
        'remove all images with support device': Remove all images with support device
        'set_to_train': Put all images with no support device into the training set
        'set_to_val': Put all images with no support device into the validation set
        'set_to_test': Put all images with no support device into the test set
        'make_no_support_device_binary_classifier': Make a binary classifier with all images with no support device as class 1 and the remaining as class 0
        'make_no_support_device_class': Make a new class for no support device images without changing the other classes

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    nsd_list = np.loadtxt("data/CheXpert/manual_annotations/no_support_device.txt",dtype=str)
    nsd_list = ['CheXpert-v1.0-small/'+str(element) for element in nsd_list ]
    #Get all images with no support device
    nsd_data =  dataset['total_df'][dataset['total_df']['Path'].isin(nsd_list)]

    #Remove all images that contain a support device (useful for making an ID or OOD dataset)
    if 'remove all images with support device' in criteria:
        dataset['total_df'] = nsd_data 
    
    #Put all images with no support device into the training, validation or test set (enables more control over the dataset images)
    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], nsd_data]) if 'train_df' in dataset else nsd_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], nsd_data]) if 'validation_df' in dataset else nsd_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], nsd_data]) if 'test_df' in dataset else nsd_data
        dataset['total_df'] = dataset['total_df'].drop(nsd_data.index)

    #Set all images with no support device into class 1 and the remaining to class 0 (for making a binary classifier)
    if 'make_no_support_device_binary_classifier' in criteria:
        dataset['total_df']['class'] = dataset['total_df']['Path'].apply(lambda x: 1 if x in nsd_data else 0)
    elif 'make_no_support_device_class' in criteria: #Make a new class for no support device
        new_class_int = int(max(dataset['total_df']['class']) + 1)
        dataset['total_df']['class'] = dataset['total_df']['Path'].map({x: new_class_int for x in nsd_data})

    return dataset


def select_support_device_images(dataset,criteria=['remove all images without support device']):
    """
    A function for selecting images with a support device and controlling how they are used in the dataset.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images with a support device. These include:
        'remove all images without support device': Remove all images without a support device
        'set_to_train': Put all images with a support device into the training set
        'set_to_val': Put all images with a support device into the validation set
        'set_to_test': Put all images with a support device into the test set
        'make_support_device_binary_classifier': Make a binary classifier with all images with a support device as class 1 and the remaining as class 0
        'make_support_device_class': Make a new class for support device images without changing the other classes

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    sd_list = np.loadtxt("data/CheXpert/manual_annotations/other_support_device.txt",dtype=str)
    sd_list = ['CheXpert-v1.0-small/'+str(element) for element in sd_list ]
    #Get all images with support device
    sd_data =  dataset['total_df'][dataset['total_df']['Path'].isin(sd_list)]

    #Remove all images that do not contain a support device (useful for making an ID or OOD dataset)
    if 'remove all images without support device' in criteria:
        dataset['total_df'] = sd_data

    #Put all images with a support device into the training, validation or test set (enables more control over the dataset images)
    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], sd_data]) if 'train_df' in dataset else sd_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], sd_data]) if 'validation_df' in dataset else sd_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], sd_data]) if 'test_df' in dataset else sd_data
        dataset['total_df'] = dataset['total_df'].drop(sd_data.index)

    #Set all images with a support device into class 1 and the remaining to class 0 (for making a binary classifier)
    if 'make_support_device_binary_classifier' in criteria:
        dataset['total_df']['class'] = dataset['total_df']['Path'].apply(lambda x: 1 if x in sd_data else 0)
    elif 'make_support_device_class' in criteria:
        new_class_int = int(max(dataset['total_df']['class']) + 1)
        dataset['total_df']['class'] = dataset['total_df']['Path'].map({x: new_class_int for x in sd_data})

    return dataset


def seperate_patient_IDs(dataset,k_fold_split=False,**kwargs):
    """
    A function for splitting the dataset into training, validation and test sets, while ensuring that there are no duplicate patient IDs in the validation and test sets.
    This is specifically designed for the CheXpert dataset, where there are multiple images for each patient ID.
    
    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    k_fold_split: bool
        Whether to use k-fold cross validation, or a single test/train/validation set. default: False

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    if k_fold_split == True:
        dataset = k_fold_split_seperate_patient_IDs(dataset,**kwargs)
    else:
        dataset = train_test_val_split_seperate_patient_IDs(dataset,**kwargs)
    dataset['is_split'] = True
    return dataset


def remove_duplicate_patient_ID_images(df):
    """
    A function for removing duplicate patient ID images from a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the dataset information

    Returns
    -------
    filtered_df: pd.DataFrame
        A dataframe containing the dataset information, with duplicate patient ID images removed
    """
    # Extract the patient IDs from the paths
    df['patient_id'] = df['Path'].apply(lambda x: x.split('/')[2])
    # Group the dataframe by patient ID and keep only the first path for each patient
    filtered_df = df.groupby('patient_id').first().reset_index()
    # Drop the patient_id column
    filtered_df.drop('patient_id', axis=1, inplace=True)
    df.drop('patient_id', axis=1, inplace=True)
    
    return filtered_df


def check_no_common_entries(list1, list2):
    """
    A function for checking whether two lists have any common entries.
    """
    set1 = set(list1)
    set2 = set(list2)
    common_entries = set1.intersection(set2)
    return len(common_entries) == 0
    

def k_fold_split_seperate_patient_IDs(dataset,fold=0,k=5,dataset_seed=42):
    """
    A function for splitting the dataset into training and validation sets, while ensuring that there are no duplicate patient IDs in the training and validation datasets.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    fold: int
        The fold to use for the validation set
    k: int
        The number of folds to use for the training set
    dataset_seed: int
        The seed to use for splitting the dataset

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    dataset['total_df']['patient_id'] = dataset['total_df']['Path'].apply(lambda x: x.split('/')[2]) # Extract patient ID from the path column
    patient_counts = dataset['total_df']['patient_id'].value_counts() # Count the number of images per patient
    single_images = dataset['total_df'][dataset['total_df']['patient_id'].map(patient_counts) == 1] # Filter out patients with only one image

    if min(single_images['class'].value_counts()) < k: #If there are too few classes, don't stratify
        kf = KFold(n_splits=k, shuffle=True, random_state=int(dataset_seed))
        single_patient_folds= list(kf.split(single_images)) # Split single_images into folds
    else:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(dataset_seed))
        single_patient_folds= list(skf.split(single_images,single_images['class'])) # Split single_images into folds

    multiple_patient_dict = patient_counts[patient_counts > 1].to_dict() #Get a dictionary of patients with more than one image
    num_images_per_patient_per_fold = [[] for _ in range(k)]
    patient_IDs_per_fold = [[] for _ in range(k)]
    #Assign each patient to a group, so that the number of images per fold is approximately balanced
    for key, value in multiple_patient_dict.items():
        min_sum_group = min(num_images_per_patient_per_fold, key=sum)
        min_sum_group.append(value)
        patient_IDs_per_fold[num_images_per_patient_per_fold.index(min_sum_group)].append(key)

    multiple_patient_folds = [] # Create an empty list to hold the dataframes for each fold
    for idx in range(k):
        # Get the dataset rows into the fold
        rows_in_fold = dataset['total_df'][dataset['total_df']['patient_id'].isin(patient_IDs_per_fold[idx])]
        multiple_patient_folds.append(pd.DataFrame(rows_in_fold)) # Append the dataframe to the list
    
    #Place single patient ID images into the validation and training set
    train_data = single_images.iloc[single_patient_folds[fold][0]]
    validation_data = single_images.iloc[single_patient_folds[fold][1]]
    #Place multiple patient ID images into the test and training set
    fold_indices_list = [0,1,2,3,4]
    dataset['validation_df'] = pd.concat([validation_data,multiple_patient_folds[fold]])
    fold_indices_list.remove(fold)
    for fold_idx in fold_indices_list:
        train_data = pd.concat([train_data,multiple_patient_folds[fold_idx]])
    dataset['train_df'] = train_data

    assert check_no_common_entries(dataset['train_df']['patient_id'],dataset['validation_df']['patient_id']) == True, 'There are the same patient IDs in the training and validation datasets.'
    dataset['train_df'].drop('patient_id', axis=1, inplace=True)
    dataset['validation_df'].drop('patient_id', axis=1, inplace=True)

    return dataset


def train_test_val_split_seperate_patient_IDs(dataset,valSize=0.1,testSize=0.1,balance_classes_in_val_test=True,dataset_seed=42):
    """
    A function for splitting the dataset into training, validation and test sets, while ensuring that there are no duplicate patient IDs in the validation and test sets.
    This is specifically designed for the CheXpert dataset, where there are multiple images for each patient ID.
    
    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    valSize: float
        The fraction of the dataset to use for the validation set
    testSize: float
        The fraction of the dataset to use for the test set
    balance_class_val_test: bool
        Whether to balance the classes in the validation and test sets
    dataset_seed: int
        The seed to use for splitting the dataset

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    if isinstance(valSize,(float,int)) == False or isinstance(testSize,(float,int)) == False or valSize<0 or testSize<0 or valSize+testSize>1:
        raise ValueError('valSize and testSize must be floats between 0 and 1, and valSize+testSize must be less than 1')

    #If all dataset is to be used for training, validation or testing, then return the entire dataset
    if valSize==0 and testSize==0:
        dataset['train_df'] = pd.concat([dataset['train_df'], dataset['total_df']]) if 'train_df' in dataset else dataset['total_df']
        return dataset
    elif valSize==1 and testSize==0:
        dataset['validation_df'] = pd.concat([dataset['validation_df'], dataset['total_df']]) if 'validation_df' in dataset else dataset['total_df']
        return dataset
    elif valSize==0 and testSize==1:
        dataset['test_df'] = pd.concat([dataset['test_df'], dataset['total_df']]) if 'test_df' in dataset else dataset['total_df']
        return dataset

    #Remove duplicate patient ID images
    filtered_df = remove_duplicate_patient_ID_images(dataset['total_df'])

    if balance_classes_in_val_test == True:
        #Make a new dataset with balanced classes, in order to make balanced val/test dataset
        dataset_balanced = balance_df(filtered_df,random_state=int(dataset_seed))
        #Place the entries not included to keep balanced classes into a different dataset
        dataset_remaining = rows_not_in_df(filtered_df,dataset_balanced)
        if (len(filtered_df)/len(dataset_balanced))*(len(dataset['total_df'])/len(filtered_df))*(valSize+testSize) >= 1:
                print('Warning: Dataset too imbalanced to make class balanced val/test sets, so balance_classes_in_val_test=True in train_val_test_split_criteria was ignored.')
                dataset_balanced = filtered_df
                dataset_remaining = pd.DataFrame()
    else:
        #If we don't want to balance the classes, then just use the filtered dataset
        dataset_balanced = filtered_df
        dataset_remaining = pd.DataFrame()

    #Place the repeated patient ID entries into a different dataset, so we have a dataset of unique patient IDs
    dataset_repeated = rows_not_in_df(dataset['total_df'],filtered_df)

    #Split the balanced dataset into the required fractions
    #Adjusts the fraction so that valSize and testSize are fractions of the total dataset
    frac_adjust = (len(filtered_df)/len(dataset_balanced))*(len(dataset['total_df'])/len(filtered_df))

    if min(dataset_balanced['class'].value_counts()) == 1:
        df_train_partial, df_test_full = train_test_split(dataset_balanced, test_size=(valSize+testSize)*frac_adjust, random_state=int(dataset_seed))
    else:
        df_train_partial, df_test_full = train_test_split(dataset_balanced, test_size=(valSize+testSize)*frac_adjust, stratify=dataset_balanced['class'], random_state=int(dataset_seed))
    if testSize==0: #If no test set is required, then just split the dataset into training and validation sets
        dataset['validation_df'] = pd.concat([dataset['validation_df'], df_test_full]) if 'validation_df' in dataset else df_test_full
    elif valSize==0: #If no validation set is required, then just split the dataset into training and test sets
        dataset['test_df'] = pd.concat([dataset['test_df'], df_test_full]) if 'test_df' in dataset else df_test_full
    else: #If both validation and test sets are required, then split the dataset into training, validation and test sets
        if min(dataset_balanced['class'].value_counts()) == 1:
                df_test, df_validation = train_test_split(df_test_full, test_size=(testSize/(valSize+testSize)), random_state=int(dataset_seed))
        else:
            df_test, df_validation = train_test_split(df_test_full, test_size=(testSize/(valSize+testSize)), stratify=df_test_full['class'], random_state=int(dataset_seed))
        dataset['validation_df'] = pd.concat([dataset['validation_df'], df_validation]) if 'validation_df' in dataset else df_validation
        dataset['test_df'] = pd.concat([dataset['test_df'], df_test]) if 'test_df' in dataset else df_test

    #Combine the training dataset with the entries removed when balancing the classes
    dataset['train_df'] = pd.concat([df_train_partial,dataset_remaining,dataset_repeated])

    return dataset


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
