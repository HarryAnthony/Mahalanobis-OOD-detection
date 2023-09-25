import torchvision.transforms as T
import torch
from source.dataloaders.chexpert_dataloader import ChexpertSmall, select_pacemaker_images, seperate_patient_IDs, select_no_support_device_images, select_support_device_images
import numpy as np
from source.util.general_utils import DefaultDict

Database_class = ChexpertSmall

#Training parameters
num_epochs = 400
momentum = 0.9 
weight_decay = 1e-10 
lr_milestones = [int(num_epochs*0.5),int(num_epochs*0.75)]
lr_gamma = 0.2
criterion = 'CrossEntropyLoss'
initialisation_method = 'he'

# network architecture
dropout = 0.3 
depth = 28
widen_factor = 10

# data parameters
image_size = 224

# location of data
root = 'data/CheXpert/CheXpert-v1.0-small'
loader_root = 'data/CheXpert/'
df_name = 'cheXpert'

def database_specific_selections(dataset,selections={},**kwargs):
    """
    Make selections on the dataset which are specific to the CheXpert dataset.

    Parameters
    ----------
    dict
        A dictionary containing the training, validation and test sets.
    selections : dict, optional
        A dictionary containing the selections to be made on the dataset. The default is {}.
        The keys of the dictionary are the names of the selections and the values are the criteria for the selection.
        The possible selections are:
            'pacemaker_selection': selects images with pacemaker devices using the manual annotations.
            'no_support_device_selection': selects images with no support devices using the manual annotations.
            'support_device_selection': selects images with support devices (not including pacemakers) using the manual annotations.
            'seperate_patient_IDs': Ensures that the same patient does not appear in both the training and test sets.

    Returns
    -------
    dict
        A dictionary containing the training, validation and test sets after the selections have been made.
    """
    if 'pacemaker_selection' in selections.keys():
        dataset = select_pacemaker_images(dataset,criteria=selections['pacemaker_selection'])
    if 'no_support_device_selection' in selections.keys():
        dataset = select_no_support_device_images(dataset,criteria=selections['no_support_device_selection'])
    if 'support_device_selection' in selections.keys():
        dataset = select_support_device_images(dataset,criteria=selections['support_device_selection'])
    if 'seperate_patient_IDs' in selections.keys():
        dataset = seperate_patient_IDs(dataset,**kwargs)
    return dataset

#If setting is not known, then will use the default transform with mean and std of the CheXpert dataset
transform_train = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), # to make the images square
            T.RandomRotation(degrees=15), #Randomly rotate the image by 5 degrees
            T.RandomCrop(224, padding=25), #Randomly crop the image by 4 pixels
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.5330],std=[0.0349]),
            lambda x: x.expand(3,-1,-1)]),

            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.RandomRotation(degrees=5),
            T.RandomCrop(224, padding=4),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.4919],std=[0.2953]),
            lambda x: x.expand(3,-1,-1)]),
            
            'setting2' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), 
            T.RandomRotation(degrees=5),
            T.RandomCrop(224, padding=4),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.50688833],std=[0.2894137]),
            lambda x: x.expand(3,-1,-1)]),
            
            'setting3' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.RandomRotation(degrees=15),
            T.RandomCrop(224, padding=20),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.50688833],std=[0.2894137]),
            lambda x: x.expand(3,-1,-1)])})

transform_test = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.5330],std=[0.0349]),
            lambda x: x.expand(3,-1,-1)]),

            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.4919],std=[0.2953]),
            lambda x: x.expand(3,-1,-1)]),
            
            'setting2' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), 
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.50688833],std=[0.2894137]),
            lambda x: x.expand(3,-1,-1)]),
            
            'setting3' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.50688833],std=[0.2894137]),
            lambda x: x.expand(3,-1,-1)])})

#Pre-made dataset selection settings for the CheXpert dataset
dataset_selection_settings = {'setting1': {'class_selections' : {'classes_ID': ['Cardiomegaly','Pneumothorax'], 'classes_OOD': ['Fracture']},
                                           'demographic_selections' : {},
                                           'dataset_selections': {'seperate_patient_IDs': True},
                                           'train_val_test_split_criteria': {'valSize': 0.1, 'testSize': 0.1}},
                            #Setting 1 trains a classifier on Cardiomegaly and Pneumothorax and uses the Fracture class in the dataset as OOD

                            'setting2': {'class_selections' : {'classes_ID': ['Pleural Effusion'], 'classes_OOD': [], 'atleast_one_positive_class': False, 'replace_values_dict':{}},
                                           'demographic_selections' : {},
                                           'dataset_selections': {'no_support_device_selection':['remove all images with support device'],'seperate_patient_IDs': True},
                                           'train_val_test_split_criteria': {'k_fold_split': True, 'k': 5, 'fold': 0}},
                            #Setting 2 trains a classifier on Pleural Effusion or not Pleural Euffsion for scans with no support devices and uses images with pacemakers as OOD.
                                           
                            'setting3': {'class_selections' : {'classes_ID': ['Pleural Effusion'], 'classes_OOD': [], 'atleast_one_positive_class': False, 'replace_values_dict':{}},
                                           'demographic_selections' : {'Sex':['Male','equal']},
                                           'dataset_selections': {'no_support_device_selection':['remove all images with support device'],'seperate_patient_IDs': True},
                                           'train_val_test_split_criteria': {'k_fold_split': True, 'k': 5, 'fold': 0}}}
                            #Setting 3 trains a classifier on Pleural Effusion or not Pleural Euffsion for male scans with no support devices and uses female scans with no support devices as OOD.


OOD_selection_settings = {'setting1': {'class_selections' : {'classes_ID': ['Fracture'], 'classes_OOD': ['Cardiomegaly','Pneumothorax']},
                                           'demographic_selections' : {},
                                           'dataset_selections': {'seperate_patient_IDs': True},
                                           'train_val_test_split_criteria': {'valSize': 0, 'testSize': 1}},
                            #Setting 1 trains a classifier on Cardiomegaly and Pneumothorax and uses the Fracture class in the dataset as OOD

                            'setting2': {'class_selections' : {'classes_ID': ['Pleural Effusion'], 'classes_OOD': [], 'atleast_one_positive_class': False, 'replace_values_dict':{np.nan:0,-1:0}},
                                           'demographic_selections' : {},
                                           'dataset_selections': {'pacemaker_selection':['remove all images without pacemaker']},
                                           'train_val_test_split_criteria': {'valSize': 0, 'testSize': 1}},
                            #Setting 2 trains a classifier on Pleural Effusion or not Pleural Euffsion for scans with no support devices and uses images with pacemakers as OOD.
                                           
                            'setting3': {'class_selections' : {'classes_ID': ['Pleural Effusion'], 'classes_OOD': [], 'atleast_one_positive_class': False, 'replace_values_dict':{}},
                                           'demographic_selections' : {'Sex':['Female','equal']},
                                           'dataset_selections': {'no_support_device_selection':['remove all images with support device']},
                                           'train_val_test_split_criteria': {'valSize': 0, 'testSize': 1}}}
                            #Setting 3 trains a classifier on Pleural Effusion or not Pleural Euffsion for male scans with no support devices and uses female scans with no support devices as OOD.


classes_ID = {
    'setting1': ['Cardiomegaly','Pneumothorax'],
    'setting2': ['Pleural Effusion','No Pleural Effusion'],
    'setting3': ['Pleural Effusion','No Pleural Effusion'],
}

classes_OOD = {
    'setting1': ['Fracture'],
    'setting2': ['Scans with pacemakers'],
    'setting3': ['Female scans with no support device'],
}

#The classes in the CheXpert dataset
classes = ('No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices')
