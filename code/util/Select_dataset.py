import pandas as pd
from util.processing_data_utils import balance_df, rows_not_in_df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

class Dataset_selection_methods():
    """
    A class for selecting the dataset to use for training, validation and testing.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the dataset.
    cf : str
        The configuration file for the dataset.
    mode : str, optional
        The mode to use the dataset in, can be 'train' or 'test'. For 'train', the validation and training datasets are outputted. 
        For 'test', the test datasets are outputted. The default is 'train'.
    verbose : bool, optional
        Whether to print details relating to the dataset selections. The default is False.
    """
    def __init__(self, data, cf, mode='train',verbose=False):
        self.data = data
        self.cf = cf
        self.mode=mode
        self.verbose = verbose
        self.dataset = {}


    def apply_selections(self,class_selections={'classes_ID':['All'],'classes_OOD':[]},demographic_selections={},dataset_selections={},train_val_test_split_criteria={}):
        """
        Applies the selections to the dataset, and outputs a dictionary containing the training, validation and test datasets.

        Parameters
        ----------
        class_selections : dict, optional
            A dictionary of the class selections to apply on the data. The key is the type of selection and the value is a list of the selections to make.
            The dictionary should include a 'classes_ID' key (classes to include in the training dataset) and a 'classes_OOD' key (classes to exclude in the training dataset). 
            The selections can be 'classes_ID', 'classes_OOD', 'atleast_one_positive_class', 'allow_multiple_positive_classes' or 'replace_values_dict'.
            The default is {'classes_ID':['All'],'classes_OOD':[]}.
        demographic_selections : dict, optional
            A dictionary of the demographic criteria to selects, Selects rows from the dataframe based on the demographic criteria given in demographic_selections.
            This is for making selections not related to classes (such as age, gender, race, etc). The key is the column name and the value is a list of form [criteria,operation].
            The operation can be 'above', 'below', 'not equal' or 'equal'. An example is {'Age':[40,'above']}, which would select all rows with an age above 40.
        dataset_selections : dict, optional
            A dictionary of the dataset specific selections selections to make. The key is the name of the selection and the value is a list of criteria.
        train_val_test_split_criteria: dict, optional
            A dictionary of the train/validation/test split criteria for splitting the dataset. The key is the split criteria and the value is the crtieria's threshold.
            The selections can be 'valSize' (float), 'testSize' (float), 'balance_classes_in_val_test' (bool) or 'dataset_seed' (int) for a single split, or 
            the selections can be 'k_fold_split' (bool), 'fold' (int), k (int), dataset_seed (int) for k-fold cross validation. The default is {}.

        Returns
        -------
        dict
            A dictionary containing the training, validation and test sets.
        """
        self.dataset['total_df'] = self.select_classes(**class_selections)
        self.select_demographic_criteria(demographic_selections=demographic_selections)
        self.dataset = self.cf.database_specific_selections(self.dataset,selections=dataset_selections,**train_val_test_split_criteria)
        self.split_train_test_val(**train_val_test_split_criteria)

        self.dataset.pop('total_df',None)
        if self.mode == 'train':
            if 'test_df' in self.dataset.keys():
                if 'validation_df' not in self.dataset.keys():
                    self.dataset['validation_df'] = self.dataset['test_df']
                self.dataset.pop('test_df',None)
        elif self.mode == 'test':
            if 'validation_df' in self.dataset.keys():
                if 'test_df' not in self.dataset.keys():
                    self.dataset['test_df'] = self.dataset['validation_df']
                self.dataset.pop('validation_df',None)
        return self.dataset


    def select_classes(self, classes_ID, classes_OOD, atleast_one_positive_class=True,allow_multiple_positive_classes=False, replace_values_dict={np.nan: 0, -1: 0}):
        """
        Selects the classes of interest from the dataframe and assigns a class integer to each row.

        Parameters
        ----------
        classes_ID : list
            A list of the ID classes.
        classes_OOD : list
            A list of the OOD classes.
        atleast_one_positive_class : bool, optional
            If True, only rows with at least one positive class in classes_ID are kept. The default is True.
        allow_multiple_positive_classes : bool, optional
            If True, rows with more than one positive class in classes_ID are kept. The default is False.
        verbose : bool, optional
            If True, print the number of images with each class. The default is True.
        replace_values_dict : dict, optional
            A dictionary of values to replace in the dataframe. The default is {np.nan: 0, -1: 0}.

        Returns
        -------
        pd.DataFrame
            The dataframe with the classes selected and a class integer assigned to each row.
        """
        classes_ID = list(self.cf.classes) if classes_ID == ['All'] else classes_ID
        total_classes_set = classes_ID + classes_OOD # Create a set of total classes
        self.data = self.data.drop(columns=[c for c in self.cf.classes if c not in total_classes_set]) # Filter the classes we don't care about

        #Used to replace class values with 0 or 1, given by replace_values_dict
        if bool(replace_values_dict): #Checks dictionary is not empty
            self.data[total_classes_set] = self.data[total_classes_set].replace(replace_values_dict)

        # Filter rows out that contain positive OOD cases
        data_with_ood = self.data[classes_OOD].any(axis=1)
        self.data = self.data[~data_with_ood]
        self.data = self.data.drop(columns=classes_OOD) #Drop OOD columns

        #Filter out rows with unclear labels i.e. NaN values
        rows_without_unsure_labels = self.data[classes_ID].isin([0, 1]).all(axis=1)
        self.data = self.data[rows_without_unsure_labels]

        # Remove rows that don't have at least one class of interest
        if atleast_one_positive_class == True:
            data_with_positive_classe = self.data[classes_ID].any(axis=1)
            self.data = self.data[data_with_positive_classe]
        
        # Remove rows that have at more than one positive class in classes_ID if allow_multiple_classes is False
        if allow_multiple_positive_classes == False:
            rows_with_single_class = self.data[classes_ID].sum(axis=1) <= 1 
            self.data = self.data[rows_with_single_class]
        
        #Assign a class integer to each row
        self.assign_class_integer(classes_ID)        

        # Print the number of images with each class
        if self.verbose == True:
            for class_int, count in self.data.groupby('class').size().items():
                class_statement = []
                for i, col in enumerate(classes_ID):
                    positive = 'positive' if class_int & (2 ** i) else 'negative'
                    class_statement.append(f"{positive} {col}")
                print(f"There are {count} images with {' and '.join(class_statement)}")
        
        #If using one-hot encoding, convert the class integer to a one-hot vector
        if atleast_one_positive_class == True and allow_multiple_positive_classes == False:
            self.data['class'] = np.log2(self.data['class']).astype(int)
        else:
            self.reduce_classes()

        # Drop the 'classes_ID' columns
        self.data = self.data.drop(columns=classes_ID)

        return self.data
    

    def assign_class_integer(self, classes_ID):
        """
        Assigns an integer to each row of the dataframe, based on the classes_ID columns. The integer is calculated by treating the classes_ID columns
        as binary digits and converting to an integer, this is done to ensure that each class combination is unique.

        Parameters
        ----------
        classes_ID : list
            A list of the ID classes.
        """
        powers_of_2 = np.array([2 ** i for i in range(len(classes_ID))])
        class_integer = self.data[classes_ID].dot(powers_of_2)
        self.data['class'] = class_integer.astype(int)

    def reduce_classes(self):
        # Find unique values in the 'class' column
        unique_classes = self.data['class'].unique()

        # Create a mapping from unique values to their nearest integers
        class_mapping = {}
        for idx, value in enumerate(sorted(unique_classes)):
            class_mapping[value] = idx

        # Apply the mapping to the 'class' column to reduce the values to the nearest integers
        self.data['class'] = self.data['class'].map(class_mapping)


    def select_demographic_criteria(self,demographic_selections={}):
        """
        Selects rows from the dataframe based on the demographic criteria given in demographic_selections.
        This function is for making selections based on criteria not related to classes (such as age, gender, race, etc)

        Parameters
        ----------
        demographic_selections : dict, optional
            A dictionary of the demographic criteria to select. The key is the column name and the value is a list of form [criteria,operation]. 
            The operation can be 'above', 'below', 'not equal' or 'equal'. An example is {'Age':[40,'above']}, which would select all rows with an age above 40.
            Another example is {'Sex':['Male','equal']} to filter the dataset to only have male entries. The default is {}.

        """
        for selection in demographic_selections.keys():
            if selection not in self.dataset['total_df'].columns:
                raise ValueError('Demographic selection not in dataset. Columns in dataset: ',self.dataset['total_df'].columns)
            else:
                if demographic_selections[selection][1] == 'above':
                    self.dataset['total_df'] = self.dataset['total_df'][self.dataset['total_df'][selection] > demographic_selections[selection][0]]
                elif demographic_selections[selection][1] == 'below':
                    self.dataset['total_df'] = self.dataset['total_df'][self.dataset['total_df'][selection] < demographic_selections[selection][0]]
                elif demographic_selections[selection][1] == 'not equal':
                    self.dataset['total_df'] = self.dataset['total_df'][self.dataset['total_df'][selection] != demographic_selections[selection][0]]
                elif demographic_selections[selection][1] == 'equal':
                    self.dataset['total_df'] = self.dataset['total_df'][self.dataset['total_df'][selection] == demographic_selections[selection][0]]
                else:
                    raise ValueError('Demographic selection not recognised. Must be "above", "below", "not equal" or "equal".')
                
                
    def split_train_test_val(self,k_fold_split=False,**kwargs):
        """
        Splits the dataset into training, validation and test sets.

        Parameters
        ----------
        k_fold_split : bool, optional
            If True, split the dataset into k folds for k-fold cross validation. The default is False.
        """
        if 'is_split' in self.dataset and self.dataset['is_split'] == True:
            return None
        if k_fold_split == True:
            self.k_fold_cross_val(**kwargs)
        else:
            self.single_train_val_test_split(**kwargs)


    def single_train_val_test_split(self,valSize=0.1,testSize=0.1,balance_classes_in_val_test=True,dataset_seed=42):
        """
        A function for splitting the dataset into training, validation and test sets. 
        
        Parameters
        ----------
        valSize: float
            The fraction of the dataset to use for the validation set
        testSize: float
            The fraction of the dataset to use for the test set
        balance_class_val_test: bool
            Whether to balance the classes in the validation and test sets
        dataset_seed: int
            The seed to use for splitting the dataset
        """
        if isinstance(valSize,(float,int)) == False or isinstance(testSize,(float,int)) == False or valSize<0 or testSize<0 or valSize+testSize>1:
            raise ValueError('valSize and testSize must be floats between 0 and 1, and valSize+testSize must be less than 1')

        #If all dataset is to be used for training, validation or testing, then return the entire dataset
        if valSize==0 and testSize==0:
            self.dataset['train_df'] = self.dataset['total_df']
            return
        elif valSize==1 and testSize==0:
            self.dataset['validation_df'] = self.dataset['total_df']
            return
        elif valSize==0 and testSize==1:
            self.dataset['test_df'] = self.dataset['total_df']
            return

        if balance_classes_in_val_test == True:
            #Make a new dataset with balanced classes, in order to make balanced val/test dataset
            dataset_balanced = balance_df(self.dataset['total_df'],random_state=int(dataset_seed))
            #Place the entries not included to keep balanced classes into a different dataset
            dataset_remaining = rows_not_in_df(self.dataset['total_df'],dataset_balanced)
            if len(self.dataset['total_df'])/len(dataset_balanced)*(valSize+testSize) >= 1:
                print('Warning: Dataset too imbalanced to make class balanced val/test sets, so balance_classes_in_val_test=True in train_val_test_split_criteria was ignored.')
                dataset_balanced = self.dataset['total_df']
                dataset_remaining = pd.DataFrame()
        else:
            #If we don't want to balance the classes, then just use the filtered dataset
            dataset_balanced = self.dataset['total_df']
            dataset_remaining = pd.DataFrame()

        #Split the balanced dataset into the required fractions
        #Adjusts the fraction so that valSize and testSize are fractions of the total dataset
        frac_adjust = len(self.dataset['total_df'])/len(dataset_balanced)
        if min(dataset_balanced['class'].value_counts()) == 1:
            df_train_partial, df_test_full = train_test_split(dataset_balanced, test_size=(valSize+testSize)*frac_adjust, random_state=int(dataset_seed))
        else:
            df_train_partial, df_test_full = train_test_split(dataset_balanced, test_size=(valSize+testSize)*frac_adjust, stratify=dataset_balanced['class'], random_state=int(dataset_seed))
        if testSize==0: #If no test set is required, then just split the dataset into training and validation sets
            self.dataset['validation_df'] = df_test_full
        elif valSize==0: #If no validation set is required, then just split the dataset into training and test sets
            self.dataset['test_df'] = df_test_full
        else: #If both validation and test sets are required, then split the dataset into training, validation and test sets
            if min(dataset_balanced['class'].value_counts()) == 1:
                df_test, df_validation = train_test_split(df_test_full, test_size=(testSize/(valSize+testSize)), random_state=int(dataset_seed))
            else:
                df_test, df_validation = train_test_split(df_test_full, test_size=(testSize/(valSize+testSize)), stratify=df_test_full['class'], random_state=int(dataset_seed))
            self.dataset['validation_df'] = df_validation
            self.dataset['test_df'] = df_test

        #Combine the training dataset with the entries removed when balancing the classes
        self.dataset['train_df'] = pd.concat([df_train_partial,dataset_remaining])


    def k_fold_cross_val(self,fold=0,k=5,dataset_seed=42):
        """
        A function for splitting the dataset into k folds for k-fold cross validation.

        Parameters
        ----------
        fold: int
            The fold to use for validation
        k: int
            The number of folds to split the dataset into
        dataset_seed: int
            The seed to use for splitting the dataset
        """
        if fold >= k:
            raise ValueError('fold must be less than k (%d)' % k)
        if min(self.dataset['total_df']['class'].value_counts()) < k: #If there are too few classes, don't stratify
            kf = KFold(n_splits=k, shuffle=True, random_state=int(dataset_seed))
            image_folds= list(kf.split(self.dataset['total_df'])) # Split single_images into folds
        else:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(dataset_seed))
            image_folds= list(skf.split(self.dataset['total_df'], self.dataset['total_df']['class'])) # Split single_images into folds
        #Place single patient ID images into the validation and training set
        self.dataset['train_df'] = self.dataset['total_df'].iloc[image_folds[fold][0]]
        self.dataset['validation_df'] = self.dataset['total_df'].iloc[image_folds[fold][1]]




            


