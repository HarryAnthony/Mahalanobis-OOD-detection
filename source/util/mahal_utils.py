import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from source.util.general_utils import print_progress, variable_use_cuda


def get_class_means(data,RMD=False):
    '''
    Calculates the mean of each class in the data.

    Parameters
    ----------
    data : list
        List of tensors containing the data for each class.
    RMD : bool, optional
        If True, the mean of all classes is calculated (required for Relative Mahalanobis Distance). The default is False.

    Returns
    -------
    class_means : tensor
        Tensor containing the mean of each class.
    total_mean : tensor
        Tensor containing the mean of all classes (only returned if RMD is True).
    '''
    for target_class in range(len(data)):
        if target_class == 0:
            class_means = torch.mean(data[target_class], 0).view(1,-1)
        else:
            class_means = torch.cat((class_means,torch.mean(data[target_class],0).view(1,-1)),0)
    if RMD: #RMD requires calculating the mean for every latent dimension regardless of class
        total_mean = torch.mean(class_means,axis=0)
        return class_means, total_mean
    return class_means


def estimate_inv_covariance(data,class_means,RMD=False,total_mean=None):
    '''
    Calculates the inverse covariance matrix for each class in the data, using the penrose-moore pseudo-inverse.

    Parameters
    ----------
    data : list
        List of tensors containing the data for each class.
    class_means : tensor
        Tensor containing the mean of each class.
    RMD : bool, optional
        If True, the inverse covariance matrix of all classes is calculated (required for Relative Mahalanobis Distance). The default is False.
    total_mean : tensor, optional
        Tensor containing the mean of all classes (only required if RMD is True). The default is None.

    Returns
    -------
    class_precisions : list
        Tensor containing the inverse covariance matrix of each class.
    total_precision : tensor
        Tensor containing the inverse covariance matrix of all classes (only returned if RMD is True).
    '''
    class_precisions = []
    for target_class in range(len(data)):
        delta = data[target_class] - class_means[target_class]
        cov_matrix = torch.matmul(delta.T, delta) / delta.size(0)

        if target_class == 0:
            class_precisions = torch.linalg.pinv(cov_matrix).view(1,-1).reshape(1,len(cov_matrix[0]),len(cov_matrix[0]))
        else:
            class_precisions = torch.cat((class_precisions, torch.linalg.pinv(cov_matrix).view(1,-1).reshape(1,len(cov_matrix[0]),len(cov_matrix[0]))), 0)

    if RMD: #RMD requires calculating the covariance matrix regardless of class
        for target_class in range(len(data)):
            delta_total_temp = data[target_class] - total_mean #Difference of data from mean of classes
            if target_class == 0:
                delta_total = delta_total_temp
            else:
                delta_total = torch.cat((delta_total,delta_total_temp),0)
        cov_matrix_total = torch.matmul(delta_total.T, delta_total) / delta_total.size(0)
        total_precision = torch.linalg.pinv(cov_matrix_total).view(1,-1).reshape(1,len(cov_matrix[0]),len(cov_matrix[0]))
        return class_precisions, total_precision
    return class_precisions


def organise_data(data,feature_extractor,requires_grad=False):
    '''
    Extracts feature maps from the data using the feature extractor, and uses the means of the feature maps to define a vector for the data.

    Parameters
    ----------
    data : list
        List of tensors containing the data for each class.
    feature_extractor : torch.nn.Module
        Feature extractor network.

    Returns
    -------
    vector : list
        Means of the feature maps.
    '''
    if requires_grad == False:
        with torch.no_grad():
            out_features = feature_extractor(data)
    else:
        out_features = feature_extractor(data)
    
    out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
    return torch.mean(out_features, 2)


def perturb_x_Mahalanobis(inputs,mahal_score,criterion,noiseMagnitude):
        """
        Perturb the inputs x to improve the confidence in the prediction.

        Parameters
        ----------
        inputs : torch.Tensor
            The input data.
        mahal_score : torch.Tensor
            The minimum Mahalanobis distance between the data and the mean of each class.
        criterion : torch.nn.Module
            The loss function.
        noiseMagnitude : float
            The magnitude of the noise to be added to the data.

        Returns
        -------
        inputs : torch.Tensor
            The perturbed data.              
        """
        loss = criterion(mahal_score, torch.zeros(mahal_score.size()).cuda())
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        return inputs - gradient*noiseMagnitude 


def calc_gaussian_score(out_features,training_data_statistics_module,num_classes,RMD=False,use_cuda=False):
    '''
    Calculates the Mahalanobis distance between the data and the mean of each class.

    parameters:
    ----------------
    out_features: torch.Tensor (shape (batch_size, num_features))
        The test data to be classified.
    sample_mean: torch.Tensor (shape (num_classes, num_features))
        The mean of each class of the training data
    precision: torch.Tensor (shape (num_classes, num_features, num_features))
        The precision matrix of each class og the training data.
    num_classes: int
        The number of classes in the training data.
    RMD: bool, optional
       If true, the Relative Mahalanobis Distance is used.
    mu_total: torch.Tensor (shape (num_features)), optional
        The mean of the total training data (required for RMD).
    precision_total: torch.Tensor (shape (num_features, num_features)), optional
        The precision matrix of the total training data.
    use_cuda: bool, optional
        Whether to use cuda. The default is True.
    myfile: file, optional
        The file to write the results to.

    returns:
    ----------------
    gaussian_score: torch.Tensor (shape (batch_size, num_classes))
        The Mahalanobis distance between the data and the mean of each class.
    '''
    gaussian_score = 0
    for i in range(num_classes):
        class_mean = training_data_statistics_module['mean_list'][i] #Get sample mean
        delta = out_features - class_mean #Subtract mean from data

        if RMD:
            delta_total = out_features - training_data_statistics_module['total_mean_list']
            delta_total = delta_total.view(delta.size(0),-1)
            term_gau = torch.matmul(torch.matmul(delta, training_data_statistics_module['precision_list'][i]),delta.T).diag() - torch.matmul(torch.matmul(delta_total, training_data_statistics_module['total_precision_list'][0]),delta_total.T).diag()
        else:
            term_gau = torch.matmul(torch.matmul(delta, training_data_statistics_module['precision_list'][i]),delta.T).diag()

        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
        
    return gaussian_score


def calc_mahalanobis_score(inputs, feature_extractor, modules, num_classes, training_data_statistics, magnitude=0,preprocess=False,RMD=False,use_cuda=True,**kwargs):
    """
    Calculate the Mahalanobis distance to the closest class centroid in the training data between the data and the mean of each class.

    Parameters
    ----------
    inputs : torch.Tensor
        The input data.
    feature_extractor : torch.nn.Module
        Feature extractor network.
    modules : list
        List of names of the modules to be evaluated.
    num_classes : int
        Number of classes in the dataset.
    training_data_statistics : dict
        A dictionary containing the following for each module in module_names:
            sample_mean : list (shape = [num_classes, latent_dim])
                List of sample mean for each class.
            precision : list (shape = [num_classes, latent_dim, latent_dim])
                List of precision (inverse of covariance matrix) for each class.
            mu_total : list (shape = [latent_dim])
                List of mean of the data for each class (only returned if RMD is True).
            precision_total : list (shape = [latent_dim, latent_dim])
                List of covariance matrix of the data for each class (only returned if RMD is True).
    magnitude : float, optional
        The magnitude of the perturbation to be added to the data for FGSM. Default is 0.
    preprocess : bool, optional
        Whether to apply FGSM on the data to increase the confidence of the Mahalanobis score. Default is True.
    RMD : bool, optional
        Whether to calculate the relative Mahalanobis distance (RMD) instead of Mahalanobis distance. Default is False.
    use_cuda : bool, optional
        Whether to use GPU or not. Default is True.
        
    Returns
    -------
    score_list : list
        List of the Mahalanobis distance between the data and the mean of each class for each module in modules.
    """
    
    inputs = Variable(inputs,requires_grad=True)
    if preprocess == True:
        requires_grad = True
    else:
        requires_grad = False
    score_list = []

    inputs = variable_use_cuda(inputs,use_cuda)

    if requires_grad == False:
        with torch.no_grad():
            out_features = feature_extractor(inputs)
    else:
        inputs = Variable(inputs,requires_grad=True)
        out_features = feature_extractor(inputs)

    training_data_statistics_module = {}
    
    for module_index, module in enumerate(modules):

        out_features[module] = out_features[module].view(out_features[module].size(0), out_features[module].size(1), -1)
        out_features[module] = torch.mean(out_features[module], 2)

        for statistic in training_data_statistics:
            training_data_statistics_module[statistic] = training_data_statistics[statistic][module_index]

        gaussian_score = calc_gaussian_score(out_features[module],training_data_statistics_module, num_classes, RMD=RMD, use_cuda=use_cuda)
        score_list.append(torch.min(gaussian_score,1).values)

    #Apply FGSM to increase the confidence of the Mahalanobis score
    if preprocess:
        score_sum = torch.sum(torch.stack(score_list),0)
        criterion = nn.MSELoss() #Use MSE loss to backpropogate with
        inputs_perturbed = perturb_x_Mahalanobis(inputs,score_sum,criterion,magnitude)
        del inputs, score_sum
        preprocess = False #Preprocessing has been applied, so no need to apply it again
        score_list = calc_mahalanobis_score(inputs_perturbed,feature_extractor,modules,num_classes,
                                                          training_data_statistics,preprocess=False,
                                                          RMD=RMD,use_cuda=use_cuda,**kwargs)
        

    return score_list
    


def estimate_mean_precision(module_names, feature_extractor, trainloader=None, num_classes=2,use_cuda=True,RMD=False,verbose=True,**kwargs):
    """
    Compute sample mean and precision (inverse of covariance matrix) for each class in the training data.

    Parameters
    ----------
    module_names : list
        List of names of the modules to be evaluated.
    feature_extractor : torch.nn.Module
        The feature extractor network.
    trainloader : torch.utils.data.DataLoader
        The training data loader.
    num_classes : int
        Number of classes in the dataset.
    use_cuda : bool, optional
        Whether to use GPU or not. Default is True.
    RMD : bool, optional
        Whether to calculate the mean of the data and covariance matrix of the data required for Relative Mahalanobis Distance or not. Default is False.
    verbose : bool, optional
        Whether to print the progress or not. Default is False.

    Returns
    -------
    training_data_statistics : dict
        A dictionary containing the following for each module in module_names:
            sample_mean : list (shape = [num_classes, latent_dim])
                List of sample mean for each class.
            precision : list (shape = [num_classes, latent_dim, latent_dim])
                List of precision (inverse of covariance matrix) for each class.
            mu_total : list (shape = [latent_dim])
                List of mean of the data for each class (only returned if RMD is True).
            precision_total : list (shape = [latent_dim, latent_dim])
                List of covariance matrix of the data for each class (only returned if RMD is True).
    """
    num_sample_per_class = [np.zeros(num_classes) for _ in range(len(module_names))]
    latent_dim_data = [[0 for _ in range(num_classes)] for _ in range(len(module_names))] #Stores the embedding vector for each class

    #Extract the embedding vectors for each image from the required modules
    print("Computing the sample means and precisions for each class in the training data")
    l = len(trainloader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
    for batch_idx, (data, target) in enumerate(trainloader): #Iterate through each image in training data
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        data = variable_use_cuda(data,use_cuda)
        batch_size = data.size(0)
        with torch.no_grad():
            out_features = feature_extractor(data) #Get latent dimension data
        for module_count, module in enumerate(module_names):
            out_features[module] = out_features[module].view(out_features[module].size(0), out_features[module].size(1), -1) 
            out_features[module] = torch.mean(out_features[module].data, 2)
            # construct the sample matrix of embedding vectors for each class
            for i in range(batch_size): 
                if num_sample_per_class[module_count][target[i]] == 0: #If there are no samples for the class in this module
                    latent_dim_data[module_count][target[i]] = out_features[module][i].view(1, -1) 
                else:
                    latent_dim_data[module_count][target[i]] \
                            = torch.cat((latent_dim_data[module_count][target[i]], out_features[module][i].view(1, -1)), 0)
                num_sample_per_class[module_count][target[i]] += 1
        del out_features

    #Calculate the sample mean and precision for each class in the training data
    training_data_statistics = {}

    statistics_to_measure = ['mean_list', 'precision_list'] if not RMD else ['mean_list', 'precision_list', 'total_mean_list', 'total_precision_list']
    for statistic in statistics_to_measure:
        training_data_statistics[statistic] = [[] for _ in range(len(module_names))]

    for module_count, module in enumerate(module_names):
        if RMD:
            class_means, total_mean = get_class_means(latent_dim_data[module_count],RMD=True)
            class_precisions, total_precision = estimate_inv_covariance(latent_dim_data[module_count],class_means,RMD=True,total_mean=total_mean)
            training_data_statistics['total_mean_list'][module_count] = total_mean
            training_data_statistics['total_precision_list'][module_count] = total_precision
        else:
            class_means = get_class_means(latent_dim_data[module_count])
            class_precisions = estimate_inv_covariance(latent_dim_data[module_count],class_means)
        training_data_statistics['mean_list'][module_count] = class_means
        training_data_statistics['precision_list'][module_count] = class_precisions

    return training_data_statistics