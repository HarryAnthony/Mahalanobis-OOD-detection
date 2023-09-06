# On the use of Mahalanobis distance for out-of-distribution detection with neural networks for medical imaging

### :newspaper: Updates

30 August 2023:
* Paper will be added after publication.


### Introduction

This repository contains code for applying Mahalanobis distance score OOD detection on a pre-trained deep neural network trained on a specific task of interest (i.e. disease classification), proposed in [1]. This respository also contains a manually annotated labels for the CheXpert dataset, labelling if an image contains a pacemaker device or no support devices, a valuable OOD benchmark for the community. This code implements a pipeline for loading the CheXpert dataset, dividing it into ID and OOD sub-sets, and applying a handful of _post-hoc_ OOD detection methods, including the Mahalanobis distance based method described in [1]. I hope this work will insire future works into OOD detection for medical image analysis. If these ideas, code or dataset helped influence your research, please cite the following paper (bibtex given at bottom of readme).

[1] : Paper will be added after publication.


### Table of Contents
* [1. Method overview](#1-method-overview)
* [2. Requirements](#2-requirements)
* [3. Usage Instructions](#3-usage-instructions)
* [4. Conclusion, citation and acknowlegments](#4-conclusion,-citation-and-acknowlegments)
* [5. License](#5-license)


### 1. Method overview
An out-of-distribution (OOD) detection method which has gained a lot of research interest is measuing the distance of a test input to the training data in the network's latent space. The distance metric used is typically *Mahalanobis distance*. Using a feature extractor $\mathcal{F}$ (which is typically a section of the DNN), the feature maps for every layer can be extracted $h(\mathbf{x}) \in \mathbb{R}^{D \times D \times M}$, where the maps have size $D \times D$ with $M$ channels. The means of these feature maps can be used to define an embedding vector $\mathbf{z}(\mathbf{x}) \in \mathbb{R}^{M} = \frac{1}{D^2} \sum_D \sum_D \mathbf{h} (\mathbf{x})$. The mean $\mathbf{\mu_y}$ and covariance matrix $\Sigma_y$ of each class in the training data $(\mathbf{x},y) \sim \mathcal{D}_{\text {train}}$.

The Mahalanobis distance $d_{\mathcal{M}_y}$ between the vector $\mathbf{z}(\mathbf{x}^\*)$ of a test data point $\mathbf{x}^\*$ and the training data of class $y$ can be calculated as a sum over M dimensions. 
```math
d_{\mathcal{M}_y}(\mathbf{x}^*) = \sum_{i=1}^M ( \mathbf{z}(\mathbf{x^*}) - \mathbf{\mu_y}) \Sigma_y^{-1}  ( \mathbf{z}(\mathbf{x^*}) - \mathbf{\mu_y})
```
The Mahalanobis score is defined as the minimum Mahalanobis distance between the test data point and the class centroids of the training data, which can be used as an OOD scoring function $\mathcal{S}$.
```math
\mathcal{S}_{\text {Mahal. Score}}(\mathbf{x}^*) = - \min_{y \in \mathcal{Y}} \{ d_{\mathcal{M}_y}(\mathbf{x}^*) \}
```
where the negative sign is used to stay consistent with the convention of having a higher scoring function for ID than OOD inputs. OOD detection can be viewed as a binary classification problem, labelling an input $\mathbf{x}$ as OOD when the scoring function $\mathcal{S}(\mathbf{x},f)$ is below a threshold $\lambda$, and ID if it is above. Such a scoring function should identify if the input is from a different distribution to $\mathcal{D}_{\text {train}}$. 
```math
G_{\lambda}(\mathbf{x})= \begin{cases}\text { OOD } & \text { if } \mathcal{S}(\mathbf{x}) \leq \lambda \\ \text { ID } & \text { if } \mathcal{S}(\mathbf{x}) > \lambda \end{cases}
```


![](figures/workflow_jpg.jpg) 

**Figure 1**: (Left) Method to extract embeddings after a network module. (Right) Mahalanobis score $d_{\mathcal{M}}$ of an input to the closest training class centroid. Figure is from [1].

This research studies the best practises for the application of Mahalanobis distance for OOD detection. Our results in [1] highlight that different OOD artefacts are optimally detected at different depths of the network, which motivates using multiple OOD detectors operating at different depths of a network. To study this further, the network was divided into sections, split by successive downsampling operations, which we refer to as branches. The Mahalanobis scores measured after every module in a branch $\ell \in L_b$, where $L_b$ is the set of modules in branch b, is added together to produce a scoring function at each branch of the network. Before the summation, each the Mahalanobis scores after every module are normalised using the means and standard deviatations of the distances of the training data (see [1] for details).  This method was given the name **Multi-Branch Mahalanobis (MBM)** and it results in several OOD detectors at different depths of the network.
```math
\mathcal{S}_{\text {MBM, branch-b}}(\mathbf{x}^*) = \sum_{\ell \in L_b} \frac{d_{\mathcal{M}}^\ell (\mathbf{x}) - \mu_b^\ell}{\sigma_b^\ell}
```

![](figures/MBM_visualisation_jpg.jpg) 
**Figure 2**: Visualisation of the branches used for the MBM method (grey). Figure is from [1].

### 2. Requirements
 
#### a) Installation requirements
The system requires the following (latest version tested):
- [Python](https://www.python.org/downloads/): Developed using Python 3 (3.9.12).
- [numpy](http://www.numpy.org/) : Package for analysing and using arrays (1.24.2).
- [scipy](http://www.scipy.org/) : Scientific packages used for image transformations (1.9.3).
- [PyTorch](https://pytorch.org/) : Library for deep learning (1.13.0).
- [Torchvision](https://pytorch.org/vision/stable/index.html#module-torchvision) : Library used for datasets, transforms, and models (0.14.0).
- [pandas](https://pandas.pydata.org/) : Data manipulation and analysis library (1.5.2).
- [Pillow](https://pillow.readthedocs.io/en/stable/): Library for image processing (9.2.0).
- [Scikit-image](https://scikit-image.org/): Library for image processing and adaptation (0.19.3).

The project can be cloned using
```
$ git clone https://github.com/HarryAnthony/private_mahal_score/
```

#### b) Data requirements
This research was completed on the CheXpert dataset [2], a multi-label dataset of chest x-rays. Therefore, to run the example code please download the `CheXpert-v1.0-small` dataset and place in the folder `dataset`. The default settings used for th datasets are described in the `config/chexpert.py` file.



### 3. Usage instructions

#### a) Training models
Training models using the settings that were used for our project can be achied using the following code:
```
python3 training.py [-h] [--setting SETTING] [--lr LR] [--net_type NET_TYPE] [--depth DEPTH] [--widen_factor WIDEN_FACTOR] [--dropout DROPOUT] [--act_func_dropout ACT_FUNC_DROPOUT]
                   [--cuda_device CUDA_DEVICE] [--seed SEED] [--dataset_seed DATASET_SEED] [--batch_size BATCH_SIZE] [--dataset DATASET] [--allow_repeats ALLOW_REPEATS] [--verbose VERBOSE]
                   [--Optimizer OPTIMIZER] [--Scheduler SCHEDULER] [--save_model SAVE_MODEL] [--max_lr MAX_LR] [--act_func ACT_FUNC] [--class_selections CLASS_SELECTIONS]
                   [--demographic_selections DEMOGRAPHIC_SELECTIONS] [--dataset_selections DATASET_SELECTIONS] [--train_val_test_split_criteria TRAIN_VAL_TEST_SPLIT_CRITERIA] [--fold FOLD]
                   [--label_smoothing LABEL_SMOOTHING]
```
The arguments of the file allow for strong autonomy in controlling the how the model is trained. For training models on the CheXpert dataset, there are pre-made settings which can be used with the ` --setting` argument:
* `setting1`: Train a model to classify between Positive Cardiomegaly and Positve Pnuemothorax X-ray scans, with patient IDs kept seperate in test:val:test sets.
* `setting2`: Train a model to classify between Positive Pleural Effusion and Negative Pleural Effision X-ray scans of images, with no visible support devices (see sec 3.b) and patient IDs kept seperate in test:val:test sets.
* `setting3`: Train a model to classify between Positive Pleural Effusion and Negative Pleural Effision X-ray scans of images of male only scans, with no visible support devices (see sec 3.b) and patient IDs kept seperate in test:val:test sets.

If the ` --setting` --setting argument is not one of the above, then the arguments will be used to select the data to train the model. The configurations for each dataset are given in the `config/` directory. The configurations of each of the models that are trained using this file are saved to the file `checkpoint/Model_list.csv`.


#### b) Accessing manual annotations for CheXpert
A significant contribution of this paper was manually annotation of the lateral X-ray scans of the CheXpert dataset into four categories, given by four textfiles: 
* `pacemaker.txt` : X-ray scans with a visible pacemaker device.
* `no_support_device.txt` : X-ray scans with a visible support devices, using the definition for support devices given by CheXpert [2].
* `support_devices.txt` : X-ray scans with a visible support device, but not including a visible pacemaker device.
* `uncertain.txt` : Low-quality X-ray scans in which it is difficult to dissern which of the above categories the image belongs.
These manual annotations were done because CheXpert's annotations are *sub-optimal*. These files contain the `Path` to the image, which means selections on the dataset can be used in the following way:
```
pacemaker_list = np.loadtxt("pacemaker.txt",dtype=str)
pacemaker_list = ['CheXpert-v1.0-small/'+str(element) for element in pacemaker_list ]
pacemaker_data =  dataset['Path'].isin(pacemaker_list)]
```
I hope that this will become a useful baseline  for OOD detection (for example, training a model on images with no support devices, and using the pacemaker dataset as an OOD test set). If you use these datasets in your research, please cite this work.



#### c) Creating synthetic artefacts
This repository contains a collection of classes (`Image_augmentations.py`) which enable the creation of synthetic artefacts to images. This tool is designed to integrate into the torchvision transforms library, making it easy to augment your image datasets with synthetic artifacts. These classes can be used to generate synthetic artefacts of various shapes and textures:



![](figures/Image_augmentations_jpg.jpg) 

**Figure 3**: Visualisation of the different shapes and textures  of synthetic artefactswhich can be created with `Image_augmentations.py`.



These can be integrated with `torchvision.transforms` the following way:
```
import torchvision.transforms as T
transformations = T.Compose([T.Resize((224,224)),
            T.ToTensor(),
            T.RandomErasing_square(p=1) #Square synthetic artefact added to every image.
            lambda x: x.expand(3,-1,-1)])})
```
It can also be used in conjunction with the function `modify_transforms` which enables the addition of synthetic artefact transformations to the list of transformations for a dataloader without the need for redefinition. I hope this becomes a useful tool for studying how neural networks interact with different OOD artefacts, as a means of improving OOD detection methods.


#### d) Applying OOD detection
Given the seed of the experiment, saved in the file `checkpoint/Model_list.csv`, OOD detection methods can be applied using the file:
```
python 3 main.py [-h] [--method METHOD] [--cuda_device CUDA_DEVICE] [--batch_size BATCH_SIZE] [--verbose VERBOSE] [--seed SEED] [--ood_class_selections OOD_CLASS_SELECTIONS]
               [--ood_demographic_selections OOD_DEMOGRAPHIC_SELECTIONS] [--ood_dataset_selections OOD_DATASET_SELECTIONS] [--ood_train_val_test_split_criteria OOD_TRAIN_VAL_TEST_SPLIT_CRITERIA]
               [--ood_type OOD_TYPE] [--ood_dataset OOD_DATASET] [--filename FILENAME] [--temperature TEMPERATURE] [--noiseMagnitude NOISEMAGNITUDE]
               [--MCDP_samples MCDP_SAMPLES] [--deep_ensemble_seed_list DEEP_ENSEMBLE_SEED_LIST] [--save_results SAVE_RESULTS] [--plot_metric PLOT_METRIC] [--return_metrics RETURN_METRICS]
               [--evaluate_ID_accuracy EVALUATE_ID_ACCURACY] [--evaluate_OOD_accuracy EVALUATE_OOD_ACCURACY] [--mahalanobis_layer MAHALANOBIS_LAYER]
```
The argument `--ood_type` can be one of the following:
* `different_class` : use an unseen class from the same dataset as the OOD test cases. In conjunction with sec. 3.a, there are three pre-made settings to use: 
	* `setting1`: Use unseen X-ray scans containing a fracture as OOD test cases.
	* `setting2`: Use X-ray scans only containing pacemaker devices as OOD test cases.
	* `setting3`: Use X-ray scans with no support devices as OOD test cases.
	If the setting is not one of the above, then the OOD dataset selections will be made using the arguments.
* `synthetic` : adds a synthetic artefact using `Image_augmentations.py` to the ID test dataset to be the OOD test cases.
* `different_database` : use in conjunction with `ood_dataset`  to use another dataset as the OOD test cases.

Once the OOD test dataset is decided, the OOD detection method that is used is chosen with the `--method` argument, the choices are: MCP, MCDP, ODIN, deep ensembles, mahalanobis and MBM. Each method will print the AUROC and AUCPR of the scoring function. To save the outputs, use the function `--save_results` which will save the socring function for the ID cases and the OOD cases as well as the AUROC and AUCPR for each score.

#### Using Mahalanobis score and MBM
Using the argument `--method Mahalanobis` will calculate the Mahalanobis score and use it as the scoring function for OOD detection. The argument `--mahalanobis_layer` can be used to control which layers of the network are used, the options are:
* A signle value will calculate the Mahalanobis score at that module of the network.
* An array of values will calculate the Mahalanobis score at multiple modules of the network.
* The string 'all' can be used to calculate the Mahalanobis score at all modules of the network.


If more than one layer is selected, the argument `--feature_combination` decides whether to keep the scoring function of each module seperate and calculate an individual AUROC and AUCPR for each layer (`--feature_combination False`) or whether to combine the distances of the layers selected to have one scoring function for OOD detection (`--feature_combination True`).
Using the arguement `--method MBM` will select the modules for each *branch* (see sec.1) and will combine them using feature combination. Note that MBM is currently only avaliable for models ResNet18 and VGG16_bn, but additional models can be added by extending the *mahalanobis_module_dict* in `methods/mahalanobis.py`.

### 4. Conclusion, citation and acknowlegments
I hope this work is useful for further understanding how neural networks behave when encountering an OOD input. If you found this work useful or have any comments, do let me know.  Please emaul me your feedback or any issues to: **harry.anthony@eng.ox.ac.uk**.

Citation will be added when paper is published.

I would like to acknowlegde the work done by Christoph Berger [3], as their project code was very helpful for my project.

[2]: Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., et al.: Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In: Proceedings of the AAAI conference on artificial intelligence. vol. 33, pp. 590–597 (2019)

[3]: Berger, Christoph, Magdalini Paschali, Ben Glocker, and Konstantinos Kamnitsas. "Confidence-based Out-of-Distribution Detection: A Comparative Study and Analysis." arXiv preprint arXiv:2107.02568 (2021)


### 5. License

MIT License

Copyright (c) 2023 Harry Anthony

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.






