# On the use of Mahalanobis distance for out-of-distribution detection with neural networks for medical imaging

### News

4 August 2023:
* Code will be released shortly.


### Introduction

This repository contains code for applying Mahalanobis distance score OOD detection on a pre-trained deep neural network trained on a specific task of interest (i.e. disease classification), proposed in [1].

This respository also contains a manually annotated labels for the CheXpert dataset, labelling if an image contains a pacemaker device or no support devices, a valuable OOD benchmark for the community.

This code implements a pipeline for loading the CheXpert dataset, dividing it into ID and OOD sub-sets, and applying a handful of _post-hoc_ OOD detection methods, including the Mahalanobis distance based method described in [1].

I hope this work will insire future works into OOD detection for medical image analysis. If these ideas, code or dataset helped influence your research, please cite the following paper (bibtex given at bottom of readme).

[1] : Paper will be added after publication.


### Table of Contents
* [1. Method overview](#1-method-overview)

### 1. Method overview
An out-of-distribution (OOD) detection method which has gained a lot of research interest is measuing the distance of a test input to the training data in the network's latent space. The distance metric used is typically Mahalanobis distance. Using a feature extractor $\mathcal{F}$ (which is typically a section of the DNN), the feature maps for every layer can be extracted $h(\mathbf{x}) \in \mathbb{R}^{D \times D \times M}$, where the maps have size $D \times D$ with $M$ channels. The means of these feature maps can be used to define an embedding vector $\mathbf{z}(\mathbf{x}) \in \mathbb{R}^{M} = \frac{1}{D^2} \sum_D \sum_D \mathbf{h} (\mathbf{x})$. The mean $\mathbf{\mu_y}$ and covariance matrix $\Sigma_y$ of each class in the training data $(\mathbf{x},y) \sim \mathcal{D}_{\text {train}}$.

The Mahalanobis distance between the vector $\mathbf{z}(\mathbf{x})$ of a test data point $\mathbf{x}$ and the training data of class $y$ can be calculated as a sum over M dimensions. The Mahalanobis score $d_{\mathcal{M}}$ is defined as the minimum Mahalanobis distance between the test data point and the class centroids of the training data,
```math
d_{\mathcal{M}_y}(\mathbf{x}^*) = \sum_{i=1}^M ( \mathbf{z}(\mathbf{x^*}) - \mathbf{\mu_y}) \Sigma_y^{-1}  ( \mathbf{z}(\mathbf{x^*}) - \mathbf{\mu_y})
```
