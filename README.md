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
