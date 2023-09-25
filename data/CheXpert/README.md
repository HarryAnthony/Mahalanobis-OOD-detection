#### Data requirements
Please download the [CheXpert-v1.0-small](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset and place it in this directory. 


#### Manual annotations for CheXpert
In this research, we manually annotated ~50% of the frontal X-rays in the CheXpert dataset (labels for 100% coming soon!) into four categories: *no support device*, *pacemaker*, *other support device* (i.e. no pacemaker), *uncertain*. The aim was to create a clean and reliable OOD evaluation benchmark for medical imaging: The *no support device* class is a set of images that can be used as In-Distribution set for training a model, while the images of *pacemaker* class, which contain a visually-distinct image pattern, can be used as the OOD test set - enabling analyses of the performance of OOD detection methods on a real OOD artefact. As a contribution of this work, we make these annotations publicly available here, and hope they will be useful for assessment of OOD methods in future works by the community. Please cite this work if you use this data in your research.

The annotations are given by the following textfiles: 
* `pacemaker.txt` : X-ray scans with a visible pacemaker device.
* `other_support_device.txt` : X-ray scans with a visible support device (lines, PICC, tube, valve, catheter, hardware, arthroplast, plate, screw, cannula, coil, mediport) visibly obscuring the chest, but not including a visible pacemaker device.
* `no_support_device.txt` : X-ray scans without any visible support device (nor pacemaker).
* `uncertain.txt` : Low-quality X-ray scans in which it is difficult to dissern which of the above categories the image belongs.

These files contain the `Path` to the image, which means selections on the dataset can be used in the following way:
```
pacemaker_list = np.loadtxt("pacemaker.txt",dtype=str)
pacemaker_list = ['CheXpert-v1.0-small/'+str(element) for element in pacemaker_list ]
pacemaker_data =  dataset['Path'].isin(pacemaker_list)]
```

Creating this set of annotations was necessary to create a reliable OOD evaluation because we found the original class *support devices* of CheXpert contained some label noise (as it's made by an NLP model) and contained a heterogeneous set of devices (as opposed to our *pacemaker* class), which complicated analysis of OOD patterns. If you find any issues, please let us know so they can be addressed.

**DISCLAIMER**: These annotations were made by author Harry Anthony (PhD candidate in Engineering Science) based on visual inspection, and were **not validated by medical experts**. This data is for **research purposes only**.


![](../../figures/summary_of_manual_annotations_jpg.jpg)
**Figure 3**: Visualisation of the manual annotations made for ~50% of the frontal scans for the CheXpert dataset, giving examples for the four different labels used for our annotations. These annotations are available in the _data_ directory.


