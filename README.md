# Nuclei annotation
performing nuclei annotation (manually or semi-automatically) using ImageJ

<p align="center">
  <img src="https://github.com/masih4/nuclei_annotation/blob/main/git_figures/figure_1.png" alt="Description of the image" width="80%"/>
  <br>
  <em>Figure 1: Main tools.</em>
</p>



## Table of Contents 
[Citation](#citation)

[Manual instance segmentation with ImageJ](#manual-instance-segmentation-with-imagej)

[Manual instance segmentation and classification with ImageJ](#manual-instance-segmentation-and-classification-with-imagej)


[Semi automatic segmentation with ImageJ and AnnotatorJ (instance segmentation and classification)](#semi-automatic-segmentation-with-imagej-and-annotatorj-instance-segmentation-and-classification)

[Codes to generate instance segmentation masks (only one class)](#codes-to-generate-instance-segmentation-masks-only-one-class)

[Codes to generate instance segmentation and classification masks (more than one class)](#codes-to-generate-instance-segmentation-and-classification-masks-more-than-one-classes)





## Citation
The instructions used in this repository are adapted from these articles. If you use them, please cite the following articles:

BibTex entries:
```
@article{CryoNuSeg2021,
title = "{CryoNuSeg}: A dataset for nuclei instance segmentation of cryosectioned H\&E-stained histological images",
journal = "Computers in Biology and Medicine",
volume = "132",
pages = "104349",
year = "2021",
doi = "https://doi.org/10.1016/j.compbiomed.2021.104349",
author = "Amirreza Mahbod and Gerald Schaefer and Benjamin Bancher and Christine L\"{o}w and Georg Dorffner and Rupert Ecker and Isabella Ellinger"
}
```

```
@article{doi:10.1091/mbc.E20-02-0156,
title = "{AnnotatorJ}: an ImageJ plugin to ease hand annotation of cellular compartments",
journal = "Molecular Biology of the Cell",
volume = "31",
number = "20",
pages = "2179-2186",
year = "2020",
doi = "10.1091/mbc.E20-02-0156",
author = "Hollandi, R\'{e}ka and Di\'{o}sdi, \'{A}kos and Hollandi, G\'{a}bor and Moshkov, Nikita and Horv\'{a}th, P\'{e}ter"
}
```

```
@article{Rueden2017,
title={ImageJ2: ImageJ for the next generation of scientific image data},
journal={BMC Bioinformatics},
volume={18},
number={1},
pages={529},
year={2017},
doi={10.1186/s12859-017-1934-z},
author={Rueden, Curtis T.
and Schindelin, Johannes
and Hiner, Mark C.
and DeZonia, Barry E.
and Walter, Alison E.
and Arena, Ellen T.
and Eliceiri, Kevin W.}
}
```

## Manual instance segmentation with ImageJ
We used ImageJ software to perform manual nuclei instance segmentation. We followed these steps to annotate the images manually:
- Download and open the software (ImageJ-win64.exe), download from Fiji: https://imagej.net/software/fiji/downloads
- Update the software for the first time (help --> update)
- Open an image with the software
- From tabs:  Analyse --> Tools --> ROI manager. Ensure that both "show all" and "labels" are activated in the ROI manager. 
- Zoom in/out to have a clear view of the image and all instances
- From the selection options, select "freehand selection"
- Manually annotate the border for each object and press "T". To remove an object select the labeled number inside the object and then press "Delete" to remove an ROI 
- When you are done with all nuclei, save the outputs with ROI manager--> More --> Save
- A zip file containing all ROI files will be created after saving the outputs (each ROI file represents one of the nuclei)
- The created zip file can be later processed with Matlab or Python (create labelled masks, binary masks, etc.)
- ROI files are essential for further analysis, but the overlay images in ImageJ can also be saved for visualization:<br>
  1- (this step is optional) Select all ROI --> properties --> Fill colour (e.g, yellow) --> apply for all <br>
  2- in ImageJ --> image --> overlay --> flatten --> save image <br>

<p align="center">
  <img src="https://github.com/masih4/nuclei_annotation/blob/main/git_figures/figure%202.jpg" alt="Description of the image" width="50%"/>
  <br>
  <em>Figure 2: Example of manual instance segmentation with ImageJ.</em>
</p>



  

## Manual instance segmentation and classification with ImageJ
Consider that we have two classes. The following instructions can be easily extended in case of more nuclei classes.
- Download and open the software (ImageJ-win64.exe), download from Fiji: https://imagej.net/software/fiji/downloads
- Update the software for the first time (help --> update)
- Click on ">>" to add "ROI Menu" to the main software tabs
- Open an image with the software
- From the selection options, select "freehand selection"
- From tabs:  Analyse --> Tools --> ROI manager. Ensure that both "show all" and "labels" are activated in the ROI manager.
  
  #### For class one: "ROI" in the main tools (see Figure 1) --> set default group (e.g., 0 represents yellow)
- Zoom in/out to have a clear view of the image and all instances
- Manually annotate the border for each object for the first class and press "T". To remove an object select the labeled number inside the object and then press "Delete" to remove an ROI
  
  #### For class two: "ROI" in the main tools (see Figure 1) --> set default group (e.g., 2 represents red)
- Zoom in/out to have a clear view of the image and all instances
- Manually annotate the border for each object for the second class and press "T". To remove an object select the labeled number inside the object and then press "Delete" to remove an ROI
  
- When you are done with all nuclei for both classes, then:<br>
  Select all instances   --> save the outputs with ROI manager--> More --> Save (create a zip file with instances of all classes)
  

Note: 
- The class of instance can be changed using the "properties" option
- Other numbers for each class can be selected (each number represents a color e.g., number 3 for green)

- ROI files are essential for further analysis, but the overlay images in ImageJ can also be saved for visualization:<br>
1-(this step is optional) select all ROI of class 1 --> properties --> Fill color (e.g,. yellow) --> apply for all <br>
2-(this step is optional) select all ROI of class 2 --> properties --> Fill color (e.g,. red) --> apply for all <br>
3-in ImageJ --> image --> overlay --> flatten --> save image <br>
  
<p align="center">
  <img src="https://github.com/masih4/nuclei_annotation/blob/main/git_figures/figure%203.jpg" alt="Description of the image" width="50%"/>
  <br>
  <em>Figure 3: Example of manual instance segmentation and classification with ImageJ. Two classes are represented with yellow and red colours. </em>
</p>


## Semi automatic segmentation with ImageJ and AnnotatorJ (instance segmentation and classification)
#### set up AnnotatorJ
- Full guideline: https://github.com/spreka/annotatorj/blob/master/AnnotatorJ_documentation.pdf
- Follow the instructions to add a Fiji update site to your Fiji installation: https://imagej.net/update-sites/following
- When prompted for the update site info, enter the following: URL: https://sites.imagej.net/Spreka/ and Name: AnnotatorJ
- Restart Fiji after the update
- Download "models.zip" from: https://github.com/spreka/annotatorj/releases/tag/v0.0.2-model and extract it. Then paste the "model" folder to where you have Fiji and in the "plugins" folder (the model folder should contain three files: AnnotatorJconfig.txt, model_real.json and model_real_weights.h5)
#### instructions
- Plugin --> AnnotatorJ
- Open an image and choose instance segmentation
- Activate "contour assist" and "smoothing" and make sure it uses U-net backbone (check with"...")
##### For class one: "ROI" in the main tools (see Figure 1) --> set default group (e.g., 0 represents yellow)
- Annotate semi-automatically and revise (the brushing feature will be automatically activated)
##### For class two: "ROI" in the main tools (see Figure 1) --> set default group (e.g., 2 represents red)
- Annotate semi-automatically and revise (the brushing feature will be automatically activated)

Notes:
- Press "q" to add an "crtl+del" to remove an annotation
- also look at the notes for "Manual instance segmentation and classification with ImageJ"



## Codes to generate instance segmentation masks only one class
The codes are available on the CryoNuSeg repository: https://github.com/masih4/CryoNuSeg/tree/master 


## Codes to generate instance segmentation and classification masks (more than one class)
<a href="https://github.com/masih4/nuclei_annotation/tree/main/manual_masks_generator
">codes folder</a>
### Example directory structure for two classes
```
.
├── code
|   ├── manual_masks_generator_multi_classes
│       ├── tempfolder (folder)
│       ├── example_test.m
│       ├── masks_generator.m
│       ├── ReadImageJROI.m
│       ├── ROIs2Regions.m
│       └── unetwmap.m
│
│
└── data
    ├── class_1_annotations
        ├── distance_maps (folder)
        ├── label_masks (folder)
        ├── label_masks_modify (folder)
        ├── mask_binary (folder)
        ├── mask_binary_without_border (folder)
        ├── mask_binary_without_border_erode (folder)
        ├── nuclei_border (folder)
        ├── overlay_save_path (folder)
        ├── stacked_mask (folder)
        ├── weighted_maps (folder)
        └── weighted_maps_erode (folder)

    ├── class_2_annotations
        ├── distance_maps (folder)
        ├── label_masks (folder)
        ├── label_masks_modify (folder)
        ├── mask_binary (folder)
        ├── mask_binary_without_border (folder)
        ├── mask_binary_without_border_erode (folder)
        ├── nuclei_border (folder)
        ├── overlay_save_path (folder)
        ├── stacked_mask (folder)
        ├── weighted_maps (folder)
        └── weighted_maps_erode (folder)
    ├── imageJ_zips
    ├── images
        ├── DAPI (folder)
        ├── GFP (folder)
        ├── Texas (folder)
        └── Merged (folder)
    └── stats.csv
```

### Some general notes for naming:
- Avoid using spaces in file names (e.g., tile 10 → tile_10).
- Use leading zeros for numbering (e.g., tile_1 → tile_01).
- Use lowercase letters instead of capital letters (e.g., Tile → tile).
  
### contact: amirreza.mahbod@dp-uni.ac.at

