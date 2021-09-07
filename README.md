# Interpreting Raman spectroscopy towards diagnostic purposes : an explainable deep-learning based approach

Based on Raman spectra data, this project is part of a larger project to investigate new, less intrusive and faster 
diagnostic methods. The purpose of this part is to suggest an interpretation of the CNN learning mechanism proposed in 
[this paper](https://www.nature.com/articles/s41598-021-84565-3). This research hopes to implement a method in 
order to allow an intuitive understanding and further application.

In order to achieve this objective, this project investigates two different approaches, the second being a variant of 
the first:
1. Class Activation Mapping - CAM (the original method was defined in [this paper](https://arxiv.org/abs/1512.04150) and later
applied to spectral data in [this work](https://www.sciencedirect.com/science/article/abs/pii/S0003267020303767))
2. Gradient Class Activation Mapping - Grad-CAM (this method is originally proposed [here](https://arxiv.org/abs/1610.02391))

## Technical requirements 
This code was originally executed with 4 GPUs, on the [Future SOC lab](https://hpi.de/forschung/future-soc-lab.html) 
server. But, theorically, it can be run on different configuration using the librairies and the environment specified in
the folder **server-configs**.

## Organisation of the repository
This repository can be divided into 7 parts:
1. Scientific reports
2. Technical set-up
3. Transformation of the original model using the Pytorch library
4. Bacterial study case
5. Implementation of CAM method
6. Implementation of Grad-CAM method
7. Visualisation of the different results obtained

### Scientific reports :
All files written during this project are grouped in the **report** folder. There are :
- *Presentation_PDF_version* : it is the presentation of this work in PDF.
- *Presentation_PowerPoint_version* : it is the original presentation of this work.
- *Project_synthesis* : it the synthesis of this project (5 pages).
- *Report_complete* : it is the complete report of this work.
- *Report_short_version* : the report of this work in shorter.

### Technical set-up :
All the files needed to set up the environment required to run this work are contained in the **server-configs** folder.
This folder contains several shell scripts :
- *create_docker_container* : script to automate the creation of the specified docker container.
- *install_ssh* : script created to able ssh into the docker container.
- *update_environment* : script to automate the update of the conda environment in the docker container.

In addition, it contains the *jupy_config* file which specifies the configuration of jupyter notebooks, and the subfolder
**pytorch_conda_config** which contains :
- *Dockerfile* : the Dockerfile used to create the container. 
- *environment* : contains the definition of the conda environment, with all the librairies needed to compute this project.
- *docker_clean* : script for clean the docker environment in order to re-build a container.
- *condasetup* : script to activate the conda environment needed.

Moreover, to run this code an implementation of the cubic spline is used. It can be found in the folder **torchcubicspline**
and its taken from this [GitHub repository](https://github.com/patrick-kidger/torchcubicspline).

### Pytorch transformation :
The main code of this transformation is contained in **scripts** folder. It is divided in two main parts :
1. **data_manager** : it contains all the file needed to manage the COVID data.
    - *Datasets* : class to create a dataset composed of multiple Raman Dataset in order to compute the Leave-One-Patient-Out
    Cross-Validation.
    - *RamanDataset* : class to define a single Raman Dataset.
    - *data_script* : contains all the functions needed to read, split and manage the COVID data.
2. **model_manager** : it contains all the file needed to create and manage the CNN used in this project.
    - *ConvNet* : class to define the CNN in Pytorch.
    - *model_script* : contains all the functions needed to create, parallelize, train and test the model. 

This transformation is prepared and tested in several Jupyter notebooks available in the **Model_Pytorch** folder :
1. *Raman_model_Pytorch* : This is the original notebook in which the code condensed in the **scripts** folder is developed. 
2. *Model_with_PD_Dataset* : This notebook aims to test the developed model on Parkinson's data to confirm the results.
3. *New_test_set* : In this notebook, the idea is to separate a patient before performing the Leavo-One-Patient-Out 
Cross-Validation method in order to simulate a completely new patient for all models. 

All models created during this phase are not available pre-trained in this repository for size limitationr reason, 
but can be obtained on reasonalble request at <singlan.nina@gmail.com>.

### Bacterial study case :
The data and model used for the bacteria study case are taken from this [work](https://github.com/csho33/bacteria-ID),
and is available into the **bacteria** folder.

### CAM :
The Class Activation Mapping is implemented and tested into the **cam** folder. The files concerning the implementation
are :
- *importance* : which contains the code in order to obtain the importance for an input spectra.
- *CAM* : which contains the code needed to compute the Class Activation Mapping method on an entire dataset.
- *CAM_on_bacteria* : contains the computation of the method on the bacterial case study.
- *CAM_on_covid* : contains the computation of the method on the COVID case study.

As explained in the scientific reports, the Class Activation Mapping method need a specific architecture which is created
and trained for both case study, thus, the CAM model for the bacterial dataset is in the **bacteria** folder. Then, it is
defined in *resnetCAM*, trained and fine-tuned in *CAM_Model* and finally saved in *CAMModel* and *CAMModelFine-Tuned*.
And, for the COVID case, the model is defined in the file *ConvNetCAM* in the **scripts/model_manager** folder. Then, is
trained in the notebook *CAM_Model* located into the **Model_Pytorch** folder. Finally, each model generated by the 
Leave-One-Patient-Out Cross-Validation is saved into the **saved_models/CAM_Covid** folder.

### Grad-CAM :
The Gradient Class Activation Mapping is implemented and tested into the **cam** folder. The files concerning the
implementation are : 
- *Grad-CAM* : which contains the code in order to get the Gradient and the Feature Maps for a model.
- *compute_Grad_CAM* : which contains the code needed to compute the Grad-CAM over a entire dataset.
- *Grad-CAM_on_bacteria* : contains the computation of the method on the bacterial case study.
- *Grad-CAM_on_Covid* : contains the computation of the method over the COVID dataset.

### Visualisation :
Visualisation results are not available in this repository for size limitationr reason, but can be obtained on 
reasonalble request at <singlan.nina@gmail.com>. 

It contains differents folders : 
1. **CAM_Results** : in which all the results produced by the Class Activation Mapping can be found. 
    - **Bacteria** : contains all the results produced on the bacterial case.
        - Each folder named by an integer contains the results obtained on the spectra belonging to the class identified 
        by this integer. 
            - **Graphs** : contains the visualisation of the results.
            - **Values** : contains text files in which the values visualised in the **Graphs** folder are listed.
        - **Statistics** : contains the statistics obtained.
            - **BarChart** : contains the visualisation of the statistics.
            - **Values** : contains text files in which the values visualised in the **BarChart** folder are listed.
    - **COVID** : contains all the results produced on the covid case.
        - Each folder named by "model *i*" where *i* is an integer contains the results obtained with the model trained 
        on the *i*-th fold of the Leave-One-Patient-Out Cross-Validation.
            - Folder named by an integer contains the results obtained on the spectra belonging to the class identified
            by this integer.
                - **Graphs** : contains the visualisation of the results.
                - **Values** : contains text files in which the values visualised in the **Graphs** folder are listed.
            - **Statistics** : contains the statistics obtained.
                - **BarChart** : contains the visualisation of the statistics.
                - **Values** : contains text files in which the values visualised in the **BarChart** folder are listed.
2. **Grad-CAM_Results** :
    - **Bacteria** : contains all the results produced on the bacterial case.
        - Each folder named by an integer contains the results obtained on the spectra belonging to the class identified 
        by this integer. 
            - **Graphs** : contains the visualisation of the results.
            - **Values** : contains text files in which the values visualised in the **Graphs** folder are listed.
        - **Statistics** : contains the statistics obtained.
            - **BarChart** : contains the visualisation of the statistics.
            - **Values** : contains text files in which the values visualised in the **BarChart** folder are listed.
    - **COVID** : contains all the results produced on the covid case.
        - Each folder named by "model *i*" where *i* is an integer contains the results obtained with the model trained 
        on the *i*-th fold of the Leave-One-Patient-Out Cross-Validation.
            - Folder named by an integer contains the results obtained on the spectra belonging to the class identified
            by this integer.
                - **Graphs** : contains the visualisation of the results.
                - **Values** : contains text files in which the values visualised in the **Graphs** folder are listed.
            - **Statistics** : contains the statistics obtained.
                - **BarChart** : contains the visualisation of the statistics.
                - **Values** : contains text files in which the values visualised in the **BarChart** folder are listed.
3. **train_visualization** : contains the visualisation of the training phase for each model created by the 
Leave-One-Patient-Out Cross-Validation.

In addition, in the folder **cam**, the two files named *visualize_results* and *run_visualization* are dedicated to the
calculation of statistical results.