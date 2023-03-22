click_prediction
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── final          <- contains the training and validation dataset, along with predictions for test dataset in      
    |   |                         test_dataset_with_preds.csv
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Notebooks outlining the EDA, Modelling and hyperparameter search performed for this problem.
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── parameter_tuning.py
    │   │
    │   └── preprocessing  <- Contains the Pipelines and Transformers used for Modelling 
    │       └── pipelines.py
    │       |__ processors.py  
    |
    |__Config              <- Contains all the configs used in the project
       |_ config.py

--------

Solution for the Take Home Assignemnt Shared on 20th March, 2023. 

The Notebooks containing the approach, EDA and Modelling are contained in the `click_prediction/notebooks` folder. 

