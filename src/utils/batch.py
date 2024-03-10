
import os 

import numpy as np
import pandas as pd

import src.config as config
from src.text.classifiers import MLClassifier, TFbertClassifier
from src.image.classifiers import ImgClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

def fit_save_all(params_list, X_train, y_train, X_test, y_test, result_file_name = 'results.csv'):
    """ 
    The fit_save_all function is designed to automate the process of fitting multiple machine learning models, 
    evaluating their performance, and saving the results along with the models themselves. This function takes 
    a list of parameters for different models, training and testing datasets, and an optional file name for 
    saving the results. It supports multiple classifier types, including ML-based classifiers, 
    BERT-based classifiers for text, and image classifiers. Summary of the results will be appended to the 
    result.csv file if it already exists.

    * Parameters
        * params_list (list of dictionaries): Each dictionary contains the configuration for a model to be trained. 
          Keys include 'modality', 'class', 'base_name', 'vec_method', 'param_grid', and 'nfolds_grid', among others. 
          These dictionaries specify the model type, vectorization method, parameters for grid search, and the number 
          of folds for cross-validation.
        * X_train (DataFrame): The training features dataset.
        * y_train (DataFrame): The training labels dataset.
        * X_test (DataFrame): The testing features dataset.
        * y_test (DataFrame): The testing labels dataset.
        * result_file_name (str, optional): The name of the CSV file to store the results of the model training and 
          evaluation. Defaults to 'results.csv'.
    
    * Functionality
        Directory and File Preparation: The function first checks if the directory for storing results exists; 
        if not, it creates the directory. It then checks if the specified results CSV file exists within this directory; 
        if not, it creates a new CSV file with the necessary columns.

        Data Preparation for Cross-Validation: It concatenates the training and testing datasets to prepare for 
        cross-validation.

        Model Fitting and Evaluation:
        For each set of parameters in params_list, the function initializes the specified classifier, performs grid 
        search cross-validation if specified, and fits the model on the training data.
        It calculates the F1 score on the test dataset, as well as cross-validated scores on the combined dataset, if applicable.
        Result Recording:

        Saves the best parameters found (if grid search is applied), test scores, cross-validation scores, and the time 
        taken for fitting during cross-validation.
        Saves the trained model to a specified path.
        Updates the results CSV file with the new results for each model.
    
    
    * Output
        Returns the results dataframe
    
    Usage Notes
    Ensure all required libraries (os, pandas, sklearn, etc.) and configurations (e.g., config.path_to_results) 
    are properly set up before using this function.
    This function is highly flexible and supports various classifier types and parameter configurations. Users should 
    carefully prepare the params_list according to their specific requirements for model training and evaluation.
    The function assumes that the models and scoring methods (like classification_score and cross_validate) are implemented 
    and available for use.
    """
    if not os.path.exists(config.path_to_results):
        os.makedirs(config.path_to_results)
    
    results_path = os.path.join(config.path_to_results, result_file_name)
    
    #If results.csv doesn't exist, we create it
    if not os.path.isfile(results_path):
        df_results = pd.DataFrame(columns=['modality', 'class', 'vectorization', 'classifier', 'tested_params', 
                                           'best_params','score_test', 'score_cv_test', 'score_cv_train', 'fit_cv_time',
                                           'model_path'])
        df_results.to_csv(results_path)
    
    #Concatenating train and text sets for CV scores    
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)
    
    #Mandatory fields in parameters
    mandatory_fields =  ['modality', 'class', 'base_name', 'vec_method', 'param_grid']

    for params in params_list:
        #Checking mandatory fields
        for fname in mandatory_fields:
            if fname not in params.keys():
                params[fname] = np.nan
                
        #Making sure params['params_grid'] dictionnary values
        # are provided as list (important for GridSearchCV)
        for key in params['param_grid'].keys():
            if not isinstance(params['param_grid'][key], list):
                params['param_grid'][key] = [params['param_grid'][key]]
        
        #Populating results with parameters
        results = {'modality': params['modality'], 'class': params['class'], 'classifier': params['base_name'],
                   'vectorization': params['vec_method'], 'tested_params': params['param_grid']}
        
        #GridsearCV on one parameter
        print('Fitting: ', params['base_name'], params['vec_method'])
        
        #Fetching first params of list in param_grid in case no GridSearchCV
        #is requested
        clf_params = {}
        for key in params['param_grid'].keys():
            clf_params[key] = params['param_grid'][key][0]
        print(clf_params)   
        
        #Instanciating the classifier
        if params['class'] == 'MLClassifier':
            clf = MLClassifier(base_name=params['base_name'], vec_method=params['vec_method'], **clf_params)
        elif params['class'] == 'TFbertClassifier':
            clf = TFbertClassifier(base_name=params['base_name'], **clf_params)
        elif params['class'] == 'ImgClassifier':
            clf = ImgClassifier(base_name=params['base_name'], **clf_params)
        elif params['class'] == 'TFbertClassifier':
            clf = TFbertClassifier(base_name=params['base_name'], **clf_params)
        
        #paramters to feed into GridSearchCV   
        param_grid = params['param_grid']
        
        #Kfold stratification
        if params['nfolds_grid'] > 0:
            cvsplitter = StratifiedKFold(n_splits=params['nfolds_grid'], shuffle=True, random_state=123)
        else:
            cvsplitter = None
        
        #Gridsearch or fit on train set
        if cvsplitter is not None:
            gridcv = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1_weighted', cv=cvsplitter)
            gridcv.fit(X_train, y_train)
            print('GridSearch: ', gridcv.best_params_)
            
            #saving best params
            results['best_params'] = gridcv.best_params_
            
            #Keeping the best parameter
            clf = gridcv.best_estimator_
        else:
            clf.fit(X_train, y_train)
            results['best_params'] = np.nan
        
        #Calculating scores on test set
        f1score_test = clf.classification_score(X_test, y_test)
        print('Test set, f1score: ', f1score_test)
        
        #saving f1score_test
        results['score_test'] = f1score_test
        
        #Calculating score by k-fold cross-validation
        if params['nfolds_cv'] > 0:
            f1score_cv = clf.cross_validate(X, y, cv=params['nfolds_cv'])
            print('CV f1score: ', f1score_cv)
        
            #saving CV f1score on test, train and fit time
            results['score_cv_test'] = clf.cv_scores['test_score']
            results['score_cv_train'] = clf.cv_scores['train_score']
            results['fit_cv_time'] = clf.cv_scores['fit_time']
        else:
            results['score_cv_test'] = np.nan
            results['score_cv_train'] = np.nan
            results['fit_cv_time'] = np.nan
        
        #Saving the model (trained on training set only)
        if np.isnan(params['vec_method']):
            model_path = params['modality'] + '/' + params['base_name'] + '_' + params['vec_method']
        else:
            model_path = params['modality'] + '/' + params['base_name']
            
        clf.save(model_path)
        
        #saving where the model is saved
        results['model_path'] = model_path
        
        #Loading results.csv, adding line and saving it
        #Loading results.csv
        df_results = pd.read_csv(results_path, index_col=0)
        for col in df_results.columns:
            if col not in results.keys():
                results[col] = np.nan
        df_results.loc[len(df_results)] = results
        df_results.to_csv(results_path)
        
    return df_results