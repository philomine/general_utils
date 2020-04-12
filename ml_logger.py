import numpy as np 
import pandas as pd 
import time 
import tqdm 
import pickle 

import os 
import datetime 

class MLLogger:
    def __init__(self, experiment_name, metrics):
        """ Create a logger for an experiment
        You should name the experiment and define the metrics you will use 
        throughout the experiment.

        Parameters
        ----------
        experiment_name : string
            The name of the ML experiment.
        
        metrics : list of strings 
            List the metrics by which you will compare the different results.
        """
        self._create_filesystem(experiment_name)

        # Check if the ML experiment already exists
        self._experiment_filepath = \
            f'./ml_experiments/results/{experiment_name}.pickle'
        experiment_exists = os.path.isfile(self._experiment_filepath)

        # Building the result logger dataframe
        if experiment_exists:
            logger = pickle.load(open(self._experiment_filepath, 'rb'))
            correct_metrics = logger.metrics == metrics
            if not correct_metrics:
                raise AttributeError(
                    'An experiment with that name and different metrics ' + 
                    'already exists.'
                )
            result_log = logger.result_log
        else:
            result_log = pd.DataFrame(
                columns=['time', 'method', *metrics]
            )
        
        self.experiment_name = experiment_name
        self.metrics = metrics
        self.result_log = result_log
        self._save()
    
    def _save(self):
        pickle.dump(self, open(self._experiment_filepath, 'wb'))
    
    def _create_filesystem(self, experiment_name):
        experiment_models_folder = \
            f'./ml_experiments/models/{experiment_name}'
        
        if not os.path.isdir('./ml_experiments/'):
            os.mkdir('./ml_experiments/')
        if not os.path.isdir('./ml_experiments/results/'):
            os.mkdir('./ml_experiments/results/')
        if not os.path.isdir('./ml_experiments/models/'):
            os.mkdir('./ml_experiments/models/')
        if not os.path.isdir(experiment_models_folder):
            os.mkdir(experiment_models_folder)
    
    def log(self, method_name, model, metrics):
        """ Logs a result in the result log
        Adds a row to the result_log dataframe to register the performance of a 
        model, and saves the model in pickle format.

        Parameters
        ----------
        method_name : string
            The name of the method (or algorithm used).
        
        model : trained object with a predict method
            The trained model on which the test was made
        
        metrics : dict 
            Dictionary of metrics to log for that trained model. Should contain 
            the defined metrics for that logger.
        """
        log_time = str(datetime.datetime.now())[:19].replace(':', '-')

        metrics['time'] = log_time
        metrics['method'] = method_name 
        self.result_log = self.result_log.append(metrics, ignore_index=True)

        pickle.dump(
            model, 
            open(
                f'./ml_experiments/models/{self.experiment_name}/' + 
                f'{log_time}_{method_name}.pickle',
                "wb"
            )
        )
        self._save()