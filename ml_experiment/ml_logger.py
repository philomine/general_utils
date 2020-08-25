import datetime
import os
import pickle
import time

import pandas as pd


class MLLogger:
    def __init__(self, experiment_name):
        """ Create a logger for an experiment. The experiment name will define
        your experiment. To load it later, you'll just have to use the same
        name when initialising the logger.

        Parameters
        ----------
        experiment_name : string
            The name of the ML experiment.
        """
        self._create_filesystem(experiment_name)
        self._experiment_filepath = f"./ml_experiments/experiments/{experiment_name}.pickle"

        # If the experiment already exists, load it
        if os.path.isfile(self._experiment_filepath):
            logger = pickle.load(open(self._experiment_filepath, "rb"))
            self.experiment_name = logger.experiment_name
            self.result_log = logger.result_log
        # Otherwise, init the attributes
        else:
            self.experiment_name = experiment_name
            self.result_log = pd.DataFrame()
            self._save()

    def _save(self):
        pickle.dump(self, open(self._experiment_filepath, "wb"))

    def _create_filesystem(self, experiment_name):
        if not os.path.isdir("./ml_experiments/"):
            os.mkdir("./ml_experiments/")
        if not os.path.isdir("./ml_experiments/experiments/"):
            os.mkdir("./ml_experiments/experiments/")
        if not os.path.isdir("./ml_experiments/models/"):
            os.mkdir("./ml_experiments/models/")
        if not os.path.isdir(f"./ml_experiments/models/{experiment_name}"):
            os.mkdir(f"./ml_experiments/models/{experiment_name}")

    def _delete_filesystem(self, experiment_name):
        if os.path.isdir("./ml_experiments/"):
            if os.path.isdir("./ml_experiments/experiments/"):
                if os.path.isfile(f"./ml_experiments/experiments/{experiment_name}.pickle"):
                    os.remove(f"./ml_experiments/experiments/{experiment_name}.pickle")
            if os.path.isdir("./ml_experiments/models/"):
                if os.path.isdir(f"./ml_experiments/models/{experiment_name}"):
                    for f in os.listdir(f"./ml_experiments/models/{experiment_name}"):
                        os.remove(f"./ml_experiments/models/{experiment_name}/{f}")
                    os.rmdir(f"./ml_experiments/models/{experiment_name}")

    def log(self, result_name, model, metrics):
        """ Logs a result in the result log
        Adds a row to the result_log dataframe to register the performance of a
        model, and saves the model in pickle format.

        Parameters
        ----------
        result_name : string
            The name you'll find your result under (can be algorithm named 
            joined with its parameters for example)
        
        model : trained object with a predict method
            The trained model on which the test was made
        
        metrics : dict 
            Dictionary of metrics to log for that trained model. Should contain 
            the defined metrics for that logger.
        """
        log_time = str(datetime.datetime.now())[:19].replace(":", "-")

        metrics["time"] = log_time
        metrics["name"] = result_name
        self.result_log = self.result_log.append(metrics, ignore_index=True)

        model_location = f"./ml_experiments/models/{self.experiment_name}/{result_name}.pickle"
        pickle.dump(model, open(model_location, "wb"))
        self._save()

    def load(self, model_name):
        """Loads a model thanks to the given model name. model_name should be in the saved models."""
        return pickle.load(open(f"./ml_experiments/models/{self.experiment_name}/{model_name}.pickle", "rb"))

    def delete(self):
        """Deletes the entire experiment."""
        self._delete_filesystem(self.experiment_name)
