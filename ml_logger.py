import datetime
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split

from ..plotly_reporter import generate_report


def _train(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).flatten()
    predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    return model, predictions


def _train_binary(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1].flatten()
    predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    return model, predictions


def _train_cv(model, X, y, cv):
    y_pred = np.array([])
    y_true = np.array([])
    for train_index, test_index in cv.split(X):
        X_train = X.copy()[train_index]
        y_train = y.copy()[train_index]
        X_test = X.copy()[test_index]
        y_test = y.copy()[test_index]
        model.fit(X_train, y_train)
        y_pred = np.append(y_pred, model.predict(X_test).flatten().copy())
        y_true = np.append(y_true, y_test.flatten().copy())

    predictions = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    return model, predictions


def _get_figures(model_name, y_true, y_pred):
    return []


def _get_metrics(y_true, y_pred):
    return {}


class MLLogger:
    def __init__(self, experiment_name):
        """Create a logger for an experiment. The experiment name will define
        your experiment. To load it later, you'll just have to use the same
        name when initialising the logger.

        Parameters
        ----------
        experiment_name : string
            The name of the ML experiment.
        """
        self._create_filesystem(experiment_name)

        # If the experiment already exists, load it
        if os.path.isfile(f"./ml_experiments/results/{experiment_name}.pickle"):
            logger = pickle.load(open(f"./ml_experiments/results/{experiment_name}.pickle", "rb"))
            self.experiment_name = logger.experiment_name
            self.result_log = logger.result_log
            self.kwargs = logger.kwargs
            self.experiment_plan = logger.experiment_plan
            self.feature_importances = logger.feature_importances
            self._save()
        # Otherwise, init the attributes
        else:
            self.experiment_name = experiment_name
            self.result_log = pd.DataFrame(
                columns=["name", "time", "model", "data", "train function", "score"]
            ).set_index("name")
            self.kwargs = pd.DataFrame(columns=["name"]).set_index("name")
            self.experiment_plan = {}
            self.feature_importances = pd.DataFrame(columns=["name"])
            self._save()

    def _save(self):
        self.generate_report()
        pickle.dump(self, open(f"./ml_experiments/results/{self.experiment_name}.pickle", "wb"))

    def _create_filesystem(self, experiment_name):
        if not os.path.isdir("./ml_experiments/"):
            os.mkdir("./ml_experiments/")

        if not os.path.isdir("./ml_experiments/models/"):
            os.mkdir("./ml_experiments/models/")
        if not os.path.isdir(f"./ml_experiments/models/{experiment_name}"):
            os.mkdir(f"./ml_experiments/models/{experiment_name}")

        if not os.path.isdir("./ml_experiments/predictions/"):
            os.mkdir("./ml_experiments/predictions/")
        if not os.path.isdir(f"./ml_experiments/predictions/{experiment_name}"):
            os.mkdir(f"./ml_experiments/predictions/{experiment_name}")

        if not os.path.isdir(f"./ml_experiments/reports/"):
            os.mkdir(f"./ml_experiments/reports/")

        if not os.path.isdir("./ml_experiments/results/"):
            os.mkdir("./ml_experiments/results/")

    def _delete_filesystem(self, experiment_name):
        if os.path.isdir("./ml_experiments/"):

            if os.path.isdir("./ml_experiments/models/"):
                if os.path.isdir(f"./ml_experiments/models/{experiment_name}"):
                    for f in os.listdir(f"./ml_experiments/models/{experiment_name}"):
                        os.remove(f"./ml_experiments/models/{experiment_name}/{f}")
                    os.rmdir(f"./ml_experiments/models/{experiment_name}")

            if os.path.isdir("./ml_experiments/predictions/"):
                if os.path.isdir(f"./ml_experiments/predictions/{experiment_name}"):
                    for f in os.listdir(f"./ml_experiments/predictions/{experiment_name}"):
                        os.remove(f"./ml_experiments/predictions/{experiment_name}/{f}")
                    os.rmdir(f"./ml_experiments/predictions/{experiment_name}")

            if os.path.isdir("./ml_experiments/reports/"):
                if os.path.isfile(f"./ml_experiments/reports/{experiment_name}.html"):
                    os.remove(f"./ml_experiments/reports/{experiment_name}.html")

            if os.path.isdir("./ml_experiments/results/"):
                if os.path.isfile(f"./ml_experiments/results/{experiment_name}.pickle"):
                    os.remove(f"./ml_experiments/results/{experiment_name}.pickle")

    def log(self, result_name, model, model_kwargs, predictions, feature_importances, metrics):
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
        
        predictions: pd.DataFrame with at least y_true, y_pred columns
            The predictions of the model
        
        metrics : dict 
            Dictionary of metrics to log for that trained model. Should contain 
            the defined metrics for that logger.
        """
        if result_name in self.result_log.index:
            self.result_log = self.result_log.drop(result_name)
        if result_name in self.kwargs.index:
            self.kwargs = self.kwargs.drop(result_name)
        if result_name in self.feature_importances.index:
            self.feature_importances = self.feature_importances.drop(result_name)

        log = metrics.copy()
        log["time"] = str(datetime.datetime.now())[:19].replace(":", "-")
        log = pd.Series(log, name=result_name)
        self.result_log = self.result_log.append(log)

        model_kwargs = model_kwargs.copy()
        model_kwargs = pd.Series(model_kwargs, name=result_name)
        self.kwargs = self.kwargs.append(model_kwargs)

        feature_importances = feature_importances.copy()
        feature_importances = pd.Series(feature_importances, name=result_name)
        self.feature_importances = self.feature_importances.append(feature_importances)

        model_location = f"./ml_experiments/models/{self.experiment_name}/{result_name}.pickle"
        pickle.dump(model, open(model_location, "wb"))
        pred_location = f"./ml_experiments/predictions/{self.experiment_name}/{result_name}.pickle"
        pickle.dump(predictions, open(pred_location, "wb"))
        self._save()

    def set_experiment_plan(self, experiment_plan):
        """Adds entries to the experiment plan. The experiment plan is a dict
        of models to train. New entries in the experiment plan will be added.
        Entries already existing will be replaced.
        
        Parameters
        ----------
        experiment_plan: dict of list
            Expecting a dict in the shape
            {"entry_name": ["model_name", "data_name", "train_name", model_kwargs]}
        """
        for entry_name, conf in experiment_plan.items():
            self.experiment_plan[entry_name] = conf
        self._save()

    def get_experiment_plan(self):
        return self.experiment_plan.copy()

    def get_feature_importances(self, result_name=None):
        if result_name is None:
            return self.feature_importances.copy()
        else:
            return self.feature_importances.copy().loc[result_name].dropna()

    def train(self, models={}, data={}, train_functions={}, include=[], exclude=[]):
        """if a name is in include and exclude, it is included."""
        models_to_train = []
        for name in self.experiment_plan:
            if not os.path.isfile(f"./ml_experiments/models/{self.experiment_name}/{name}.pickle"):
                models_to_train.append(name)
        models_to_train = [model for model in models_to_train if model not in exclude]
        models_to_train = np.unique(models_to_train + list(include))

        for name in models_to_train:
            print(f"Training {name}...")
            conf = self.experiment_plan[name]
            model = models[conf[0]]
            X, y, cv_index = data[conf[1]]
            train_function = train_functions[conf[2]]
            model_kwargs = conf[3]
            features = conf[4]

            model, score, predictions, feature_importances = train_function(
                model, model_kwargs, X, y, cv_index, features
            )
            metrics = {"model": conf[0], "data": conf[1], "train function": conf[2], "score": score}
            self.log(name, model, model_kwargs, predictions, feature_importances, metrics)

    def load_model(self, model_name):
        """Loads a model thanks to the given model name. model_name should be in the saved models."""
        return pickle.load(open(f"./ml_experiments/models/{self.experiment_name}/{model_name}.pickle", "rb"))

    def load_predictions(self, result_name):
        """Loads predictions thanks to the given result name. result_name should be in the saved results."""
        return pickle.load(open(f"./ml_experiments/predictions/{self.experiment_name}/{result_name}.pickle", "rb"))

    def delete(self):
        """Deletes the entire experiment."""
        self._delete_filesystem(self.experiment_name)

    def rename(self, new_name):
        """Renames the experiment, ie, renaming the files."""
        old_name = self.experiment_name

        # First, check there is no existing experiment by the new name
        if os.path.isfile(f"./ml_experiments/results/{new_name}.pickle"):
            raise ValueError(
                f"Cannot rename to {new_name}, an experiment with that name "
                + f"already exists. Please delete it with "
                + f"MLLogger({new_name}).delete() if you wish."
            )
        else:
            # Delete other potential files
            MLLogger(new_name).delete()

        if os.path.isdir("./ml_experiments/"):

            if os.path.isdir("./ml_experiments/models/"):
                if os.path.isdir(f"./ml_experiments/models/{old_name}"):
                    os.rename(f"./ml_experiments/models/{old_name}", f"./ml_experiments/models/{new_name}")

            if os.path.isdir("./ml_experiments/predictions/"):
                if os.path.isdir(f"./ml_experiments/predictions/{old_name}"):
                    os.rename(f"./ml_experiments/predictions/{old_name}", f"./ml_experiments/predictions/{new_name}")

            if os.path.isdir("./ml_experiments/reports/"):
                if os.path.isfile(f"./ml_experiments/reports/{old_name}.html"):
                    os.rename(f"./ml_experiments/reports/{old_name}.html", f"./ml_experiments/reports/{new_name}.html")

            if os.path.isdir("./ml_experiments/results/"):
                if os.path.isfile(f"./ml_experiments/results/{old_name}.pickle"):
                    os.rename(
                        f"./ml_experiments/results/{old_name}.pickle", f"./ml_experiments/results/{new_name}.pickle"
                    )
        self.experiment_name = new_name
        self._save()

    def generate_report(self, get_figures=None, get_metrics=None, plotting_plan=None):
        report_contents = [("title", self.experiment_name)]

        result_log = self.result_log.copy()
        if get_metrics is not None:
            metrics = pd.DataFrame()
            for result_name in self.result_log.index:
                predictions = self.load_predictions(result_name)
                metrics = metrics.append(
                    pd.Series(get_metrics(predictions["y_true"], predictions["y_pred"]), name=result_name)
                )
            for col in metrics.columns:
                result_log[col] = metrics[col]
        report_contents.append(("pandas", result_log))
        report_contents.append(("pandas", self.kwargs.copy()))

        if plotting_plan is not None:
            for result_name, get_figures in plotting_plan.items():
                predictions = self.load_predictions(result_name)
                figures = get_figures(result_name, predictions["y_true"], predictions["y_pred"])
                report_contents += figures
        elif get_figures is not None:
            for result_name in self.result_log.index:
                predictions = self.load_predictions(result_name)
                figures = get_figures(result_name, predictions["y_true"], predictions["y_pred"])
                report_contents += figures

        generate_report(f"./ml_experiments/reports/{self.experiment_name}.html", report_contents)
