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
    def __init__(self, experiment_name, train_function=None, get_figures=None, get_metrics=None):
        """ Create a logger for an experiment. The experiment name will define
        your experiment. To load it later, you'll just have to use the same
        name when initialising the logger.

        Parameters
        ----------
        experiment_name : string
            The name of the ML experiment.
        train_function: function returning (model, predictions)
            model is an object with fit and predict attributes and predictions
            is a pd.DataFrame with y_true and y_pred columns, and also
            identification if you wish.
        get_figures: function returning a list of tuples for plotly_reporter
            Refer to plotly reporter module for expected format. This is a list
            of figures reporting the performance of a model. Should accept as
            parameters model_name, y_true, y_pred.
        get_metrics: function(y_true, y_pred) -> dict(metric_name: metric_value)
            Function which takes into parameter y_true, y_pred and returns a
            dict of metrics.
        """
        self._create_filesystem(experiment_name)

        # If the experiment already exists, load it
        if os.path.isfile(f"./ml_experiments/results/{experiment_name}.pickle"):
            logger = pickle.load(open(f"./ml_experiments/results/{experiment_name}.pickle", "rb"))
            self.experiment_name = logger.experiment_name
            self.result_log = logger.result_log
            self.experiment_plan = logger.experiment_plan
            self.models = logger.models
            self.data = logger.data

            if train_function is None:
                self._train = logger._train
            else:
                self._train = train_function
            if get_figures is None:
                self._get_figures = logger._get_figures
            else:
                self._get_figures = get_figures
            if get_metrics is None:
                self._get_metrics = logger._get_metrics
            else:
                self._get_metrics = get_metrics

            self._save()
        # Otherwise, init the attributes
        else:
            self.experiment_name = experiment_name
            self.result_log = pd.DataFrame(columns=["name"])
            self.experiment_plan = {}
            self.models = {}
            self.data = {}

            if train_function is None:
                self._train = _train
            else:
                self._train = train_function
            if get_figures is None:
                self._get_figures = _get_figures
            else:
                self._get_figures = get_figures
            if get_metrics is None:
                self._get_metrics = _get_metrics
            else:
                self._get_metrics = get_metrics

            self._save()

    def _save(self):
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

    def log(self, result_name, model, predictions, metrics):
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
        if result_name in self.result_log["name"]:
            self.result_log = self.result_log[self.result_log["name"] != result_name]

        log_time = str(datetime.datetime.now())[:19].replace(":", "-")

        results = {"name": result_name}
        for metric_name, metric_value in metrics.items():
            results[metric_name] = metric_value
        results["time"] = log_time
        self.result_log = self.result_log.append(results, ignore_index=True)

        model_location = f"./ml_experiments/models/{self.experiment_name}/{result_name}.pickle"
        pickle.dump(model, open(model_location, "wb"))
        pred_location = f"./ml_experiments/predictions/{self.experiment_name}/{result_name}.pickle"
        pickle.dump(predictions, open(pred_location, "wb"))
        self._save()

    def experiment(self, models, data, experiment_plan):
        """Goes through the experiment plan, retrain the new inputs and
        regenerates the report.
        
        Parameters
        ----------
        models: dict of model objects (objects with fit and predict attributes)
            List all the different models on which you want to train your data
        data: dict of data
            List all the different data versions
        experiment_plan: dict of list of tuples {result name: (model name, data name, train kwargs)}
            List all the configurations you wish to try, doesn't go over the
            ones that already have a logged result
        """
        for result_name, conf in experiment_plan.items():
            self.experiment_plan[result_name] = conf
        for model_name, model in models.items():
            self.models[model_name] = model
        for data_name, datum in data.items():
            self.data[data_name] = datum

        for result_name, (model_name, data_name, kwargs) in experiment_plan.items():
            if result_name not in self.result_log["name"].values:
                model = self.models[model_name]
                X, y = self.data[data_name]
                model, predictions = self._train(model, X, y, **kwargs)
                metrics = {"model": model_name, "data": data_name}
                self.log(result_name, model, predictions, metrics)

        self._save()
        self.generate_report()

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
        self.generate_report()

    def set_get_figures(self, get_figures):
        """get_figures should be a function(result_name, y_true, y_pred) -> [(str, object)]"""
        self._get_figures = get_figures
        self._save()
        self.generate_report()

    def set_get_metrics(self, get_metrics):
        """get_metrics should be a function(y_true, y_pred) -> {"metric_name": metric_value}"""
        self._get_metrics = get_metrics
        self._save()
        self.generate_report()

    def generate_report(self, get_figures=None, get_metrics=None, best_result=None):
        if get_figures is None:
            get_figures = self._get_figures
        if get_metrics is None:
            get_metrics = self._get_metrics

        report_contents = []
        metrics = pd.DataFrame()
        for result_name in self.result_log["name"].values:
            predictions = self.load_predictions(result_name)
            metrics = metrics.append(get_metrics(predictions["y_true"], predictions["y_pred"]), ignore_index=True)
            if best_result is None:
                for content in get_figures(result_name, predictions["y_true"], predictions["y_pred"]):
                    report_contents.append(content)
            else:
                if result_name == best_result:
                    for content in get_figures(result_name, predictions["y_true"], predictions["y_pred"]):
                        report_contents.append(content)
        temp = self.result_log.copy()
        for col in metrics.columns:
            temp[col] = metrics[col]

        report_contents = [("title", self.experiment_name), ("pandas", temp)] + report_contents
        generate_report(f"./ml_experiments/reports/{self.experiment_name}.html", report_contents)

    def retrain(self, subset=None, exclude=None):
        """exclude is ignore if subset is not None."""
        results_to_retrain = self.result_log["name"].values
        if subset is None:
            if exclude is not None:
                results_to_retrain = [res for res in results_to_retrain if res not in exclude]
        else:
            results_to_retrain = subset

        for result_name, (model_name, data_name, kwargs) in self.experiment_plan.items():
            if result_name in results_to_retrain:
                model = self.models[model_name]
                X, y = self.data[data_name]
                model, predictions = self._train(model, X, y, **kwargs)
                metrics = {"model": model_name, "data": data_name}
                self.log(result_name, model, predictions, metrics)

        self.generate_report()
