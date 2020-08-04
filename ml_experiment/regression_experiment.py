import os
import pickle

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..ml_logger import MLLogger
from ..plotly_reporter import generate_report


class RegressionExperiment(MLLogger):
    def __init__(self, experiment_name):
        """ Create an experiment for a regression problem. This class relies on 
        MLLogger class. TODO: inheritance. It adds plotly reports with the 
        results log and histograms of errors.

        Parameters
        ----------
        experiment_name: string
            The name under which you want to save your results.
        """
        super().__init__(experiment_name)

        if not os.path.isdir("./ml_experiments/reports/"):
            os.mkdir("./ml_experiments/reports/")

        # If the experiment already exists, check for error plots
        if os.path.isfile(self._experiment_filepath):
            logger = pickle.load(open(self._experiment_filepath, "rb"))
            if hasattr(logger, "error_plots"):
                self.error_plots = logger.error_plots
            else:
                self.error_plots = []
                self._save()
        # Otherwise, init the error plots
        else:
            self.error_plots = []
            self._save()

    def experiment(self, models, data):
        """ Launches an experiment: trains the data and tests the results on 
        every of the models, logs the results and plots the error logs.
        
        Parameters
        ----------
        models: list of tuples (scikit-learn model, list of kwargs)
            Model to fit and predict
        data: list of 4 np.array (2d, 1d, 2d, 1d)
            X_train, y_train, X_test, y_test
        """
        X_train, y_train, X_test, y_test = data
        for model, kwargs_list in models:
            for i, kwargs in enumerate(kwargs_list):
                mod = model(**kwargs)
                try:
                    mod.fit(X_train, y_train)
                    y_pred = mod.predict(X_test).flatten()

                    name = f"{type(mod).__name__}_{i}"
                    metrics = kwargs.copy()
                    metrics["mse"] = mean_squared_error(y_test, y_pred)
                    metrics["mae"] = mean_absolute_error(y_test, y_pred)

                    error = y_pred - y_test
                    fig = go.Figure(data=[go.Histogram(x=error)])
                    fig = fig.update_layout(title=name)

                    self.error_plots.append(fig)
                    self.log(name, mod, metrics)
                except Exception as e:
                    print(f"Warning: Model {type(mod).__name__} didn't work. Error: {e}")

        report_contents = [("title", self.experiment_name), ("pandas", self.result_log)]
        for error_plot in self.error_plots:
            report_contents.append(("fig", error_plot))
        generate_report(f"./ml_experiments/reports/{self.experiment_name}.html", report_contents)
