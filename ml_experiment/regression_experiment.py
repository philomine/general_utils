import os
import pickle

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..plotly_reporter import generate_report
from .ml_logger import MLLogger


class RegressionExperiment(MLLogger):
    def __init__(self, experiment_name):
        """ Create an experiment for a regression problem. This class relies on 
        MLLogger class. It adds plotly reports with the results log and
        histograms of errors.

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
            if hasattr(logger, "report_content"):
                self.report_content = logger.report_content
            else:
                self.report_content = []
                self._save()
        # Otherwise, init the error plots
        else:
            self.report_content = []
            self._save()

    def experiment(self, models, data, reporter=None):
        """ Launches an experiment: trains the data and tests the results on 
        every of the models, logs the results and plots the error logs.
        
        Parameters
        ----------
        models: dict of models with fit and predict methods {"model name": model}
            Model to fit and predict
        data: list of 4 np.array (2d, 1d, 2d, 1d)
            X_train, y_train, X_test, y_test
        reporter: function 
            Returns figures to add to the report
        """
        X_train, y_train, X_test, y_test = data
        for model_name, model in models.items():
            if "name" in self.result_log.columns and model_name in self.result_log["name"]:
                print(f"There already is a model {model_name} in your experience")
            else:
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test).flatten()

                    metrics = {}
                    metrics["mse"] = mean_squared_error(y_test, y_pred)
                    metrics["mae"] = mean_absolute_error(y_test, y_pred)

                    self.report_content.append(("subtitle", model_name))

                    error = y_pred - y_test
                    fig = go.Figure(data=[go.Histogram(x=error)])
                    fig = fig.update_layout(title="Error plot")
                    self.report_content.append(("fig", fig))

                    if reporter is not None:
                        for fig in reporter(y_test, y_pred):
                            self.report_content.append(("fig", fig))

                    self.log(model_name, model, metrics)
                except Exception as e:
                    print(f"Warning: Model {type(mod).__name__} didn't work. Error: {e}")

        report_content = [("title", self.experiment_name), ("pandas", self.result_log)]
        report_content = report_content + self.report_content
        generate_report(f"./ml_experiments/reports/{self.experiment_name}.html", report_content)
