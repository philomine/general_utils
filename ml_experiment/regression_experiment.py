import os
import pickle

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..ml_logger import MLLogger
from ..plotly_reporter import generate_report


class RegressionExperiment:
    def __init__(self, experiment_name):
        """ Create an experiment for a regression problem. This class relies on 
        MLLogger class. TODO: inheritance. It adds plotly reports with the 
        results log and histograms of errors.

        Parameters
        ----------
        experiment_name: string
            The name under which you want to save your results.
        """
        self.experiment_name = experiment_name
        self.logger = MLLogger(experiment_name, ["mse"])

        if os.path.isfile(self.logger._experiment_filepath):
            logger = pickle.load(open(self.logger._experiment_filepath, "rb"))
            if not hasattr(logger, "error_plots"):
                self.logger.error_plots = []
            else:
                self.logger.error_plot = logger.error_plots
        else:
            self.logger.error_plots = []

        if not os.path.isdir("./ml_experiments/reports/"):
            os.mkdir("./ml_experiments/reports/")

    def experiment(self, models, data):
        """ Launches an experiment: trains the data and tests the results on 
        every of the models, logs the results and plots the error logs.
        
        Parameters
        ----------
        models: list of scikit learn models with fit and predict method
            Model to fit and predict
        data: list of 4 pd.DataFrame
            X_train, y_train, X_test, y_test
        """
        X_train, y_train, X_test, y_test = data
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mape = np.mean(mean_absolute_error(y_test, y_pred))
            mse = np.mean(mean_squared_error(y_test, y_pred))

            metrics = {"mse": mse, "mape": mape}
            name = type(model).__name__

            fig = go.Figure(data=[go.Histogram(x=(y_pred - np.array(y_test)).flatten())])
            fig = fig.update_layout(title=name)

            self.logger.log(name, model, metrics)
            self.logger.error_plots.append(fig)

        report_contents = [("title", self.experiment_name), ("pandas", self.logger.result_log)]
        for error_plot in self.logger.error_plots:
            report_contents.append(("fig", error_plot))
        generate_report(f"./ml_experiments/reports/{self.experiment_name}.html", report_contents)