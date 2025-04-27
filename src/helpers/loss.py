import os
import re
import random
from math import sqrt
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def standardize_values(y: np.ndarray,
                       stations: np.ndarray,
                       station_stats: pd.DataFrame) -> np.ndarray:
    """
    Standardize values based on station-level statistics.

    Parameters:
        y (np.ndarray): The values to standardize.
        stations (np.ndarray): The station codes.
        station_stats (pd.DataFrame): The station-level statistics.

    Returns:
        np.ndarray: The standardized values.
    """
    out = np.empty_like(y, dtype=float)
    for s in np.unique(stations):
        idx = stations == s
        min = station_stats.loc[s, 'min']
        max = station_stats.loc[s, 'max']
        out[idx] = (y[idx] - min) * 100.0 / (max - min)
    return out

def get_station_stats(
    y: np.ndarray,
    station_code: np.ndarray
) -> pd.DataFrame:
    """
    Compute station-level statistics for the given data.

    Args:
        y (np.ndarray): A NumPy array of numeric measurements or target values.
        station_code (np.ndarray): A NumPy array of station codes.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds
            to a unique station code and includes statistics.
    """
    df = pd.DataFrame({'y': y, 'station_code': station_code})
    station_stats = df.groupby('station_code')['y'].agg(
        ['mean', 'std', 'min', 'max'])
    return station_stats


def standardize_prediction_intervals(
    y_pred_intervals: np.ndarray,
    stations: np.ndarray,
    station_stats: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardizes the prediction interval values for a set of
    stations using the provided station statistics.

    Args:
        y_pred_intervals (np.ndarray): the predicted interval values.
        stations (np.ndarray): station codes.
        station_stats (pd.DataFrame): statistics for each station.

    Returns:
        Tuple[np.ndarray, np.ndarray]: standardized lower and
            upper prediction interval values.
    """
    if y_pred_intervals is None:
        return None, None

    if len(y_pred_intervals.shape) == 3:
        y_pred_lower_std = standardize_values(
            y_pred_intervals[:, 0, 0],
            stations,
            station_stats)
        y_pred_upper_std = standardize_values(
            y_pred_intervals[:, 1, 0],
            stations,
            station_stats)
    else:
        y_pred_lower_std = standardize_values(
            y_pred_intervals[:, 0],
            stations,
            station_stats)
        y_pred_upper_std = standardize_values(
            y_pred_intervals[:, 1],
            stations,
            station_stats)

    return y_pred_lower_std, y_pred_upper_std


def compute_per_station_metrics(
    y_true_std: np.ndarray,
    y_pred_std: np.ndarray,
    stations: np.ndarray,
    y_pred_lower_std: np.ndarray = None,
    y_pred_upper_std: np.ndarray = None
) -> pd.DataFrame:
    """
    Compute station-level performance metrics including scaled RMSE,
    scaled MAE, coverage, scaled prediction interval size,
    and Gaussian negative log-likelihood.

    Parameters:
        y_true_std (np.ndarray): Standardized ground truth.
        y_pred_std (np.ndarray): Array of predicted standardized predictions.
        stations (np.ndarray): Station codes.
        y_pred_lower_std (np.ndarray): lower prediction interval values.
        y_pred_upper_std (np.ndarray): upper prediction interval values.

    Returns:
    pd.DataFrame
        Dataframe with the following metrics:
            - station_code: Identifier for the station.
            - scaled_rmse: Scaled Root Mean Squared Error for the station.
            - scaled_mae: Scaled Mean Absolute Error for the station.
            - coverage
            - scaled_interval_size: Average size of the prediction interval
            - log_likelihood: Gaussian negative log-likelihood.
    """
    station_list = np.unique(stations)

    records = []

    has_intervals = (
        (y_pred_lower_std is not None) and
        (y_pred_upper_std is not None)
    )

    for s in station_list:
        idx = (stations == s)
        y_true_s = y_true_std[idx]
        y_pred_s = y_pred_std[idx]

        rmse_s = sqrt(mean_squared_error(y_true_s, y_pred_s))
        mae_s = mean_absolute_error(y_true_s, y_pred_s)

        if has_intervals:
            y_lower_s = y_pred_lower_std[idx]
            y_upper_s = y_pred_upper_std[idx]

            # Estimate sigma using the 95% confidence interval approximation
            sigma_s = (y_upper_s - y_lower_s) / 3.29

            # Compute Gaussian negative log-likelihood
            nll_s = (1 / len(y_true_s)) * np.sum(
               np.log(sigma_s) + abs((y_true_s - y_pred_s)) / abs(2 * sigma_s)
            )

            coverage_s = np.mean(
                (y_true_s >= y_lower_s) & (y_true_s <= y_upper_s))
            interval_size_s = np.mean(y_upper_s - y_lower_s)
        else:
            sigma_s = np.std(y_true_s - y_pred_s)  # Fallback estimation
            sigma_s = max(sigma_s, 1e-6)  # Ensure non-zero, positive sigma

            nll_s = (1 / len(y_true_s)) * np.sum(
                np.log(sigma_s) + abs((y_true_s - y_pred_s)) / abs(2 * sigma_s)
            )

            coverage_s = np.nan
            interval_size_s = np.nan

        # Collect station-level metrics
        records.append({
            'station_code': s,
            'scaled_rmse': rmse_s,
            'scaled_mae': mae_s,
            'coverage': coverage_s,
            'scaled_interval_size': interval_size_s,
            'log_likelihood': nll_s
        })

    return pd.DataFrame(records)


def summarize_metrics(
    metrics: pd.DataFrame,
    model_name: str,
    dataset_type: str
) -> pd.DataFrame:
    """
    Given a station-level metrics DataFrame, compute average (per station)
    values and a final score.

    Parameters:
        metrics (pd.DataFrame): station-level metrics.
        model_name (str): The name of the model.
        dataset_type (str): The type of dataset used (e.g., "test").

    Returns:
    pd.DataFrame
        A DataFrame containing the final model-level metrics
        (scaled RMSE, log-likelihood, scaled MAE, coverage,
        scaled interval size).
    """
    rmse_final = np.nanmean(metrics['scaled_rmse'])
    mae_final = np.nanmean(metrics['scaled_mae'])
    log_likelihood = np.nanmean(metrics['log_likelihood'])

    if metrics['coverage'].count() == 0:
        coverage_final = np.nan
        interval_size_final = np.nan
    else:
        coverage_final = np.nanmean(metrics['coverage'])
        interval_size_final = np.nanmean(metrics['scaled_interval_size'])

    data = {
        "model": [model_name],
        "dataset": [dataset_type],
        "scaled_rmse": [rmse_final],
        "log_likelihood": [log_likelihood],
        "scaled_mae": [mae_final],
        "coverage": [coverage_final],
        "scaled_interval_size": [interval_size_final],
    }
    return pd.DataFrame(data)

def custom_log_likelihood(estimator,
                          X,
                          y_true,
                          cv_data,
                          station_stats,
                          alpha=.1):
    """
    Custom log-likelihood scoring function.

    Parameters:
        estimator : The fitted estimator with a .predict method.
        X : DataFrame of predictor variables.
        y_true : True target values.
        cv_data : Full DataFrame that includes extra columns
        (e.g., "station_code").
        station_stats : Station-level statistics needed for standardization.
        alpha : Significance level (default from ALPHA).

    Returns:
        nll_s : Computed log-likelihood score.
    """
    # Align y_true with X.
    y_true = pd.Series(y_true.values, index=X.index)

    # Get predictions.
    y_pred = estimator.predict(X)

    # Get quantile predictions.
    y_quantiles = estimator.predict(X, quantiles=[alpha / 2, 1 - alpha / 2])

    # Retrieve station codes from cv_data using X's indices.
    current_stations = cv_data.loc[X.index, "station_code"].to_numpy()

    # Standardize the values.
    y_true_std = standardize_values(
        y_true.to_numpy(),
        current_stations,
        station_stats)
    y_pred_std = standardize_values(
        y_pred,
        current_stations,
        station_stats)
    y_lower_std, y_upper_std = standardize_prediction_intervals(
        y_quantiles,
        current_stations,
        station_stats)

    # Compute sigma from the prediction interval.
    sigma_std = (y_upper_std - y_lower_std) / 3.29
    sigma_std = np.maximum(sigma_std, 1e-6)

    # Compute the negative log-likelihood.
    nll_s = (1 / len(y_true_std)) * np.sum(
        np.log(sigma_std) + np.abs(y_true_std - y_pred_std) / (2 * sigma_std)
    )

    # Optionally, print some diagnostics.
    cov = np.mean(
        (y_true_std >= y_lower_std) & (y_true_std <= y_upper_std))
    i_size = np.mean(y_upper_std - y_lower_std)
    print(
        f"Fold: coverage = {cov:.3f}, interval size = {i_size:.3f}")

    return nll_s

def compute_non_negative_log_likelihood(
    y_true,
    y_pred, 
    y_quantiles,
    station_codes,
    station_stats,
    alpha=.1
):
    """
    Custom log-likelihood scoring function.

    Parameters:
        y_pred : The predictions
        y_quantiles: Prediction Intervals
        y_true : True target values.
        cv_data : Full DataFrame that includes extra columns
        (e.g., "station_code").
        station_stats : Station-level statistics needed for standardization.
        alpha : Significance level (default from ALPHA).

    Returns:
        nll_s : Computed log-likelihood score.
    """
    # Standardize the values.
    y_true_std = standardize_values(
        y_true.to_numpy(),
        station_codes,
        station_stats)
    y_pred_std = standardize_values(
        y_pred,
        station_codes,
        station_stats)
    y_lower_std, y_upper_std = standardize_prediction_intervals(
        y_quantiles,
        station_codes,
        station_stats)

    # Compute sigma from the prediction interval.
    sigma_std = (y_upper_std - y_lower_std) / 3.29
    sigma_std = np.maximum(sigma_std, 1e-6)
    
    # Compute the negative log-likelihood.
    nll_s = (1 / len(y_true_std)) * np.sum(
        np.log(sigma_std) + np.abs(y_true_std - y_pred_std) / (2 * sigma_std)
    )

    # Optionally, print some diagnostics.
    cov = np.mean(
        (y_true_std >= y_lower_std) & (y_true_std <= y_upper_std))
    i_size = np.mean(y_upper_std - y_lower_std)
    print(
        f"Fold: coverage = {cov:.3f}, interval size = {i_size:.3f}")

    return nll_s
