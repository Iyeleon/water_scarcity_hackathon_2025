from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
import pandas as pd
import numpy as np

class GroupMinMaxScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, group_column):
        self.group_column = group_column
        self.scalers_ = {}
        self.input_features_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pandas DataFrame")
        self.input_features_ = list(X.columns)
        for group, group_data in X.groupby(self.group_column):
            features = group_data.drop(columns=self.group_column)
            self.scalers_[group] = {
                'min': features.min(),
                'max': features.max().replace(0, 1)
            }
        self._is_fitted = True
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pandas DataFrame")
        X_scaled = pd.DataFrame(index=X.index, columns=X.columns)
        for group, group_data in X.groupby(self.group_column):
            features = group_data.drop(columns=self.group_column)
            min_ = self.scalers_[group]['min']
            max_ = self.scalers_[group]['max']
            scaled = (features - min_) / (max_ - min_)
            scaled[self.group_column] = group
            X_scaled.loc[group_data.index] = scaled
            X_scaled[self.group_column] = group
        X_scaled = X_scaled[self.input_features_]
        return X_scaled

    def inverse_transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pandas DataFrame")
        X_inv = pd.DataFrame(index=X.index, columns=X.columns)
        for group, group_data in X.groupby(self.group_column):
            features = group_data.drop(columns=self.group_column)
            min_ = self.scalers_[group]['min']
            max_ = self.scalers_[group]['max']
            unscaled = features * (max_ - min_) + min_
            unscaled[self.group_column] = group
            X_inv.loc[group_data.index] = unscaled
            X_inv[self.group_column] = group
        X_inv = X_inv[self.input_features_]
        return X_inv

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.input_features_ is None:
                raise ValueError("input_features must be provided or fit must be called first.")
            input_features = self.input_features_
        return self.input_features_#[f for f in input_features if f != self.group_column]

class GroupStandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, group_column):
        self.group_column = group_column
        self.scalers_ = {}
        self.input_features_ = None
        self._is_fitted = False

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pandas DataFrame")
        self.input_features_ = list(X.columns)
        for group, group_data in X.groupby(self.group_column):
            features = group_data.drop(columns=self.group_column)
            self.scalers_[group] = {
                'mean': features.mean(),
                'std': features.std().replace(0, 1)
            }
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("This GroupStandardScaler instance is not fitted yet.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pandas DataFrame")
        X_scaled = pd.DataFrame(index=X.index, columns=X.columns)
        for group, group_data in X.groupby(self.group_column):
            features = group_data.drop(columns=self.group_column)
            mean = self.scalers_[group]['mean']
            std = self.scalers_[group]['std']
            scaled = (features - mean) / std
            scaled[self.group_column] = group
            X_scaled.loc[group_data.index] = scaled
            X_scaled[self.group_column] = group
        X_scaled = X_scaled[self.input_features_]
        return X_scaled

    def inverse_transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("This GroupStandardScaler instance is not fitted yet.")
        X_inv = pd.DataFrame(index=X.index, columns=X.columns)
        for group, group_data in X.groupby(self.group_column):
            features = group_data.drop(columns=self.group_column)
            mean = self.scalers_[group]['mean']
            std = self.scalers_[group]['std']
            unscaled = (features * std) + mean
            unscaled[self.group_column] = group
            X_inv.loc[group_data.index] = unscaled
            X_inv[self.group_column] = group
        X_inv = X_inv[self.input_features_]
        return X_inv

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if not self._is_fitted:
            raise RuntimeError("This GroupStandardScaler instance is not fitted yet.")
        if input_features is None:
            input_features = self.input_features_
        return self.input_features_#[f for f in input_features if f != self.group_column]

