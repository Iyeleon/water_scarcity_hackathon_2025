import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
import tqdm
import json
import joblib
import datetime
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, pairwise_distances
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

import catboost as cb

from src.helpers import (
    ContiguousGroupKFold, 
    ContiguousTimeSeriesSplit,
    custom_log_likelihood, 
    compute_per_station_metrics, 
    get_station_stats, 
    standardize_values, 
    standardize_prediction_intervals, 
    compute_per_station_metrics, 
    summarize_metrics,
    compute_non_negative_log_likelihood,
    GroupMinMaxScaler,
    GroupStandardScaler
)

class MultiOutputRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, estimator_params, n_targets = None, chained = True):
        self.base_estimator = base_estimator
        self.estimator_params = estimator_params
        self.chained = chained
        self.n_targets = n_targets
        self.models_ = []

    def fit(self, X, y, eval_set=None, **fit_params):
        # X = np.array(X)
        # Y = np.array(y)
        self.models_ = []

        self.n_targets = y.shape[1]

        X_val, Y_val = None, None
        if eval_set:
            X_val, Y_val = eval_set[0]

        X_train_chain = X.copy()
        X_val_chain = X_val.copy() if eval_set else None

        for i in range(self.n_targets):
            params = self.estimator_params.copy()
            model = self.base_estimator(**params)
            y_train_i = y.iloc[:, i]

            # Resolve fit parameters (per target if needed)
            local_fit_params = {}
            for k, v in fit_params.items():
                if isinstance(v, (list, tuple)) and len(v) == self.n_targets:
                    local_fit_params[k] = v[i]
                else:
                    local_fit_params[k] = v

            if eval_set:
                y_val_i = Y_val.iloc[:, i]
                model.fit(
                    X_train_chain, y_train_i,
                    eval_set=[(X_val_chain, y_val_i)],
                    **local_fit_params
                )
            else:
                model.fit(X_train_chain, y_train_i, **local_fit_params)

            self.models_.append(model)
            if self.chained:
                y_pred_train = model.predict(X_train_chain).reshape(-1, 1)
                X_train_chain[f'pred_{i}'] = y_pred_train #np.hstack([X_train_chain, y_pred_train])

                if eval_set:
                    y_pred_val = model.predict(X_val_chain).reshape(-1, 1)
                    X_val_chain[f'pred_{i}'] = y_pred_val #np.hstack([X_val_chain, y_pred_val])

        return self

    def predict(self, X):
        X = np.array(X)
        X_chain = X.copy()
        preds = []

        for model in self.models_:
            y_pred = model.predict(X_chain).reshape(-1, 1)
            preds.append(y_pred)
            X_chain = np.hstack([X_chain, y_pred])

        return np.hstack(preds)

    def save_model(self, path):
        """Save the chained model to disk."""
        os.makedirs(path, exist_ok=True)

        meta = {
            'n_targets': self.n_targets,
            'estimator_params': self.estimator_params,
            'chained': self.chained,
            'model_paths': []
        }

        for idx, model in enumerate(self.models_):
            model_path = os.path.join(path, f"catboost_target_{idx}.cbm")
            model.save_model(model_path)
            meta['model_paths'].append(f"catboost_target_{idx}.cbm")

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load_model(cls, path, base_estimator):
        """Load the chained model from disk."""
        with open(os.path.join(path, "metadata.json")) as f:
            meta = json.load(f)

        model = cls(base_estimator = base_estimator, estimator_params = meta['estimator_params'], chained = meta['chained'], n_targets=meta['n_targets'])
        model.models_ = []

        for model_file in meta['model_paths']:
            # cbm = clone(base_estimator)
            cbm = base_estimator()#**self.estimator_params)
            cbm.load_model(os.path.join(path, model_file))
            model.models_.append(cbm)

        return model

    def get_best_iter(self):
        return np.array([model.best_iter for model in self.models_])
def create_gbt_model(model_type = 'catboost', model_params = {}, chained = True):
    if model_type == 'catboost':
        base_model = cb.CatBoostRegressor#(*args, **kwargs)
    elif model_type == 'lightgbm':
        # base_model = lgbm.LGBMRegressor(*args, **kwargs)
        raise NotImplementedError
    elif model_type == 'xgboost':
        # base_model = xgb.XGBRegressor(*args, **kwargs)
        raise NotImplementedError
    else:
        raise ValueError('Unsupported model type')
    model = MultiOutputRegressor(base_model, model_params, chained = chained)
    return model
        
class BaseRegressor(ABC):
    def __init__(
        self, 
        model_fn = None, 
        model_params = None, 
        preprocessor = None, 
        cv = 5, 
        cv_group = 'year', 
        alphas = [0.05, 0.95], 
        method = 'indirect',
        n_models = 5,
        cat_features = None,
        exclude_cols = None,
        random_state = 42,
        bootstrap = True,
        delta = 0.01,
        patience = 5,
        split_type = 'time_series'
    ):
        self.model_fn = model_fn
        self.model_params = model_params
        self.preprocessor = preprocessor
        self.cv = cv
        self.cv_group = cv_group
        self.alphas = alphas
        self.method = method
        self.n_models = n_models
        self.cat_features = cat_features[::] if cat_features is not None else None
        self.exclude_cols = exclude_cols
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.delta = delta
        self.patience = patience
        self.split_type = split_type
        # if self.model_params is not None:
        #     self.lr = self.model_params.pop('lr') if 'lr' in self.model_params.keys() else 0.01

        self.preprocessors = {}
        self.models = {}
        self.history = {}

    @abstractmethod
    def fit(self, X, y):           
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, save_path):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, load_path):
        raise NotImplementedError

    @abstractmethod
    def prep_data_for_model(self, X):
        raise NotImplementedError

    def get_data(self, X, y, train_ids, val_ids):
        X_train, y_train = X.iloc[train_ids], y.iloc[train_ids]
        X_val, y_val = X.iloc[val_ids], y.iloc[val_ids]
        return X_train, X_val, y_train, y_val

    def cv_split(self, X, group):
        if self.split_type == 'time_series':
            cvf = ContiguousTimeSeriesSplit(self.cv, 0.6)
        elif self.split_type == 'group_kfold':
            cvf = ContiguousGroupKFold(self.cv)
        else:
            raise ValueError('`split_type` must be one of `time_series` or `group_kfold`')
        return cvf.split(X, groups = group)


class GBTEnsembleRegressor(BaseRegressor):
    def __init__(self, *args, model_type = 'catboost', lr = None, k = 12, chained = True, use_priors = False, enforce_location = True, location_column = None, min_patience = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
        if self.lr is not None:
            if self.method == 'direct':
                assert len(self.lr) == 3, '`len(lr)` must be 3 if method = direct'
            elif self.method == 'indirect':
                assert len(self.lr) == self.n_models, '`len(lr)` must be same as n_models'
        self.model_type = model_type
        self.chained = chained
        self.use_priors = use_priors
        self.enforce_location = enforce_location
        self.location_column = location_column
        if self.enforce_location:
            assert self.location_column is not None, '`location_column` is None! You must provide location_column present in your dataset'
        self.min_patience = min_patience
        self.results = {}
        self.residuals_ = []
        self.feature_names_ = None
        self.stations_df = None
        self.priors_df = None
        self.k = k
        self.kmeans = None

    def _prep_kmeans_data(self, X):
        assert 'station_code' in X.columns, 'X must contain `station_code` column!'
        assert 'longitude' in X.columns, 'X must containt `longitude` column!'
        assert 'latitude' in X.columns, 'X must contain `latitude` column!'
        data = X.groupby(['station_code', 'longitude', 'latitude']).agg(['mean', 'std']).reset_index().set_index('station_code')
        return data

    def _do_kmeans(self, X):
        X_ = self._prep_kmeans_data(X)
        km = Pipeline([
            ('scaler', MinMaxScaler()),
            ('kmeans', KMeans(self.k, random_state = 42))
        ])
        km.fit(X_)
        return km

    def _create_cluster_column(self, X, km, label_map = None):
        # copy dataset
        X = X.copy()
        # prepare and predict label
        X_ = self._prep_kmeans_data(X[self.km_columns])
        X_['labels'] = km.predict(X_)
        # align with label map
        if label_map is not None:
            X_['labels'] = X_['labels'].map(label_map)
        # merge to original dataframe
        X['cluster'] = X.station_code.map(X_.labels.to_dict()).astype('category')
        return X
        
    def _align_cluster_centers(self, gcc, cc):
        cost_matrix = pairwise_distances(cc, gcc)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        label_map = dict(zip(row_ind, col_ind))
        return label_map
        
    def fit(self, X, y, batch_size = 32, epochs = 100):

        self.stations_df = X.drop_duplicates(subset = 'station_code')[['station_code', 'river', 'location', 'river_ranking', 'latitude', 'longitude']]
        self.priors_df = self._compute_empirical_priors(X, y, target_col = 'water_flow_week_1')

        # kmeans
        self.km_columns = [i for i in X.columns if ('water_flow' in i and 'trend' not in i) or i in ['latitude', 'longitude', 'station_code']]
        data = X[self.km_columns]
        self.km = self._do_kmeans(data)
        global_cluster_centers = self.km['kmeans'].cluster_centers_
        self.cat_features.append('cluster')
    
        
        # split the data into folds
        splits = self.cv_split(X, X[self.cv_group])

        # loop through each fold
        for idx, (train_ids, val_ids) in enumerate(splits):
            print(f'---------------------FOLD {idx}-------------------------')
            # get fold data
            X_train, X_val, y_train, y_val = self.get_data(X, y, train_ids, val_ids)

            # do kmeans clustering
            X_train_km_data = X_train[self.km_columns]
            km = self._do_kmeans(X_train_km_data)
            label_map = self._align_cluster_centers(global_cluster_centers, km['kmeans'].cluster_centers_)
            X_train = self._create_cluster_column(X_train, km, label_map)
            X_val = self._create_cluster_column(X_val, km, label_map)
            print('Training on years:', *sorted(X_train.year.unique()))
            print('Validating on years:', *sorted(X_val.year.unique()))

            # get priors
            if self.use_priors:
                stations_df = X_train.drop_duplicates(subset = 'station_code')[['station_code', 'river', 'location', 'river_ranking', 'latitude', 'longitude']]
                priors_df = self._compute_empirical_priors(X_train, y_train, target_col = 'water_flow_week_1')
                X_train = self._compute_train_empirical_priors(X_train, y_train, target_col = 'water_flow_week_1')
                X_val = self._get_empirical_priors(X_val, priors_df, stations_df)

            # get station codes before preprocessing the data
            station_codes = X_val.station_code.values.tolist()
            
            # preprocess data
            if self.preprocessor is not None:                
                preprocessor = clone(self.preprocessor)
                X_train_ = preprocessor.fit_transform(X_train)
                X_val_ = preprocessor.transform(X_val)
    
                cols_ = preprocessor.get_feature_names_out()
                cols = [col.split('__')[-1] for col in cols_]
    
                X_train_ = pd.DataFrame(X_train_, columns = cols, index = y_train.index)
                X_val_ = pd.DataFrame(X_val_, columns = cols, index = y_val.index)

                # save preprocessor
                self.preprocessors[f'fold_{idx}'] = preprocessor

            X_train_ = self.prep_data_for_model(X_train_)
            X_val_ = self.prep_data_for_model(X_val_)
            
            self.feature_names_ = list(X_train_.columns)
            if self.enforce_location:
                weights = self._get_feature_weights('location', self.feature_names_)
            else:
                weights = None

            # get the type of model to build, direct or indirect
            if self.method == 'direct':
                self.n_models = 3
                losses = [f'Quantile:alpha={self.alphas[0]}', 'Quantile:alpha=0.5', f'Quantile:alpha={self.alphas[1]}']
                od_wait = [self.patience // 5, self.patience, self.patience // 5]
                y_out = []
                for i in range(self.n_models):
                    print(f'Training model fold_{idx}_model_{i}')
                    # create model
                    model = self.build_model(self.model_type, loss_function = losses[i], eval_metric = 'MAE', lr = self.lr[i], od_wait = od_wait[i], feature_weights = weights, random_state = self.random_state)  # use quantile loss
                    # fit model
                    model.fit(X_train_, y_train, eval_set = [(X_val, y_val)])
                    # store model
                    self.models[f'fold_{idx}_model_{i}'] = model
                    print('----------------------------------------------------')
                    # store X_val predictions for evaluation
                    pred = model.predict(X_val_)
                    pred = np.clip(pred, 0, np.inf)
                    y_out.append(pred)
                y_pred = y_out[1].reshape(-1)
                y_lower = y_out[0].reshape(-1)
                y_upper = y_out[2].reshape(-1)

                # # get conformal scores
                # yv_ = y_val.values.reshape(-1)
                # # yv_ = np.expm1(yv_)
                # resids = np.maximum(y_lower - yv_, yv_ - y_upper) / (y_pred + 1)
                # self.residuals_.extend(resids)
                # delta = np.quantile(resids, self.alphas[1] - self.alphas[0])

                # # adjust results
                # y_out = np.sort(np.array([y_lower, y_pred, y_upper]), axis = 0)
                # y_pred = y_out[1]
                # y_lower = np.clip(y_out[0] - (delta * y_pred), 0, np.inf)  #self.delta
                # y_upper = np.clip(y_out[2] + (delta * y_pred), 0, np.inf) + 0.01 #self.delta
       
            elif self.method == 'indirect':
                y_out = []
                residuals_ = []
                for i in range(self.n_models):
                    SEED = i + idx*100
                    print(f'Training model fold_{idx}_model_{i}')
                    # create model
                    quantiles = norm.cdf(np.linspace(-0.5, 0.5, self.n_models))#np.linspace(-0.67449, 0.67449, self.n_models)
                    od_wait = np.clip(
                        (np.exp(-np.abs(np.arange(self.n_models) - (self.n_models / 2))) * self.patience).astype(int), 
                        self.min_patience, 
                        self.patience
                    )
                    model = self.build_model(
                        self.model_type, 
                        loss_function = 'MAE', 
                        eval_metric = 'MAE', 
                        od_wait = int(od_wait[i]), 
                        feature_weights = weights, 
                        random_state = SEED
                    )
                    # fit model
                    X_, y_ = X_train_.copy(), y_train.copy()
                    model.fit(X_, y_, eval_set = [(X_val_, y_val)])
                    # store model
                    self.models[f'fold_{idx}_model_{i}'] = model
                    print('----------------------------------------------------')
                    pred = model.predict(X_val_)
                    pred = np.clip(pred, 0, np.inf)
                    y_out.append(pred)
                y_out = np.stack(y_out, axis = 1)
                y_pred = np.mean(y_out, axis = 1).reshape(-1)
                y_lower = np.quantile(y_out, self.alphas[0], axis = 1).reshape(-1) #- self.delta
                y_upper = np.quantile(y_out, self.alphas[1], axis = 1).reshape(-1) #+ self.delta
                
            yv_ = y_val.values.reshape(-1)

            # conformal calibration
            relative_residuals = np.maximum(y_lower - yv_, yv_ - y_upper) / (y_pred + 1e-2)
            residuals_df = pd.DataFrame({'cluster': np.repeat(X_val.cluster.values, y_val.shape[1]), 'residuals': relative_residuals})
            self.residuals_.append(residuals_df)

            # compute residuals per cluster for each group
            residuals_per_cluster = residuals_df.groupby('cluster', observed = True).residuals.quantile(0.9)
            residuals_df['delta'] = residuals_df.cluster.map(residuals_per_cluster.to_dict()).astype('float')
            print('Empirical Coverage:', np.mean((y_lower <= yv_) & (y_upper >= yv_)))
            
            y_lower = np.clip(y_lower - (residuals_df.delta * y_pred), 0, np.inf) #self.delta
            y_upper = np.clip(y_upper + (residuals_df.delta * y_pred), 0, np.inf) + 0.01 #self.delta
                
            # compute per fold evaluation on nll
            # compute oof results
            y_true = y_val.values.reshape(-1)
            y_quantiles = np.array(list(zip(y_lower, y_upper)))
            station_codes = np.repeat(station_codes, y_val.shape[1])
            station_stats = get_station_stats(y_true, station_codes)
            nll = compute_non_negative_log_likelihood(
                pd.Series(y_true),
                y_pred, 
                y_quantiles,
                station_codes,
                station_stats,
                alpha = 0.1
            )
            self.results[f'fold_{idx}'] = nll
            print('-------------------------------------------------')
            print(f'Fold {idx} NLL:', nll, 'delta:', residuals_per_cluster.mean())

        # get delta
        global_residuals = pd.concat(self.residuals_)
        self.delta = global_residuals.groupby('cluster').residuals.quantile(self.alphas[1] - self.alphas[0]).to_dict()
        self.delta = {str(k):float(v) for k, v in self.delta.items()}
        print('Conformal adjustment:', self.delta)

    def predict(self, X):
        results = {}
        # create a copy of the test set
        X = X.copy() 
        X = self._create_cluster_column(X, self.km, None)
        if self.use_priors:
            X = self._get_empirical_priors(X, self.priors_df, self.stations_df)                
        for idx in range(self.cv):
            # copy the new data for each fold
            X_ = X.copy()
            if len(self.preprocessors) > 0:
                preprocessor = self.preprocessors[f'fold_{idx}']
                X_ = preprocessor.transform(X_)
                cols_ = preprocessor.get_feature_names_out()
                cols = [col.split('__')[-1] for col in cols_]
                X_ = pd.DataFrame(X_, columns = cols)
            X_ = self.prep_data_for_model(X_)
            for i in range(self.n_models):
                model = self.models[f'fold_{idx}_model_{i}']
                pred = np.clip(model.predict(X_), 0, np.inf)
                results[f'fold_{idx}_model_{i}'] = pred
                
        # aggregate by method.
        results_ = np.stack(list(results.values()), axis = 1)
        delta = X.cluster.astype(str).map(self.delta).values.reshape(-1, 1)
        if self.method == 'indirect':
            mean_result = results_.mean(axis = 1) #np.median(results_, axis = 1) #
            q_1 = np.quantile(results_, q = self.alphas[0], axis = 1) - (delta * mean_result)
            q_3 = np.quantile(results_, q = self.alphas[1], axis = 1) + (delta * mean_result)
            out = {'pred' : mean_result, 'inf' : q_1, 'sup' : q_3}
        elif self.method == 'direct':
            results_ = results_.reshape(X.shape[0], self.cv, 3, -1)
            results_ = results_.mean(axis = 1)
            mean_pred = results_[:, 1, :]
            q_1 = results_[:, 0, :] - (delta * mean_pred)
            q_3 = results_[:, 2, :] + (delta * mean_pred)
            out = {'pred': mean_pred, 'inf': q_1, 'sup': q_3}
        return out, results_

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        # save metadata
        metadata = {
            'method': self.method,
            'n_models': self.n_models,
            'alphas': self.alphas,
            'cv': list(self.preprocessors.keys()),
            'cat_features': self.cat_features,
            'exclude_cols': self.exclude_cols,
            'enforce_location': self.enforce_location,
            'location_column': self.location_column,
            'random_state': self.random_state,
            'bootstrap': self.bootstrap,
            'model_type': self.model_type,
            'feature_names': self.feature_names_,
            'results': self.results,
            'delta': self.delta,
            'patience': self.patience,
            'min_patience': self.min_patience,
            'split_type': self.split_type,
            'chained': self.chained,
            'k': self.k,
            'km_columns': self.km_columns
        }
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        self.priors_df.to_csv(os.path.join(save_path, 'priors.csv'), index = False)
        self.stations_df.to_csv(os.path.join(save_path, 'stations.csv'), index = False)
    
        # save each model
        for name, model in self.models.items():
            model_path = os.path.join(save_path, name)
            model.save_model(model_path)
    
        # save each preprocessor
        for name, preprocessor in self.preprocessors.items():
            joblib.dump(preprocessor, os.path.join(save_path, f'{name}_preprocessor.pkl'))

        # save kmeans object
        joblib.dump(self.km, os.path.join(save_path, 'kmeans.pkl'))

    def load_model(self, load_path, model_fn, model_params):
         # Load metadata
        with open(os.path.join(load_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
    
        self.method = metadata['method']
        self.n_models = metadata['n_models']
        self.alphas = metadata['alphas']
        self.cat_features = metadata['cat_features']
        self.exclude_cols = metadata['exclude_cols']
        self.random_state = metadata['random_state']
        self.bootstrap = metadata['bootstrap']
        self.model_type = metadata['model_type']
        self.feature_names_ = metadata['feature_names']
        self.results = metadata['results']
        self.delta = metadata['delta']
        self.patience = metadata['patience']
        self.min_patience = metadata['min_patience']
        self.split_type = metadata['split_type']

        self.model_fn = model_fn
        self.model_params = model_params
        self.chained = metadata['chained']
        self.k = metadata['k']
        self.km_columns = metadata['km_columns']
        
        self.preprocessors = {}
        self.models = {}

        self.stations_df = pd.read_csv(os.path.join(load_path, 'stations.csv'))
        self.priors_df = pd.read_csv(os.path.join(load_path, 'priors.csv'))

        # Load models
        for fold in metadata['cv']:
            for i in range(self.n_models):
                model_name = f'{fold}_model_{i}'
                model_path = os.path.join(load_path, model_name)
                model = self.build_model(self.model_type, random_state = self.random_state)
                cbm = model.__class__
                base_estimator = self.get_gbt_estimator()
                model = cbm.load_model(model_path, base_estimator)
                self.models[model_name] = model
    
            # Load preprocessor
            preproc_path = os.path.join(load_path, f'{fold}_preprocessor.pkl')
            self.preprocessors[fold] = joblib.load(preproc_path)

        # load kmeans object
        self.km = joblib.load(os.path.join(load_path, 'kmeans.pkl'))
                    
    def prep_data_for_model(self, X):
        # for deep learning convert categorical and continuous into a list of inputs
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].astype('object').astype('category')
        return X

    def build_model(
        self, 
        model_type = 'catboost', 
        loss_function = 'MAE', 
        eval_metric = 'MAE', 
        lr = 0.3, 
        od_wait = 10, 
        feature_weights = None, 
        random_state = 42
    ):
        model_params = self.model_params.copy()
        if model_type == 'catboost':
            model_params.update({
                'loss_function': loss_function, 
                'eval_metric': eval_metric, 
                'cat_features': self.cat_features,
                'random_state': random_state,
                'od_wait': od_wait,
                'learning_rate': lr,
                'feature_weights' : feature_weights
            })
        elif model_type == 'lightgbm':
            raise NotImplementedError
        elif model_type == 'xgboost':
            raise NotImplementedError
        else:
            raise ValueError('Unsupported model type')
        model = self.model_fn(self.model_type, model_params, chained = self.chained)
        return model

    def _resample(self, X, y, random_state = 42):
        if self.bootstrap:
            np.random.seed(random_state)
            idx = np.random.choice(X.index, len(X), replace = True)
            X_ = X.loc[idx, :]
            y_ = y.loc[idx, :]
        else:
            X_ = X.copy()
            y_ = y.copy()
        return X_, y_

    def get_gbt_estimator(self):
        if self.model_type == 'catboost':
            model = cb.CatBoostRegressor
        elif self.model_type == 'lightgbm':
            raise NotImplementedError
        elif self.model_type == 'xgboost':
            raise NotImplementError
        else:
            raise ValueError('Unsupported model type')
        return model

    def get_feature_names_(self):
        return self.feature_names_

    def get_feature_importances_(self):
        # Collect all importances into a list
        importances = [
            model.get_feature_importance()
            for fold_model in self.models.values()
            for model in fold_model.models_
        ]

        importances = [i[:len(self.feature_names_)] for i in importances]
        mean_importance = np.mean(importances, axis=0)
        feature_names = self.get_feature_names_()
    
        return pd.DataFrame({
            'features': feature_names,
            'importances': mean_importance
        }).sort_values(by = 'importances', ascending = False).reset_index(drop = True)
    
    def _compute_empirical_priors(self, X, y, target_col = None):
        assert 'week' in list(X.columns), 'X must contain week column'
        assert 'year' in list(X.columns), 'X must contain year column'
        assert 'station_code' in list(X.columns), 'X must contain station_code column'

        # join target and data
        tmp = pd.concat([X, y], axis = 1)

        # compute data pivot
        tmp = tmp[['station_code', 'year', 'week', target_col]]
        tmp = tmp.pivot_table(index = 'week', columns = ['station_code', 'year'], values = target_col).groupby(level=0, axis=1).mean()
        tmp = tmp.reset_index()
        tmp.columns = tmp.columns.rename('')
        return tmp

    def _compute_train_empirical_priors(self, X, y, target_col = None):
        assert 'week' in list(X.columns), 'X must contain week column'
        assert 'year' in list(X.columns), 'X must contain year column'
        assert 'station_code' in list(X.columns), 'X must contain station_code column'
        
        empirical_data = []

        # join target and data
        X = X.copy()
        X_ = pd.concat([X, y], axis = 1)
        stations = X_.station_code.unique()
        for station in stations:
            tmp = X_[X_.station_code == station][['year', 'week', 'river', target_col]].reset_index(drop = True)
            river = tmp.river.unique()[0]
            tmp = tmp.pivot_table(index = 'year', columns = 'week', values = target_col)   
            for year in X_.year.unique():
                tmp_ = tmp.drop(year)
                mean_annual_discharge = tmp_.mean(axis = 0)
                tmp_ = pd.DataFrame(mean_annual_discharge, columns = ['empirical_flow'])
                tmp_['year'] = year
                tmp_['station_code'] = station
                tmp_ = tmp_.reset_index().rename(columns = {'index':'week'})
                empirical_data.append(tmp_)
        emp = pd.concat(empirical_data, axis = 0)
        X = X.merge(emp, on = ['station_code', 'year', 'week'], how = 'left')
        return X
        
    def _get_empirical_priors(self, X, priors, stations_gdf):
        assert 'week' in list(X.columns), 'X must contain week column'
        assert 'year' in list(X.columns), 'X must contain year column'
        assert 'station_code' in list(X.columns), 'X must contain station_code column'

        X = X.copy()
        priors.columns = [str(i) for i in priors.columns]
        
        emp_ = []
        for station in X.station_code.unique():
            if station in list(priors.columns):
                data = priors[['week', station]]
                data = data.rename(columns = {station: 'empirical_flow'})
                data['station_code'] = station
            else:
                # get nearest stations by river
                tmp = X[X.station_code == station]
                river = tmp.river.unique()[0]
                rank = tmp.river_ranking.unique()[0]
                location = tmp.location.unique()[0]
                coords = tmp[['latitude', 'longitude']].drop_duplicates()
                # get is same river & is same rank
                # is same river works because the train stations are also present in the eval set
                is_same_river = stations_gdf[stations_gdf.river == river].station_code
                if location == 'france':
                    is_same_rank = stations_gdf[(stations_gdf.river_ranking.isin([rank, rank+1])) & (stations_gdf.location == location)].station_code
                elif location == 'brazil':
                    is_same_rank = stations_gdf[(stations_gdf.river_ranking.isin([rank])) & (stations_gdf.location == location)].station_code

                # get neighbors list
                neighbors_list = np.unique(np.concatenate([is_same_river, is_same_rank]))
                
                # get neighbors coords
                neighbors_df = stations_gdf[stations_gdf.station_code.isin(neighbors_list)][['latitude', 'longitude', 'river', 'station_code']]

                # compute nearest neighbors and get 2 nearest stations
                nn = NearestNeighbors(n_neighbors=2, radius = 1, metric='euclidean')
                nn.fit(neighbors_df[['latitude', 'longitude']])

                # get distances and neighbor indices
                distances, indices = nn.kneighbors(coords)
                neighbors = neighbors_df.iloc[indices[0]]

                # compute inverse distance for weighting
                inv_distance = 1 / np.maximum(distances[0], 1e-6)**1
                weights = inv_distance / inv_distance.sum()
                neighbors['weights'] = weights

                data = priors.set_index('week')[neighbors.station_code.astype(str).values]* neighbors.weights.values
                data['empirical_flow'] = data.mean(axis = 1)
                data = data.reset_index()[['week', 'empirical_flow']]
                data['station_code'] = station

            emp_.append(data)
        emp_ = pd.concat(emp_)
        emp_['week'] = emp_.week.astype(str)
        X = X.merge(emp_, on = ['station_code', 'week'], how = 'left')
        return X

    def _compute_interactions(self, X, col1, col2):
        X[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]       

    def _get_feature_weights(self, location_feature, columns):
        return [200 if i == location_feature else 1 for i in columns]