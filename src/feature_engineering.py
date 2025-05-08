import os
import tqdm
from itertools import combinations

import numpy as np
import pandas as pd

from utils import load_config
# from src.helpers import ContiguousGroupKFold

def encode_location(lat, lon):
    """Sinusoidal encoding of location coordinates
    Parameters
    ----------
    lat: float
        Latitude
    lon: float
        Longitude

    Returns
    -------
    pd.Series: A series containing (sin(lat), cos(lat), sin(lon), cos(lon))
    """
    lat_rad = np.radians(lat)  # Convert to radians
    lon_rad = np.radians(lon)
    
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)
    lon_sin = np.sin(lon_rad)
    lon_cos = np.cos(lon_rad)
    
    location_encoding = pd.Series([lat_sin, lat_cos, lon_sin, lon_cos])
    return location_encoding

def main(is_mini = False):    
    # load config
    config = config = load_config()
    
    # get constants
    PREPROCESSED_DIR = config['preprocessed_data']
    FINAL_DIR = config['final_data']

    BRAZIL_TRAIN = os.path.join(PREPROCESSED_DIR, 'preprocessed_brazil_train.csv')
    BRAZIL_EVAL = os.path.join(PREPROCESSED_DIR, 'preprocessed_brazil_eval.csv')
    FRANCE_TRAIN = os.path.join(PREPROCESSED_DIR, 'preprocessed_france_train.csv')
    FRANCE_EVAL = os.path.join(PREPROCESSED_DIR, 'preprocessed_france_eval.csv')
    MINI_EVAL = os.path.join(PREPROCESSED_DIR, 'preprocessed_mini_brazil_eval.csv')

    # compile into dict
    datasets = {
        'tr_brazil' : BRAZIL_TRAIN,
        'tr_france' : FRANCE_TRAIN,
        'ev_brazil' : BRAZIL_EVAL,
        'ev_france' : FRANCE_EVAL,
        'ev_mini' : MINI_EVAL
    }
    
    # variable groups
    LOCATION = ['longitude', 'latitude']
    CATEGORICAL = ['station_code', 'river', 'hydro_region', 'hydro_sector', 'hydro_sub_sector', 'hydro_zone']
    NUM_STATION = ['altitude', 'catchment']
    NUM_SOIL = ['bdod', 'cfvo', 'clay', 'sand']
    NUM_METEO = ['tp', 't2m', 'swvl1', 'evap']
    DISCHARGE = 'discharge'
    DATE = 'ObsDate'
    
    # load the data
    df_ = {}
    for key, directory in datasets.items():
        df_[key] = pd.read_csv(directory)

    # remove uncommon features - features that are not present in train and eval sets.
    diff_col = set(df_['tr_brazil'].columns).symmetric_difference(set(df_['tr_france'].columns))
    for key in df_.keys():
        df_[key] = df_[key].drop(columns = diff_col, errors = 'ignore')

    # drop irrelevant columns
    COLS_TO_DROP = ['station_name', 'id_region', 'id_sector', 'id_sub_sector', 'id_zone','geometry', 'eval_only']
    for key in df_.keys():
        df_[key] = df_[key].drop(columns = COLS_TO_DROP, errors = 'ignore')

    # create temporal features
    for key in df_.keys():
        # extract df
        temp = df_[key]
        
        # convert to datetime and temporal features
        temp['ObsDate'] = pd.to_datetime(temp.ObsDate)
        temp['year'] = temp.ObsDate.dt.year
        temp['month'] = temp.ObsDate.dt.month
        temp['week'] = temp.ObsDate.dt.isocalendar().week
        temp['week'] = temp.apply(lambda x: x.week if x.year not in [1993, 1997, 2001, 2005] else x.week + 1, axis = 1)
        temp['week'] = temp.week.apply(lambda x: 1 if x > 52 else x)
    
        # compute new features 
        # season - winter: 1, spring - 2, summer - 3, autumn - 4
        temp['season'] = temp.month.apply(lambda x: (x-1) // 3 + 1) # 1, 2, 3, 4
    
        # seasonality patterns
        # repeated gaussian patterns to mirror seasonality found in data
        # compute repeating gaussians with varying widths(sigmas)
        sigmas = [4, 8]
        for sigma in sigmas:
            temp[f'gaussian_{sigma}'] =  np.exp(-(((temp.week + 23) % 52 - 52/2) ** 2) / (2 * sigma ** 2))
            
        # enso patterns - model long term trends related to climate patterns 
        # enso_spans = [3, 4, 5, 6, 7]
        # for span in enso_spans:
        #     temp[f'enso_{span}_sin'] = np.sin(2 * np.pi * ((temp.year + temp.week)/52)/ span)
        #     temp[f'enso_{span}_cos'] = np.sin(2 * np.pi * ((temp.year + temp.week)/52)/ span)
    
        # convert to categorical
        # map month to months
        month_dict = {1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'}
        seasons_dict = {1:'winter', 2:'spring', 3:'summer', 4:'autumn'}
        # temp['year'] = temp.year.astype('category')
        temp['month'] = temp.month.map(month_dict).astype('category')
        # temp['week'] = temp.week.apply(lambda x: f'w{x}').astype('category')
        temp['season'] = temp.season.map(seasons_dict).astype('category')

    # convert categorical features
    hydro_scale_combinations = list(combinations(['region', 'sector', 'sub_sector', 'zone'], 2))
    for key in df_.keys():
        temp = df_[key]
        
        # create hydro feature interaction
        for comb in hydro_scale_combinations:
            col = comb[0] + '_' + comb[1]
            temp[col] = temp.apply(lambda x: str(x[f'hydro_{comb[0]}']) + '_' + str(x[f'hydro_{comb[1]}']), axis = 1).astype('category')
            
        # convert other categorical vars to category
        for col in CATEGORICAL:
            temp[col] = temp[col].astype('category')

    # location encoding
    for key in df_.keys():
        temp = df_[key]
        temp[['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos']] = temp.apply(lambda x: encode_location(x.latitude, x.longitude), axis = 1)

    # combine data from the different datasets for train and eval.
    # assign country
    for key in df_.keys():
        location = key.split('_')[-1]
        dataset = key.split('_')[0]
        if location == 'mini':
            location = 'brazil'
            dataset = 'mini'
        df_[key]['location'] = location
        if dataset == 'tr':
            continue
        df_[key]['dataset'] = dataset

    # combine datasets
    df_train = pd.concat([df_['tr_brazil'], df_['tr_france']], ignore_index = True)
    df_eval = pd.concat([df_['ev_brazil'], df_['ev_france'], df_['ev_mini']], ignore_index = True)

    # sort by station and date
    df_train = df_train.sort_values(['station_code', 'ObsDate'])
    df_eval = df_eval.sort_values(['station_code', 'ObsDate'])

    # create discharge rolling and lag features
    for df in [df_train, df_eval]:
        df['water_flow_lag_1w'] = df.groupby('station_code').discharge.transform(lambda x: x.shift(1).bfill())
        df['water_flow_lag_2w'] = df.groupby('station_code').discharge.transform(lambda x: x.shift(2).bfill())
        df['water_flow_lag_3w'] = df.groupby('station_code').discharge.transform(lambda x: x.shift(3).bfill())
        df['water_flow_lag_4w'] = df.groupby('station_code').discharge.transform(lambda x: x.shift(4).bfill())
        df['water_flow_rolling_mean_4w'] = df.groupby('station_code').discharge.transform(lambda x: x.shift(1).rolling(4).mean().bfill())
        df['water_flow_rolling_std_4w'] = df.groupby('station_code').discharge.transform(lambda x: x.shift(1).rolling(4).std().bfill())
        df['water_flow_rolling_min_4w'] = df.groupby('station_code').discharge.transform(lambda x: x.shift(1).rolling(4).min().bfill())
        df['water_flow_rolling_max_4w'] = df.groupby('station_code').discharge.transform(lambda x: x.shift(1).rolling(4).max().bfill())
        df['water_flow_trend_4w'] = (
            df.groupby('station_code').discharge.transform(lambda x: x.shift(1).bfill()) - df.groupby('station_code').discharge.transform(lambda x: x.shift(4)).bfill()
        )

    # create meteo interaction, rolling and lag features
    for df in [df_train, df_eval]:
        df['tp_evap_diff'] = df.tp + df.evap
        for col in df.filter(regex = '|'.join(NUM_METEO)).columns:
            if 'tp' in col or 'evap' in col :
                df[f'{col}_rolling_sum_3w'] = df.groupby('station_code')[col].transform(lambda x: x.rolling(3).sum().bfill())
            elif 't2m' in col or 'swvl1' in col:
                df[f'{col}_rolling_mean_3w'] = df.groupby('station_code')[col].transform(lambda x: x.rolling(3).mean().bfill())
            df[f'{col}_lag_1w'] = df.groupby('station_code')[col].transform(lambda x: x.shift(1).bfill().values)

    # create target vars
    df_train['water_flow_week_1'] = df_train.discharge
    df_train['water_flow_week_2'] = df_train.groupby('station_code').discharge.transform(lambda x: x.shift(-1))
    df_train['water_flow_week_3'] = df_train.groupby('station_code').discharge.transform(lambda x: x.shift(-2))
    df_train['water_flow_week_4'] = df_train.groupby('station_code').discharge.transform(lambda x: x.shift(-3))
    df_train = df_train.drop(columns = ['discharge'])

    # keep inference rows
    missing_values = df_eval["discharge"].isnull()
    df_eval = df_eval[missing_values]
    df_eval = df_eval.drop(columns = 'discharge', errors = 'ignore')

    # handle missing data
    train_cols_nan = df_train[df_train.columns[df_train.isna().any()].tolist()]
    for col in train_cols_nan:
        df_train[col] = df_train.groupby('location')[col].transform(lambda x:x.fillna(x.mean()))
    eval_cols_nan = df_eval[df_eval.columns[df_eval.isna().any()].tolist()]
    for col in eval_cols_nan:
        df_eval[col] = df_eval.groupby('location')[col].transform(lambda x: x.fillna(x.mean()))

    # make final folder
    os.makedirs(FINAL_DIR, exist_ok = True)

    # save train and eval
    train_name = os.path.join(FINAL_DIR, 'train.csv')
    eval_name = os.path.join(FINAL_DIR, 'eval.csv')

    df_train.to_csv(train_name, index = False)
    df_eval.to_csv(eval_name, index = False)

if __name__ == '__main__':
    main()