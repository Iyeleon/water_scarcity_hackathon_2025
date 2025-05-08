import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgbm
from utils import load_config
from sklearn.feature_selection import SelectFromModel, mutual_info_regression

if __name__ == '__main__':
    # get config and data
    config = load_config()

    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_features', type = int,  help = 'Number of features to select', default = 20, required = False)
    args = parser.parse_args()
    
    DATA_DIR = config['final_data']
    TRAIN = os.path.join(DATA_DIR, 'train.csv')

    # get vars
    CATEGORICAL = ['river', 'location', 'month', 'week', 'season', 'station_code']
    COLS_TO_DROP = ['ObsDate', 'catchment', 'hydro_region', 'hydro_sector', 'hydro_sub_sector', 'hydro_zone', 'region_sector', 
    'region_sub_sector', 'region_zone', 'sector_sub_sector', 'sector_zone', 'sub_sector_zone']
    TARGET_COLS = ['water_flow_week_1', 'water_flow_week_2', 'water_flow_week_3', 'water_flow_week_4']
    NUM_SOIL = ['bdod', 'cfvo', 'clay', 'sand']
    NUM_METEO = ['tp', 't2m', 'swvl1', 'evap']
    FEATURE_GROUPS = {'soil_features': NUM_SOIL, 'meteo_features': NUM_METEO}

    # get data
    df = pd.read_csv(TRAIN)
    X_ = df.drop(columns = TARGET_COLS + COLS_TO_DROP + CATEGORICAL, errors = 'ignore')
    y_= df.water_flow_week_1

    # start feature selection
    # 1 - Mutual Information Regression (MIR) to select top correlated features 
    # 1b - MIR on two feature groups (meteo features and soil features)
    # 2 - Select best features from lightgbm feature importances
    print('Selecting most correlated features with target using mutual information regression')
    selected_features = []
    # loop and get features per feature group
    for feature_id, feature_group in FEATURE_GROUPS.items():
        print(f'Selecting features from {feature_id} ..')
        XX = X_.filter(regex = '|'.join(feature_group))
        features = np.array(XX.columns)
        mi = mutual_info_regression(XX, y_)
        select_idx = np.where(mi > np.quantile(mi, 0.5))[0]
        feature_imp = mi[select_idx]
        features = features[select_idx]
        features_ranked = sorted(zip(features, feature_imp), key = lambda x: x[1], reverse = True)
        selected_features.extend(features_ranked)
        
    # compile selected features
    selected_features_df = pd.DataFrame(selected_features, columns = ['feature', 'y_corr'])
    new_features = selected_features_df.feature.tolist()
    new_features = new_features + [i for i in X_.columns if 'water_flow' in i]
    X_new = X_[new_features]

    # lgbm feature importance
    print('Reducing selected features with lgbm importances.')
    np.random.seed(0) # set seed for reproducibility
    reg = lgbm.LGBMRegressor(random_state = 42, verbose = 0)
    sfm = SelectFromModel(reg, threshold="median", max_features = args.num_features)
    sfm.fit(X_new, y_)

    # compile and save final selected features
    selected_features = pd.DataFrame({'features': sorted(sfm.get_feature_names_out())})
    selected_features.to_csv(os.path.join(DATA_DIR, 'selected_features.csv'), index = False)
    print(selected_features)
    
    
    
    