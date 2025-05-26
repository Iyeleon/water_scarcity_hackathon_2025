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
    parser.add_argument('-n', '--num_features', type = int,  help = 'Number of meteo features to select', default = 30, required = False)
    parser.add_argument('-p', '--pcnt_threshold', type = float,  help = 'Percentile threshold to filter soil features', default = 0.9, required = False)
    parser.add_argument('-c', '--corr_threshold', type = float,  help = 'Correlation threshold to filter soil features', default = 0.9, required = False)
    args = parser.parse_args()
    
    DATA_DIR = config['final_data']
    TRAIN = os.path.join(DATA_DIR, 'train.csv')

    # get vars
    CATEGORICAL = ['river', 'location', 'month', 'week', 'season', 'station_code']
    COLS_TO_DROP = ['ObsDate', 'catchment', 'hydro_region', 'hydro_sector', 'hydro_sub_sector', 'hydro_zone', 'region_sector', 
    'region_sub_sector', 'region_zone', 'sector_sub_sector', 'sector_zone', 'sub_sector_zone']
    TARGET_COLS = ['water_flow_week_1', 'water_flow_week_2', 'water_flow_week_3', 'water_flow_week_4']
    NUM_SOIL = ['bdod', 'cfvo', 'clay', 'sand']
    NUM_METEO = ['tp', 't2m', 'swvl1'] # select only the three groups from causal analysis
    
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

    # GET MOST PREDICTIVE SOIL FEATURES
    print(f'Selecting features from soil features ..')
    # STEP 1: Mutual Information Regression (MIR)
    X_soil = X_.filter(regex = '|'.join(NUM_SOIL))
    mi_scores = mutual_info_regression(X_soil, y_)
    mi_series = pd.Series(mi_scores, index=X_soil.columns)
    
    # Select features in the 95th percentile
    threshold_percentile = args.pcnt_threshold
    mi_threshold = np.percentile(mi_scores, threshold_percentile)

    # Get selected features
    mi_selected_features = mi_series[mi_series >= mi_threshold].index
    X_mi_selected = X_[mi_selected_features]
    
    # Drop Highly Correlated Features
    # Compute correlation matrix (absolute values)
    corr_matrix = X_mi_selected.corr().abs()
    
    # Get upper triangle of the matrix (to avoid duplicate checks)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns with correlation above the threshold (e.g., 0.9)
    corr_threshold = args.corr_threshold
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    
    # Final feature set after dropping correlated features
 
    soil_features = pd.DataFrame({'features': mi_selected_featues.drop(to_drop).index.tolist()})
    soil_features.to_csv(os.path.join(DATA_DIR, 'selected_soil_features.csv'), index = False)
    print('Soil Features:', soil_features)

        
    X_meteo = X_.filter(regex = '|'.join(NUM_METEO))
    # lgbm feature importance
    print('Reducing selected features with lgbm importances.')
    np.random.seed(0) # set seed for reproducibility
    reg = lgbm.LGBMRegressor(random_state = 42, verbose = 0)
    sfm = SelectFromModel(reg, threshold="median", max_features = args.num_features)
    sfm.fit(X_meteo, y_)

    # compile and save final selected features
    selected_features = pd.DataFrame({'features': sorted(sfm.get_feature_names_out())})
    selected_features.to_csv(os.path.join(DATA_DIR, 'selected_meteo_features.csv'), index = False)
    print('Meteo Features:', selected_features)
    
    
    
    