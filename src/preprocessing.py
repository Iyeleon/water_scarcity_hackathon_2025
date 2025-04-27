import os
import gc
import tqdm
import argparse
from collections import defaultdict
from utils import load_config
import xarray as xr
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

import pandas as pd
import geopandas as gpd
from rioxarray.exceptions import NoDataInBounds 
from src.helpers.data_loader import (
    load_station_info,
    load_hydro_data,
    load_water_flows,
    load_meteo_data,
    read_soil_data,
    read_altitude_data
)  

def get_stations(area, data_dir, crs = 'epsg:4326'):
    # get train and eval stations
    stations_train = load_station_info(area, 'train', data_dir, crs = crs)
    stations_eval = load_station_info(area, 'eval', data_dir, crs = crs)
    # eval has more stations than train. label stations present in eval only or both
    # common stations appear in both data
    common_stations = set(stations_train.station_code).intersection(stations_eval.station_code)
    stations_train['eval_only'] = False
    stations_eval['eval_only'] = stations_eval.station_code.apply(lambda x: False if x in common_stations else True)
    return stations_eval
    
def aggregate_hydro_data(hydro_data, join_info, stations, crs = 'epsg:4326'):        
    # loop through and update filter
    stations_ = stations.copy()
    for level, (col, lsuffix, rsuffix) in join_info.items():
        stations_ = gpd.sjoin(
            stations_,
            hydro_data[level][['geometry', col]],
            how = "left",
            predicate = "within",
            lsuffix = lsuffix,
            rsuffix = rsuffix
        ).rename(columns = {col:f'hydro{rsuffix}', f'index_{rsuffix}': f'id{rsuffix}'})

    for geo_scale in hydro_data.keys():
        hydro_data[geo_scale] = hydro_data[geo_scale].iloc[list(stations_[f'id_{geo_scale}'].unique())]
    
    # return filtered hydro data and stations list
    return stations_, hydro_data
    
def get_altitude(lat: float, lon: float, dem: xr.DataArray) -> float:
    """
    Get the altitude for a given latitude and longitude using
    a digital elevation model (DEM).

    Args:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        dem (xr.DataArray): Digital elevation model.

    Returns:
        float: The altitude at the given coordinates.
    """
    try:
        return dem.sel(x=lon, y=lat, method='nearest').values.item()
    except NoDataInBounds:
        return np.nan
        
def get_watercourse_data(area, data_dir):
    # load watercourse
    # get filename
    if area == 'brazil':
        filename = 'geoft_bho_2017_curso_dagua.gpkg'
    elif area == 'france':
        filename = 'CoursEau_FXX.shp'
    else:
        raise ValueError('Invalid area. Must be one of france or brazil')
    watercourse_path = os.path.join(data_dir, area, 'static_data', 'watercourse', filename)
    watercourse = gpd.read_file(watercourse_path)
    return watercourse

def get_station_watercourse_ranking(area, watercourse, stations_gdf):
    # get stations bound
    # clip watercourse gdf
    if area == 'brazil':
        ranking_col = 'nunivotcda'
    elif area == 'france':
        ranking_col = 'Classe'
    else:
        raise ValueError('Invalid area. Must be one of france or brazil')

    # clip to stations extent
    buffer = stations_gdf.geometry.total_bounds
    watercourse = gpd.clip(watercourse, buffer)
    if area == 'brazil':
        watercourse = watercourse[watercourse[ranking_col].isin([3, 4, 5])]
        watercourse[ranking_col] = watercourse[ranking_col] - 2
    elif area == 'france':
        watercourse = watercourse[watercourse[ranking_col].isin([1, 2, 3, 4])]

    # do spatial join with buffered station gdf
    stations_ = gpd.GeoDataFrame(stations_gdf, geometry = stations_gdf.buffer(0.0015))
    ranking = watercourse.to_crs(stations_.crs).sjoin(stations_, predicate = 'intersects', how = 'right')
    ranking = ranking.dropna(subset = ranking_col).drop_duplicates(subset = 'river')
    ranking = ranking[['river', ranking_col]].rename(columns = {ranking_col : 'river_ranking'})
    stations_gdf = stations_gdf.merge(ranking, on = 'river', how = 'left')
    return stations_gdf

def get_join_info(area):
    if area == "france":
            join_info = {
                "region":    ("CdRegionHydro",      "_stations",      "_region"),
                "sector":    ("CdSecteurHydro",     "_stations",      "_sector"),
                "sub_sector":("CdSousSecteurHydro", "_stations",      "_sub_sector"),
                "zone":      ("CdZoneHydro",        "_stations",      "_zone")
            }
    elif area == "brazil":
        join_info = {
            "region":    ("wts_pk", "_stations_region", "_region"),
            "sector":    ("wts_pk", "_stations_sector", "_sector"),
            "sub_sector":("wts_pk", "_stations_sub_sector", "_sub_sector"),
            "zone":      ("wts_pk", "_stations_zone", "_zone")
        }
    else:
        raise ValueError('area must be one of brazil or france')
    return join_info

def aggregate_meteo_data(data, stations_gdf, max_date, hydro_data, buffer_scales = None, crs = 'epsg:4326'):
    # define output df
    df = None

    # establish key variables
    key_vars = {
        'precipitations' : 'tp', 
        'temperatures' : 't2m',
        'soil_moisture' : 'swvl1',
        'evaporation' : 'evap'
    }

    # loop through and aggregate at different scales for the four variables
    # loop 1 - variable loop
    for key, var in key_vars.items():
        keys_list = []
        data_ = data[key]
        if var == 'evap' and 'e' in data_.data_vars:
            data_ = data_.rename({'e': 'evap'})
        data_ = data_[var]  # extract variable
        data_ = data_.rio.write_crs(crs) #project to crs
        # loop 2 - stations loop
        for idx, row in stations_gdf.iterrows():
            lat, lon = row.geometry.y, row.geometry.x
            # sample a single point (each point is sampled at 0.25 degrees approximately at 27km)
            sampled_values = data_.sel(latitude=lat, longitude=lon, method="nearest").to_dataframe().reset_index()
            # Filter by date range
            sampled_values = sampled_values[sampled_values.valid_time <= max_date]
            sampled_values["station_code"] = row.station_code
            sampled_values = sampled_values[['station_code', 'valid_time', var]]
            # loop 3 - sample at different buffer scales
            for buffer in buffer_scales:
                # select data within buffer
                geom = row.geometry.buffer(buffer / 111) # convert kilometer to degrees
                clipped_data = data_.rio.clip([geom])
                buffer_values = clipped_data.mean(dim = ['latitude', 'longitude'])
                buffer_values = buffer_values.to_dataframe().reset_index()
                buffer_values = buffer_values[buffer_values.valid_time <= max_date]
                buffer_values['station_code'] = buffer_values["station_code"] = row.station_code
                buffer_values = buffer_values.rename(columns={var: f'{var}_{buffer}km'})
    
                # Merge with point values using date
                sampled_values = sampled_values.merge(
                    buffer_values[["valid_time", 'station_code', f'{var}_{buffer}km']],
                    on=['station_code', "valid_time"],
                    how="left"
                )
            for geo_scale in hydro_data.keys():
                hydro_ = hydro_data[geo_scale]
                geom = [hydro_.loc[row[f'id_{geo_scale}']].geometry] 
                if isinstance(geom, MultiPolygon):
                    geom = list(geom.geoms)
                try:
                    clipped_data = data_.rio.clip(geom)
                    buffer_values = clipped_data.mean(dim = ['latitude', 'longitude'])
                    buffer_values = buffer_values.to_dataframe().reset_index()
                    buffer_values = buffer_values[buffer_values.valid_time <= max_date]
                except NoDataInBounds:
                    buffer_values = pd.DataFrame(data = {'valid_time': sampled_values.valid_time, f'{var}': np.nan})             
                
                buffer_values['station_code'] = row.station_code
                buffer_values = buffer_values.rename(columns={var: f'{var}_{geo_scale}'})
    
                # Merge with point values using date
                sampled_values = sampled_values.merge(
                    buffer_values[["valid_time", 'station_code', f'{var}_{geo_scale}']],
                    on=['station_code', "valid_time"],
                    how="left"
                )
                    
            keys_list.append(sampled_values)

        # Combine all DataFrames
        df_keys = pd.concat(keys_list, ignore_index=True)
        if not isinstance(df, pd.DataFrame):
            df = df_keys
        else:
            df = df.merge(
                df_keys,
                on = ['station_code', "valid_time"],
                how = 'left'
            )
    return df

def load_and_aggregate_meteo_data(area, data_dir, key, stations_gdf, max_date, hydro_data, buffer_scales = None, crs = 'epsg:4326'):
    # load meteo data
    meteo_data = load_meteo_data(area, key, data_dir)

    meteo_data = aggregate_meteo_data(
        data = meteo_data, 
        stations_gdf = stations_gdf, 
        max_date = max_date, 
        hydro_data = hydro_data,
        buffer_scales = buffer_scales, 
        crs = crs
    )
    
    return meteo_data

def aggregate_soil_data(data, stations_gdf, hydro_data, buffer_scales = None):
    """ aggregates the data at specified buffer scale. 
    If buffer scale is not provided, it aggregates to local, field and watershed scales
    """
    mean_data = []
    std_data = []
    buffer_mean_cols = [f'{var}_{buffer}km_mean' for buffer in buffer_scales for var in list(data.data_vars)]
    geoscale_mean_cols = [f'{var}_{geo_scale}_mean' for geo_scale in hydro_data.keys() for var in list(data.data_vars)]
    mean_cols = buffer_mean_cols + geoscale_mean_cols
    
    buffer_std_cols = [f'{var}_{buffer}km_std' for buffer in buffer_scales for var in list(data.data_vars)]
    geoscale_std_cols = [f'{var}_{geo_scale}_std' for geo_scale in hydro_data.keys() for var in list(data.data_vars)]    
    std_cols = buffer_std_cols + geoscale_std_cols

    # collect mean and standard deviation of of buffer and region aggregated variables per row.
    mean_data = []
    val_data = []
    for idx, row in tqdm.tqdm(stations_gdf.iterrows(), total = len(stations_gdf)):
        means = []
        stds = []
        for buffer in buffer_scales: # (km)
            geom = row.geometry.buffer(buffer / 111)
            for var in list(data.data_vars):
                try:
                    clipped_data = data[var].rio.clip([geom], stations_gdf.crs)
                    mean_val = float(clipped_data.mean().values)
                    std_val = float(clipped_data.std().values)
                except NoDataInBounds:
                    mean_val = np.nan
                    std_val = np.nan
                    print(f"No data in bounds for {var}_{buffer}km")
                # causes repetitive frame insert warning
                # stations_gdf.loc[idx, f"{var}_{buffer}km_mean"] = mean_val
                # stations_gdf.loc[idx, f"{var}_{buffer}km_std"] = std_val
                means.append(mean_val)
                stds.append(std_val)

        for geo_scale in hydro_data.keys():
            hydro_ = hydro_data[geo_scale]
            geom = [hydro_.loc[row[f'id_{geo_scale}']].geometry] 
            if isinstance(geom, MultiPolygon):
                geom = list(geom.geoms)
            for var in list(data.data_vars):
                try:
                    clipped_data = data[var].rio.clip(geom, stations_gdf.crs)
                    mean_val = float(clipped_data.mean().values)
                    std_val = float(clipped_data.std().values)
                except NoDataInBounds:
                    mean_val = np.nan
                    std_val = np.nan
                    print(f"No data in bounds for {var}_{geo_scale}")             
                
                # stations_gdf.loc[idx, f"{var}_{geo_scale}_mean"] = mean_val
                # stations_gdf.loc[idx, f"{var}_{geo_scale}_std"] = std_val
                means.append(mean_val)
                stds.append(std_val)
        mean_data.append(means)
        std_data.append(stds)  

    mean_df = pd.DataFrame(mean_data, columns = mean_cols, index = stations_gdf.index)
    std_df = pd.DataFrame(std_data, columns = std_cols, index = stations_gdf.index)
    stations_gdf = pd.concat([stations_gdf, mean_df, std_df], axis = 1)
                
    return stations_gdf

def merge_all_data(station, waterflow, meteo_data):
    station = station.copy()
    waterflow = waterflow.copy()
    meteo_data = meteo_data.copy()
    meteo_data = meteo_data.rename(columns = {'valid_time': 'ObsDate'})
    merged = meteo_data.merge(station, on = 'station_code', how = 'left')
    merged = merged.merge(waterflow, on = ['station_code', 'ObsDate'], how = 'left')
    return merged
    
def get_hydro_data(area, data_dir, crs):
    # load hydro data
    hydro_data = load_hydro_data(area, data_dir)

    # reproject hydro data to same crs as stations
    for geo_scale in hydro_data.keys():
        hydro_data[geo_scale] = hydro_data[geo_scale].to_crs(crs)

    return hydro_data

def get_soil_data(area, data_dir):
    # load soil data
    soil_data = read_soil_data(area, data_dir) 
    
    # rename data vars to remove area from variable names
    soil_data = soil_data.rename({var: var.replace(f"{area}_", "") for var in soil_data.data_vars})

    return soil_data
    
def process_mini(data_dir, sec_data_dir, soil_buffer = [1, 5, 25], meteo_buffer = [50, 100], crs = 'epsg:4326'):
    area = 'brazil'

    # get dem
    print(f"Loading DEM for {area} mini")
    dem = read_altitude_data(area, data_dir)

    # load stations
    print(f"Loading stations for {area} mini")
    # # Loading mini data station fails due to ; delimeter 
    # # regex delimeter makes france station loading fail
    # # the block of code below modifies the mini challenge delimeter for consistency
    try:
        stations = load_station_info(area, 'eval', sec_data_dir, crs = crs)
    except AttributeError:
        filepath = os.path.join(sec_data_dir, area, 'eval', 'waterflow', 'station_info.csv')
        temp = pd.read_csv(filepath, delimiter = ';')
        temp.to_csv(filepath, index = False)
        stations = load_station_info(area, 'eval', sec_data_dir, crs = crs)
    stations['altitude'] = stations.apply(lambda x: get_altitude(x['latitude'], x['longitude'], dem), axis=1)

    # load hydro data
    print(f"Loading hydro data for {area} mini")
    hydro_data = get_hydro_data(area, data_dir, crs)

    # get join info
    join_info = get_join_info(area)
    
    # aggregate hydro data
    print(f"Aggregating hydro data and merging to stations for {area} mini")
    stations, hydro_data = aggregate_hydro_data(hydro_data, join_info, stations, crs = crs) 

    # get watercourse dat and ranking
    print(f"Loading watercourse data for {area} mini")
    watercourse = get_watercourse_data(area, data_dir)
    stations = get_station_watercourse_ranking(area, watercourse, stations)
    del watercourse
    gc.collect()
    
    # load soil data
    print(f"Loading soil data for {area} mini")
    soil_data = get_soil_data(area, data_dir)

    # aggregate to buffer scales level
    print(f"Aggregating soil data for {area} mini")
    stations = aggregate_soil_data(soil_data, stations_gdf = stations, hydro_data = hydro_data, buffer_scales = soil_buffer)

    # load waterflow data
    print(f"Loading waterflow for {area} mini")
    waterflows = load_water_flows(area, 'eval', sec_data_dir)
    max_date = waterflows.ObsDate.max() + pd.Timedelta(7, unit = 'days') 

    # load and aggregate meteo data
    meteo_data = {}
    print(f"Loading and aggregating meteo data for {area} mini")
    meteo_data = load_and_aggregate_meteo_data(
        area = area, 
        data_dir = sec_data_dir, 
        key = 'eval',
        stations_gdf = stations, 
        max_date = max_date, 
        hydro_data = hydro_data,
        buffer_scales = meteo_buffer, 
        crs = crs
    )
    
    # merge all data
    print(f"Merging all data for {area} mini")
    merged = merge_all_data(stations, waterflows, meteo_data)

    return merged

def process_main(area, data_dir, sec_data_dir, soil_buffer = [1, 5, 25], meteo_buffer = [50, 100], crs = 'epsg:4326'):
    # dem
    print(f"Loading DEM for {area}")
    dem = read_altitude_data(area, data_dir)
    
    # load stations
    print(f"Loading stations for {area}")
    stations = get_stations(area, sec_data_dir, crs = crs)
    stations['altitude'] = stations.apply(lambda x: get_altitude(x['latitude'], x['longitude'], dem), axis=1)

    # load hydro data
    print(f"Loading hydro data for {area}")
    hydro_data = get_hydro_data(area, data_dir, crs)

    # get join info
    join_info = get_join_info(area)
    
    # aggregate hydro data
    print(f"Aggregating hydro data and merging to stations for {area}")
    stations, hydro_data = aggregate_hydro_data(hydro_data, join_info, stations, crs = crs) 

    # get watercourse dat and ranking
    print(f"Loading watercourse data for {area}")
    watercourse = get_watercourse_data(area, data_dir)
    stations = get_station_watercourse_ranking(area, watercourse, stations)
    del watercourse
    gc.collect()

    # load soil data
    print(f"Loading soil data for {area}")
    soil_data = get_soil_data(area, data_dir)

    # aggregate to buffer scales level
    print(f"Aggregating soil data for {area}")
    stations = aggregate_soil_data(soil_data, stations_gdf = stations, hydro_data = hydro_data, buffer_scales = soil_buffer)

    # loading waterflow
    waterflows = {}
    print(f"Loading train and eval waterflow for {area}")
    for key in ['train', 'eval']:
        waterflows[key] = load_water_flows(area, key, sec_data_dir)
    max_date = waterflows['eval'].ObsDate.max() + pd.Timedelta(7, unit = 'days')

    # load and aggregate meteo data
    meteo_data = {}
    print("Loading and aggregating meteo data for {area}")
    for key in ['train', 'eval']:
        stations_ = stations[~stations.eval_only] if key == 'train' else stations.copy()
        meteo_data[key] = load_and_aggregate_meteo_data(
            area = area, 
            data_dir = sec_data_dir, 
            key = key,
            stations_gdf = stations_, 
            max_date = max_date, 
            hydro_data = hydro_data,
            buffer_scales = meteo_buffer, 
            crs = crs
        )

    # merge all data
    merged = {}
    print(f"Merging all data for {area}")
    for key in ['train', 'eval']:
        merged[key] = merge_all_data(stations, waterflows[key], meteo_data[key])

    return merged
    
def main(is_mini = False):    
    # load config
    config = config = load_config()
    
    # get constants
    RAW_DATA = config['raw_data']
    PREPROCESSED_DATA = config['preprocessed_data']
    AREAS = config['areas']
    TYPES = config['types']
    SOIL_BUFFER = config['soil_buffer']
    METEO_BUFFER = config['meteo_buffer']
    CRS = config['crs']
    SEC_DATA = config['mini_data'] if is_mini else RAW_DATA

    # make preprocessed folder
    os.makedirs(PREPROCESSED_DATA, exist_ok = True)

    # process
    if is_mini:
        merged = process_mini(
            data_dir = RAW_DATA, 
            sec_data_dir = SEC_DATA, 
            soil_buffer = SOIL_BUFFER, 
            meteo_buffer = METEO_BUFFER, 
            crs = CRS
        )
        name = os.path.join(PREPROCESSED_DATA, f'preprocessed_mini_brazil_eval.csv')
        print('Saving data outputs for mini')
        merged.to_csv(name, index = False)
    else:
        for area in AREAS:
            merged = process_main(
                area = area,
                data_dir = RAW_DATA, 
                sec_data_dir = SEC_DATA, 
                soil_buffer = SOIL_BUFFER, 
                meteo_buffer = METEO_BUFFER, 
                crs = CRS
            )
            print(f"Saving data outputs for {area}")
            for key in ['train', 'eval']:
                data = merged[key]
                name = os.path.join(PREPROCESSED_DATA, f'preprocessed_{area}_{key}.csv')
                data.to_csv(name, index = False)
            
    print(f'All outputs saved to {PREPROCESSED_DATA}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-mini', action='store_true', help = 'Preprocess mini data')
    args = parser.parse_args()
    main(is_mini = args.is_mini)