# global imports
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
import re
from deprecated import deprecated

# data imports
import pandas as pd
import numpy as np
from pyproj import Transformer
import gzip

# logging
import logging

#setup
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("bts.py")
logger.setLevel(logging.DEBUG)

def ZippedLyloutReader(file, skiprows = 55):
    try:
        with gzip.open(file, 'rt') as f:
            tmp = pd.read_csv(f, skiprows = skiprows, header=None, names=['utc_sec', 'lat', 'lon', 'alt', 'chi', 'pdb', 'mask'], sep=r'\s+')
            tmp['number_stations'] = tmp['mask'].apply(lambda x: bin(int(x, 16)).count('1'))
            tmp_date = re.match(r'.*\w+_(\d+)_\d+_\d+\.dat\.gz', file).group(1)
            tmp['datetime'] = pd.to_datetime(tmp_date, format='%y%m%d') + pd.to_timedelta(tmp.utc_sec, unit='s')
            if not tmp[tmp['datetime'] == pd.Timestamp("2022-09-01T23:59:59.033390989")].empty:
                logger.info(f"{file}")
            tmp['flash_id'] = -1
            tmp = tmp[['datetime', 'lat', 'lon', 'alt', 'chi', 'pdb', 'number_stations', 'utc_sec', 'mask', 'flash_id']]
            tmp.reset_index(inplace=True, drop=True)
            return tmp
    except Exception as e:
        logger.warning(f"Could not open {file} due to {e}.")
        return

def OpenLylout(files):
    # manually read first file to eshtablish skiprows and lma info
    lma_stations = []
    skiprows = None
    logger.info("Starting to read LYLOUT files.")
    if files[0].endswith(".dat.gz"):
        with gzip.open(files[0], 'rt') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line.startswith("Sta_info:"):
                    parts = line.split()
                    lon = float(parts[-5])
                    lat = float(parts[-6])
                    lma_stations.append((lon, lat))
                if line.startswith("*** data ***"):
                    skiprows = i + 1
                    break
                
        lylout_read = Parallel(n_jobs=-5)(delayed(ZippedLyloutReader)(f, skiprows=skiprows) for f in files)
    else:
        with open(files[0], 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line.startswith("Sta_info:"):
                    parts = line.split()
                    lon = float(parts[-5])
                    lat = float(parts[-6])
                    lma_stations.append((lon, lat))
                if line.startswith("*** data ***"):
                    skiprows = i + 1
                    break
    
        
        lylout_read = Parallel(n_jobs=-5)(delayed(LyloutReader)(f, skiprows=skiprows) for f in files)

    all = pd.concat(lylout_read, ignore_index=True) 
    all["seconds"] = (all['datetime'] - all['datetime'].min().normalize()).dt.total_seconds()

    return all, lma_stations

def LyloutReader(file, skiprows = 55):
    try:
        tmp = pd.read_csv(file, skiprows = skiprows, header=None, names=['utc_sec', 'lat', 'lon', 'alt', 'chi', 'pdb', 'mask'], sep=r'\s+')
        tmp['number_stations'] = tmp['mask'].apply(lambda x: bin(int(x, 16)).count('1'))
        tmp_date = re.match(r'.*\w+_(\d+)_\d+_\d+\.dat', file).group(1)
        tmp['datetime'] = pd.to_datetime(tmp_date, format='%y%m%d') + pd.to_timedelta(tmp.utc_sec, unit='s')
        if not tmp[tmp['datetime'] == pd.Timestamp("2022-09-01T23:59:59.033390989")].empty:
            logger.info(f"{file}")
        tmp['flash_id'] = -1
        tmp = tmp[['datetime', 'lat', 'lon', 'alt', 'chi', 'pdb', 'number_stations', 'utc_sec', 'mask', 'flash_id']]
        tmp.reset_index(inplace=True, drop=True)
        return tmp
    except Exception as e:
        logger.warning(f"Could not open {file} due to {e}.")
        return

def OpenEntln(files, min_date):
    logger.info(f"Received min time of {min_date}")
    entln_read = Parallel(n_jobs=-5)(delayed(ENTLNReader)(f, min_date) for f in files)

    all = pd.concat(entln_read, ignore_index=True)
    all["seconds"] = (all['datetime'] - all['datetime'].min().normalize()).dt.total_seconds()

    return all

def ENTLNReader(file, min_date):
    try:
        tmp = pd.read_csv(file)
        tmp['timestamp'] = pd.to_datetime(tmp['timestamp'])
        tmp['type'] = pd.to_numeric(tmp['type'])

        # re-name to match LYLOUT file
        tmp.rename(columns={
            'timestamp': 'datetime',
            'latitude': 'lat',
            'longitude': 'lon',
            'icheight': 'alt',
        }, inplace=True)
        
        tmp['utc_sec'] = (tmp['datetime'] - min_date).dt.total_seconds()

        tmp = tmp[['datetime', 'lat', 'lon', 'alt', 'peakcurrent', 'numbersensors', 'utc_sec', 'type']]

        return tmp
    except Exception as e:
        logger.warning(f"Failed on {file} due to {e}")
        return

def DotToDot(env):
    logger.info("Starting dot to dot flashing.")
    # unpacking
    lyl = env.all[env.plot]
    env.all['flash_id'] = -1 # resetting global flash data to avoid incosistencies
    lma_stations = env.stations
    lon_0, lat_0 = tuple(sum(coords) / len(lma_stations) for coords in zip(*lma_stations))
    distance_threshold = 3000 # in meters
    time_threshold = 0.15
    
    # projecting
    to_ecef = Transformer.from_crs('EPSG:4326', 'EPSG:4978', always_xy=True)
    x_0, y_0, z_0 = to_ecef.transform(lon_0, lat_0, 0)
    xs, ys, zs = to_ecef.transform(lyl.lon, lyl.lat, lyl.alt)
    lyl['x'], lyl['y'], lyl['z'] = xs - x_0, ys - y_0, zs - z_0
    
    timethreshold_ns = int(pd.Timedelta(seconds=time_threshold).value)
    distancethreshold_2 = distance_threshold**2
    
    def dtd_flasher(df):
        fid = 0
        remaining = np.ones(len(df), dtype=bool)        
        datetimes = df['datetime'].values.astype(np.int64)
        xys = df[['x', 'y']].to_numpy()
        indices = df.index.to_numpy()
        flash_id = np.full(len(df), -1) 
        while remaining.any():
            candidates = np.flatnonzero(remaining)
            candidates_dts = datetimes[candidates]
            candidates_xys = xys[candidates]
            candidates_ids = indices[candidates]
            flash_mask = np.zeros(len(candidates), dtype=bool)
            flash_mask[0] = True
            consideration = (candidates_dts - candidates_dts[0]) <= timethreshold_ns
            consideration[0] = False
            concan = np.flatnonzero(consideration)
            lyst = list(concan)
            syt = set(concan)
            for i in lyst:
                if not flash_mask[i]:
                    flash_indices = np.where(flash_mask)[0]
                    if np.any((np.sum((candidates_xys[flash_indices] - candidates_xys[i])**2, axis=1) <= distancethreshold_2) & ((candidates_dts[i] - candidates_dts[flash_indices]) > 0) & ((candidates_dts[i] - candidates_dts[flash_indices]) <= timethreshold_ns)):
                        flash_mask[i] = True
                        consideration = ((candidates_dts - candidates_dts[flash_mask].max()) > 0) & ((candidates_dts - candidates_dts[flash_mask].max()) <= timethreshold_ns) & (~flash_mask)
                        newconcan = set(np.flatnonzero(consideration)) - syt
                        syt.update(newconcan)
                        lyst.extend(newconcan)
            update = candidates_ids[flash_mask]
            remaining[update] = False
            flash_id[update] = fid
            fid += 1
        return flash_id
    
    gap = lyl['datetime'].astype('int64').diff() > timethreshold_ns
    group_ids = gap.cumsum()
    dfs = [group.reset_index(drop=True).copy() for _, group in lyl.groupby(group_ids) if len(group) > 0]
    results = Parallel(n_jobs=-10, backend='loky')(delayed(dtd_flasher)(df) for df in dfs)
    
    offset = 0
    for _, res in enumerate(results):
        res += offset
        offset = res.max() + 1
    env.all.loc[lyl.index, 'flash_id'] = np.concatenate(results)
    logger.info("Finished dot to dot flashing.")

def McCaul(env):
    logger.info("Starting McCaul flashing.")
    # unpacking
    lyl = env.all[env.plot]
    env.all['flash_id'] = -1 # resetting global flash data to avoid incosistencies
    lma_stations = env.stations
    lon_0, lat_0 = tuple(sum(coords) / len(lma_stations) for coords in zip(*lma_stations))
    time_threshold = 0.15 # in seconds
    azimuth_threshold = 0.05 # in radians
    
    # projecting
    to_ecef = Transformer.from_crs('EPSG:4326', 'EPSG:4978', always_xy=True)
    x_0, y_0, z_0 = to_ecef.transform(lon_0, lat_0, 0)
    xs, ys, zs = to_ecef.transform(lyl.lon, lyl.lat, lyl.alt)
    lyl['x'], lyl['y'], lyl['z'] = xs - x_0, ys - y_0, zs - z_0
    
    timethreshold_ns = int(pd.Timedelta(seconds=time_threshold).value)
    
    def mcc_flasher(df):
        fid = 0
        remaining = np.ones(len(df), dtype=bool)
        
        datetimes = df['datetime'].values.astype(np.int64)
        xys = df[['x', 'y']].to_numpy()
        azimuths = np.arctan2(df['y'].to_numpy(), df['x'].to_numpy())
        indices = df.index.to_numpy()
        
        flash_id = np.full(len(df), -1) 
        
        while remaining.any():
            candidates = np.flatnonzero(remaining)
            candidates_dts = datetimes[candidates]
            candidates_xys = xys[candidates]
            candidates_azs = azimuths[candidates]
            candidates_ids = indices[candidates]
            flash_mask = np.zeros(len(candidates), dtype=bool)
            flash_mask[0] = True
            consideration = (candidates_dts - candidates_dts[0]) <= timethreshold_ns
            consideration[0] = False
            concan = np.flatnonzero(consideration)
            lyst = list(concan)
            syt = set(concan)
            for i in lyst:
                if not flash_mask[i]:
                    distancethreshold_2 = 9000000 * ((candidates_xys[i][0]**2 + candidates_xys[i][1]**2) / 100000**2)**2 # FIXME: worses r2 from center of lma 
                    flash_indices = np.where(flash_mask)[0]
                    if np.any((np.sum((candidates_xys[flash_indices] - candidates_xys[i])**2, axis=1) <= distancethreshold_2) & 
                            ((candidates_dts[i] - candidates_dts[flash_indices]) > 0) & 
                            ((candidates_dts[i] - candidates_dts[flash_indices]) <= timethreshold_ns) & 
                            (np.minimum(np.abs(candidates_azs[flash_indices] - candidates_azs[i]), 2*np.pi - np.abs(candidates_azs[flash_indices] - candidates_azs[i])) <= 0.05)):
                        flash_mask[i] = True
                        consideration = ((candidates_dts - candidates_dts[flash_mask].max()) > 0) & ((candidates_dts - candidates_dts[flash_mask].max()) <= timethreshold_ns) & (~flash_mask)
                        newconcan = set(np.flatnonzero(consideration)) - syt
                        syt.update(newconcan)
                        lyst.extend(newconcan)
            update = candidates_ids[flash_mask]
            remaining[update] = False
            flash_id[update] = fid
            fid += 1
        return flash_id

    gap = lyl['datetime'].astype('int64').diff() > timethreshold_ns
    group_ids = gap.cumsum()
    dfs = [group.reset_index(drop=True).copy() for _, group in lyl.groupby(group_ids) if len(group) > 0]
    results = Parallel(n_jobs=-10, backend='loky')(delayed(mcc_flasher)(df) for df in dfs)
    
    offset = 0
    for _, res in enumerate(results):
        res += offset
        offset = res.max() + 1
    env.all.loc[lyl.index, 'flash_id'] = np.concatenate(results)
    
    logger.info("Finished McCaul flashing.")
    