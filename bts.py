"""Core backend logic for lightning data processing in the HLMA application.

This module provides functions for reading, processing, and analyzing
lightning datasets from LYLOUT and ENTLN sources. It includes algorithms
for detecting and grouping lightning flashes, as well as utilities for
preprocessing and structuring the data for visualization and analysis
in the HLMA GUI.

Functions
---------
zipped_lylout_reader(file, skiprows=55)
    Reads a compressed LYLOUT `.dat.gz` file. Used by `open_lylout`.

lylout_reader(file, skiprows=55)
    Reads a plain-text LYLOUT `.dat` file. Used by `open_lylout`.

open_lylout(files)
    Reads multiple LYLOUT files and combines them into a single DataFrame.

entln_reader(file, min_date)
    Reads a single ENTLN CSV file. Used by `open_entln`.

open_entln(files, min_date)
    Reads multiple ENTLN CSV files and combines them into a single DataFrame.

dot_to_dot(env)
    Applies the dot-to-dot flash detection algorithm on lightning data.

mc_caul(env)
    Applies the McCaul flash detection algorithm on lightning data.

Notes
-----
- Reader functions are used by `open_lylout` and `open_entln` to process multiple files.
- Other functions are mostly internal and intended for use by the HLMA application.
- The flash detection algorithms are experimental and should be interpreted with caution.

"""


import gzip
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyproj import Transformer

from setup import State

warnings.filterwarnings("ignore")

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("bts.py")
logger.setLevel(logging.DEBUG)

def zipped_lylout_reader(file: str, skiprows: int = 55) -> pd.DataFrame | None:
    """Read a compressed LYLOUT .dat.gz file into a DataFrame. Used by `open_lylout`.

    Parses the file, calculates the number of stations contributing from the mask,
    extracts the date from the filename, computes absolute datetimes, and initializes
    a `flash_id` column.

    Parameters
    ----------
    file : str
        Path to the `.dat.gz` LYLOUT file.
    skiprows : int, optional
        Number of initial rows to skip (default is 55).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns:
        ["datetime", "lat", "lon", "alt", "chi", "pdb",
         "number_stations", "utc_sec", "mask", "flash_id"].
        Returns `None` if the file could not be read.

    """
    try:
        with gzip.open(file, "rt") as f:
            tmp = pd.read_csv(f, skiprows = skiprows, header=None, names=["utc_sec", "lat", "lon", "alt", "chi", "pdb", "mask"], sep=r"\s+")
            tmp["number_stations"] = tmp["mask"].apply(lambda x: int(x, 16).bit_count())
            tmp_date = re.match(r".*\w+_(\d+)_\d+_\d+\.dat\.gz", file).group(1)
            tmp["datetime"] = pd.to_datetime(tmp_date, format="%y%m%d") + pd.to_timedelta(tmp.utc_sec, unit="s")
            tmp["flash_id"] = -1
            return tmp[["datetime", "lat", "lon", "alt", "chi", "pdb", "number_stations", "utc_sec", "mask", "flash_id"]].reset_index(drop=True)
    except Exception as e:
        logger.warning("Could not open %s due to %s", file, e)
        return None

def lylout_reader(file: str, skiprows: int = 55) -> pd.DataFrame | None:
    """Read a plain-text LYLOUT .dat file into a DataFrame. Used by `open_lylout`.

    Parses the file, calculates the number of stations contributing from the mask,
    extracts the date from the filename, computes absolute datetimes, and initializes
    a `flash_id` column.

    Parameters
    ----------
    file : str
        Path to the LYLOUT `.dat` file.
    skiprows : int, optional
        Number of initial rows to skip (default is 55).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns:
        ["datetime", "lat", "lon", "alt", "chi", "pdb",
         "number_stations", "utc_sec", "mask", "flash_id"].
        Returns `None` if the file could not be read.

    """
    try:
        tmp = pd.read_csv(file, skiprows = skiprows, header=None, names=["utc_sec", "lat", "lon", "alt", "chi", "pdb", "mask"], sep=r"\s+")
        tmp["number_stations"] = tmp["mask"].apply(lambda x: int(x, 16).bit_count())
        tmp_date = re.match(r".*\w+_(\d+)_\d+_\d+\.dat", file).group(1)
        tmp["datetime"] = pd.to_datetime(tmp_date, format="%y%m%d") + pd.to_timedelta(tmp.utc_sec, unit="s")
        tmp["flash_id"] = -1
        tmp = tmp[["datetime", "lat", "lon", "alt", "chi", "pdb", "number_stations", "utc_sec", "mask", "flash_id"]]
        tmp = tmp.reset_index(drop=True)
    except Exception as e:
        logger.warning("Could not open %s due to %s", file, e)
        return None
    else:
        return tmp

def open_lylout(files: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Read multiple LYLOUT files (compressed or plain) and combine them into a single DataFrame.

    Determines the number of header rows by inspecting the first file, extracts
    LMA station coordinates, reads all files in parallel, concatenates the results,
    and computes a 'seconds' column relative to the first midnight.

    Parameters
    ----------
    files : list of str
        List of paths to LYLOUT files (.dat or .dat.gz).

    Returns
    -------
    tuple
        Tuple containing:
        - `pd.DataFrame`: All LYLOUT data concatenated.
        - `np.ndarray`: LMA station coordinates as float32, shape (n_stations, 2).

    """
    # manually read first file to eshtablish skiprows and lma info
    lma_stations = []
    skiprows = None
    logger.info("Starting to read LYLOUT files.")
    if files[0].endswith(".dat.gz"):
        with gzip.open(files[0], "rt") as f:
            for i, line in enumerate(f):
                if line.strip().startswith("Sta_info:"):
                    parts = line.strip().split()
                    lon = float(parts[-5])
                    lat = float(parts[-6])
                    lma_stations.append((lon, lat))
                if line.startswith("*** data ***"):
                    skiprows = i + 1
                    break

        lylout_read = Parallel(n_jobs=-5)(delayed(zipped_lylout_reader)(f, skiprows=skiprows) for f in files)
    else:
        with Path.open(files[0]) as f:
            for i, line in enumerate(f):
                if line.strip().startswith("Sta_info:"):
                    parts = line.strip().split()
                    lon = float(parts[-5])
                    lat = float(parts[-6])
                    lma_stations.append((lon, lat))
                if line.startswith("*** data ***"):
                    skiprows = i + 1
                    break


        lylout_read = Parallel(n_jobs=-5)(delayed(lylout_reader)(f, skiprows=skiprows) for f in files)

    all_data = pd.concat(lylout_read, ignore_index=True)
    all_data["seconds"] = (all_data["datetime"] - all_data["datetime"].min().normalize()).dt.total_seconds()

    return all_data, np.array(lma_stations, dtype=np.float32)

def open_entln(files: list[str], min_date: pd.Timestamp) -> pd.DataFrame:
    """Read multiple ENTLN CSV files and combine them into a single DataFrame.

    Uses `entln_reader` to process each file, then concatenates the results
    and computes a 'seconds' column relative to the first midnight of the dataset.

    Parameters
    ----------
    files : list of str
        List of paths to ENTLN CSV files.
    min_date : pd.Timestamp
        Minimum datetime reference to filter the data.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing all ENTLN data with a computed 'seconds' column.

    """
    logger.info("Received min time of %s", min_date)
    entln_read = Parallel(n_jobs=-5)(delayed(entln_reader)(f, min_date) for f in files)

    all_data = pd.concat(entln_read, ignore_index=True)
    all_data["seconds"] = (all_data["datetime"] - all_data["datetime"].min().normalize()).dt.total_seconds()

    return all_data

def entln_reader(file: str, min_date: pd.Timestamp) -> pd.DataFrame | None:
    """Read a single ENTLN CSV file and format it to match LYLOUT data. Used by `open_entln`.

    Converts timestamps, renames columns to standard names, computes UTC seconds,
    and returns a DataFrame with essential columns.

    Parameters
    ----------
    file : str
        Path to the ENTLN CSV file.
    min_date : pd.Timestamp
        Reference datetime for computing UTC seconds.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns:
        ["datetime", "lat", "lon", "alt", "peakcurrent", "numbersensors", "utc_sec", "type"].
        Returns `None` if the file cannot be read or processed.

    """
    try:
        tmp = pd.read_csv(file)
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
        tmp["type"] = pd.to_numeric(tmp["type"])

        # re-name to match LYLOUT file
        tmp = tmp.rename(columns={
            "timestamp": "datetime",
            "latitude": "lat",
            "longitude": "lon",
            "icheight": "alt",
        })

        tmp["utc_sec"] = (tmp["datetime"] - min_date).dt.total_seconds()
    except Exception as e:
        logger.warning("Failed on %s due to %s", file, e)
        return None
    else:
        return tmp[["datetime", "lat", "lon", "alt", "peakcurrent", "numbersensors", "utc_sec", "type"]]

def dot_to_dot(env: State) -> None:
    """Apply the dot-to-dot flash detection algorithm on lightning data.

    Groups lightning events in space and time to identify flashes, updates
    the `flash_id` column in the dataset, and projects data to ECEF coordinates.
    Computation is parallelized for speed.

    Parameters
    ----------
    env : State
        State object containing lightning data (`env.all`) and station coordinates (`env.stations`).

    Returns
    -------
    None

    """
    logger.info("Starting dot to dot flashing.")
    # unpacking
    lyl = env.all[env.plot]
    env.all["flash_id"] = -1 # resetting global flash data to avoid incosistencies
    lma_stations = env.stations
    lon_0, lat_0 = tuple(sum(coords) / len(lma_stations) for coords in zip(*lma_stations, strict=True))
    distance_threshold = 3000 # in meters
    time_threshold = 0.15

    # projecting
    to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    x_0, y_0, z_0 = to_ecef.transform(lon_0, lat_0, 0)
    xs, ys, zs = to_ecef.transform(lyl.lon, lyl.lat, lyl.alt)
    lyl["x"], lyl["y"], lyl["z"] = xs - x_0, ys - y_0, zs - z_0

    timethreshold_ns = int(pd.Timedelta(seconds=time_threshold).value)
    distancethreshold_2 = distance_threshold**2

    def dtd_flasher(df: pd.DataFrame) -> np.ndarray:
        fid = 0
        remaining = np.ones(len(df), dtype=bool)
        datetimes = df["datetime"].to_numpy(dtype=np.int64)
        xys = df[["x", "y"]].to_numpy()
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

    gap = lyl["datetime"].astype("int64").diff() > timethreshold_ns
    group_ids = gap.cumsum()
    dfs = [group.reset_index(drop=True).copy() for _, group in lyl.groupby(group_ids) if len(group) > 0]
    results = Parallel(n_jobs=-10, backend="loky")(delayed(dtd_flasher)(df) for df in dfs)

    offset = 0
    for i, res in enumerate(results):
        results[i] = res + offset
        offset = results[i].max() + 1
    env.all.loc[lyl.index, "flash_id"] = np.concatenate(results)

    logger.info("Finished dot to dot flashing.")

def mc_caul(env: State) -> None:
    """Apply the McCaul flash detection algorithm on lightning data.

    Groups lightning events using distance, time, and azimuth thresholds,
    updates the `flash_id` column in the dataset, and projects data to
    ECEF coordinates. Computation is parallelized for speed.

    Parameters
    ----------
    env : State
        State object containing lightning data (`env.all`) and station coordinates (`env.stations`).

    Returns
    -------
    None

    """
    logger.info("Starting McCaul flashing.")
    logger.info("Starting McCaul flashing.")
    # unpacking
    lyl = env.all[env.plot]
    env.all["flash_id"] = -1 # resetting global flash data to avoid incosistencies
    lma_stations = env.stations
    lon_0, lat_0 = tuple(sum(coords) / len(lma_stations) for coords in zip(*lma_stations, strict=True))
    time_threshold = 0.15 # in seconds
    azimuth_threshold = 0.05 # in radians

    # projecting
    to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    x_0, y_0, z_0 = to_ecef.transform(lon_0, lat_0, 0)
    xs, ys, zs = to_ecef.transform(lyl.lon, lyl.lat, lyl.alt)
    lyl["x"], lyl["y"], lyl["z"] = xs - x_0, ys - y_0, zs - z_0

    timethreshold_ns = int(pd.Timedelta(seconds=time_threshold).value)

    def mcc_flasher(df: pd.DataFrame) -> np.ndarray:
        fid = 0
        remaining = np.ones(len(df), dtype=bool)

        datetimes = df["datetime"].to_numpy(dtype=np.int64)
        xys = df[["x", "y"]].to_numpy()
        azimuths = np.arctan2(df["y"].to_numpy(), df["x"].to_numpy())
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

    gap = lyl["datetime"].astype("int64").diff() > timethreshold_ns
    group_ids = gap.cumsum()
    dfs = [group.reset_index(drop=True).copy() for _, group in lyl.groupby(group_ids) if len(group) > 0]
    results = Parallel(n_jobs=-10, backend="loky")(delayed(mcc_flasher)(df) for df in dfs)

    offset = 0
    for i, res in enumerate(results):
        results[i] = res + offset
        offset = results[i].max() + 1
    env.all.loc[lyl.index, "flash_id"] = np.concatenate(results)

    logger.info("Finished McCaul flashing.")
