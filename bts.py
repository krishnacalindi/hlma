"""Contains the core backend logic for lightning data processing in the HLMA application.

Key functionalities include:
- Reading and parsing LYLOUT and ENTLN lightning datasets.
- Implementing flash detection algorithms such as Dot-to-Dot and McCaul.
- Preprocessing, filtering, and transforming lightning event data.
- Providing data structures and utilities used by the GUI for visualization and analysis.

This module serves as the main computational engine for HLMA, enabling efficient
handling and analysis of large-scale lightning datasets.
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

warnings.filterwarnings("ignore")

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("bts.py")
logger.setLevel(logging.DEBUG)

def zipped_lylout_reader(file: str, skiprows: int = 55) -> pd.DataFrame | None:
    """Read a compressed LYLOUT `.dat.gz` file and parse it into a structured DataFrame.

    Uses:
        - Opens a gzip-compressed LYLOUT file in text mode
        - Reads data with `pandas.read_csv`, skipping header rows
        - Computes the number of stations contributing from the mask
        - Extracts the date from the filename and computes absolute datetimes
        - Initializes a `flash_id` column
        - Logs if the file contains a specific timestamp

    Args:
        file: Path to the `.dat.gz` LYLOUT file.
        skiprows: Number of initial rows to skip (default is 55).

    Returns:
        pd.DataFrame with columns:
            ["datetime", "lat", "lon", "alt", "chi", "pdb",
             "number_stations", "utc_sec", "mask", "flash_id"]
        Returns `None` if the file could not be read.

    """
    try:
        with gzip.open(file, "rt") as f:
            tmp = pd.read_csv(f, skiprows = skiprows, header=None, names=["utc_sec", "lat", "lon", "alt", "chi", "pdb", "mask"], sep=r"\s+")
            tmp["number_stations"] = tmp["mask"].apply(lambda x: int(x, 16).bit_count())
            tmp_date = re.match(r".*\w+_(\d+)_\d+_\d+\.dat\.gz", file).group(1)
            tmp["datetime"] = pd.to_datetime(tmp_date, format="%y%m%d") + pd.to_timedelta(tmp.utc_sec, unit="s")
            tmp["flash_id"] = -1
            tmp = tmp[["datetime", "lat", "lon", "alt", "chi", "pdb", "number_stations", "utc_sec", "mask", "flash_id"]]
            tmp = tmp.reset_index(drop=True)
            return tmp
    except Exception as e:
        logger.warning("Could not open %s due to %s", file, e)
        return None

def lylout_reader(file: str, skiprows: int = 55) -> pd.DataFrame | None:
    """Read a plain-text LYLOUT `.dat` file into a structured DataFrame.

    Uses:
        - Reads the file with pandas, skipping header rows
        - Computes number of stations contributing from the mask
        - Extracts date from filename and calculates absolute datetimes
        - Initializes a `flash_id` column
        - Logs if a specific timestamp is present in the data

    Args:
        file: Path to the LYLOUT `.dat` file.
        skiprows: Number of initial rows to skip (default 55).

    Returns:
        pd.DataFrame with columns:
            ["datetime", "lat", "lon", "alt", "chi", "pdb",
             "number_stations", "utc_sec", "mask", "flash_id"]
        Returns None if the file cannot be read.

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
    """Read multiple LYLOUT files (compressed or plain) and combine into a single DataFrame.

    Uses:
        - Determines the number of header rows to skip by inspecting the first file
        - Extracts LMA station coordinates from lines starting with "Sta_info:"
        - Reads all files in parallel using `ZippedLyloutReader` (for .dat.gz) or `LyloutReader`
        - Concatenates all individual DataFrames into one
        - Computes 'seconds' column relative to the first midnight

    Args:
        files: List of paths to LYLOUT files to read (.dat or .dat.gz).

    Returns:
        Tuple containing:
            - `pd.DataFrame` with all LYLOUT data concatenated
            - `np.ndarray` of LMA station coordinates as float32, shape (n_stations, 2)

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
    """Read multiple ENTLN CSV files and combine into a single DataFrame.

    Uses:
        - Logs the minimum datetime received for reference
        - Reads all files in parallel using `ENTLNReader` with the given `min_date`
        - Concatenates all individual DataFrames into one
        - Computes a 'seconds' column relative to the first midnight of the dataset

    Args:
        files: List of paths to ENTLN CSV files.
        min_date: Minimum datetime reference to filter the data.

    Returns:
        pd.DataFrame containing all ENTLN data with a computed 'seconds' column.

    """
    logger.info("Received min time of %s", min_date)
    entln_read = Parallel(n_jobs=-5)(delayed(entln_reader)(f, min_date) for f in files)

    all_data = pd.concat(entln_read, ignore_index=True)
    all_data["seconds"] = (all_data["datetime"] - all_data["datetime"].min().normalize()).dt.total_seconds()

    return all_data

def entln_reader(file: str, min_date: pd.Timestamp) -> pd.DataFrame | None:
    """Read a single ENTLN CSV file and format it to match LYLOUT structure.

    Uses:
        - Parses CSV and converts 'timestamp' to datetime
        - Converts 'type' to numeric
        - Renames columns to standard names for downstream processing
        - Computes 'utc_sec' relative to provided min_date
        - Returns a filtered DataFrame with essential columns

    Args:
        file: Path to the ENTLN CSV file.
        min_date: Reference datetime for computing UTC seconds.

    Returns:
        pd.DataFrame with columns:
        ["datetime", "lat", "lon", "alt", "peakcurrent", "numbersensors", "utc_sec", "type"]
        Returns None if file cannot be read or processed.

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

def dot_to_dot(env: object) -> None:
    """Run the dot-to-dot flash algorithm on the provided lightning data.

    Uses:
        - Groups lightning events in space and time to identify flashes
        - Updates the 'flash_id' column in env.all
        - Projects data to ECEF coordinates
        - Parallelized computation for speed

    Note:
        - This is a beta algorithm; results should be double-checked by the user.

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

def mc_caul(env: object) -> None:
    """Run the McCaul flash algorithm on the provided lightning data.

    Uses:
        - Groups lightning events using distance, time, and azimuth thresholds
        - Updates the 'flash_id' column in env.all
        - Projects data to ECEF coordinates
        - Parallelized computation for speed

    Note:
        - This is a beta algorithm; results should be double-checked by the user.

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
