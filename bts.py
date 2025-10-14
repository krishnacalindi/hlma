# global imports
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
import re
from deprecated import deprecated

# data imports
import pandas as pd
import numpy as np
np.seterr(invalid='ignore')
from pyproj import Transformer

# plot imports
import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import datashader as ds
import datashader.transfer_functions as tf

# logging
import logging

# setup
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "legend.facecolor": "black",
    "legend.edgecolor": "white",
})
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("bts.py")
logger.setLevel(logging.DEBUG)
def OpenLylout(files):
    # manually read first file to eshtablish skiprows and lma info
    lma_stations = []
    skiprows = None

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
    
    logger.info("Starting to read LYLOUT files.")
    lylout_read = Parallel(n_jobs=-5)(delayed(LyloutReader)(f, skiprows=skiprows) for f in files)
    return pd.concat(lylout_read, ignore_index=True), lma_stations
    
def LyloutReader(file, skiprows = 55):
    try:
        tmp = pd.read_csv(file, skiprows = skiprows, header=None, names=['utc_sec', 'lat', 'lon', 'alt', 'chi', 'pdb', 'mask'], sep=r'\s+')
        tmp['number_stations'] = tmp['mask'].apply(lambda x: bin(int(x, 16)).count('1'))
        tmp_date = re.match(r'.*LYLOUT_(\d+)_\d+_0600\.dat', file).group(1)
        tmp['datetime'] = pd.to_datetime(tmp_date, format='%y%m%d') + pd.to_timedelta(tmp.utc_sec, unit='s')
        tmp['flash_id'] = -1
        tmp = tmp[['datetime', 'lat', 'lon', 'alt', 'chi', 'pdb', 'number_stations', 'utc_sec', 'mask', 'flash_id']]
        tmp.reset_index(inplace=True, drop=True)
        return tmp
    except Exception as e:
        logger.warning(f"Could not open {file} due to {e}.")
        return

@deprecated("Datashader plot generator, using vispy + PyQT6.")
def QuickImage(env):
    # unpacking
    lyl = env.all[env.plot]
    lma_stations = env.stations
    map = env.plot_options.map
    features = env.plot_options.features
    cmap = env.plot_options.cmap
    cvar = env.plot_options.cvar
    lonmin = env.plot_options.lon_min
    lonmax = env.plot_options.lon_max
    latmin = env.plot_options.lat_min
    latmax = env.plot_options.lat_max
    
    imgs = []
    cvs = ds.Canvas(plot_width=1500, plot_height=150, y_range=(0 , 20000))
    agg = cvs.points(lyl, 'utc_sec', 'alt', ds.mean(cvar))
    img = tf.set_background(tf.shade(agg, cmap=cmap), "black")
    imgs.append((img, lyl['datetime'].min().floor('N'), lyl['datetime'].max().floor('n'), 0, 20))

    cvs = ds.Canvas(plot_width=1200, plot_height=150, x_range=(lonmin, lonmax), y_range=(0, 20000))
    agg = cvs.points(lyl, 'lon', 'alt', ds.mean(cvar))
    img = tf.set_background(tf.shade(agg, cmap=cmap), "black")
    imgs.append((img, lonmin, lonmax, 0, 20))

    cvs = ds.Canvas(plot_width=150, plot_height=150, y_range=(0 , 20000))
    counts, bin_edges = np.histogram(lyl["alt"], bins=10)
    hist = pd.DataFrame({'count': counts, 'edges': bin_edges[:-1]})
    agg = cvs.line(hist, 'count', 'edges')
    img = tf.set_background(tf.shade(agg, cmap="white"), "black")
    imgs.append((img, 0, hist['count'].max(), 0, 20))
    
    cvs = ds.Canvas(plot_width=1200, plot_height=1200, x_range=(lonmin, lonmax), y_range=(latmin, latmax))
    agg_map = cvs.line(map, geometry="geometry")
    agg_features = [cvs.line(fdict['gdf'], geometry="geometry") for _, fdict in features.items()]
    agg_points = cvs.points(lyl, 'lon', 'lat', ds.mean(cvar))
    agg_stat = cvs.points(pd.DataFrame(lma_stations, columns=["lon","lat"]), 'lon', 'lat', ds.count())
    img_map = tf.shade(agg_map, cmap=["white"])
    img_features = [tf.shade(agg_feat, cmap=[fdict['color']]) for agg_feat, fdict in zip(agg_features, features.values())]
    img_points = tf.shade(agg_points, cmap=cmap)
    img_stat = tf.spread(tf.shade(agg_stat, cmap=["red"]), px=3, shape='square')
    img = tf.set_background(tf.stack(img_map, *img_features, img_points, img_stat), "black")
    imgs.append((img, lonmin, lonmax, latmin, latmax))
    
    cvs = ds.Canvas(plot_width=150, plot_height=1200, x_range=(0 * 1000, 20 * 1000), y_range=(latmin, latmax))
    agg = cvs.points(lyl, 'alt', 'lat', ds.mean(cvar))
    img = tf.set_background(tf.shade(agg, cmap=cmap), "black")
    imgs.append((img, 0, 20, latmin, latmax))

    fig = plt.figure(figsize=(10, 12))

    gs = GridSpec(3, 2, height_ratios=[1, 1, 8], width_ratios=[8, 1])
    axs = []
    axs.append(fig.add_subplot(gs[0, :]))
    axs[0].name = 0
    axs.append(fig.add_subplot(gs[1, 0]))
    axs[1].name = 1
    axs.append(fig.add_subplot(gs[1, 1]))
    axs[2].name = 2
    axs.append(fig.add_subplot(gs[2, 0]))
    axs[3].name = 3
    axs.append(fig.add_subplot(gs[2, 1]))
    axs[4].name = 4
    
    for i in range(5):
        im, xmin, xmax, ymin, ymax = imgs[i]
        axs[i].imshow(im.to_pil(), aspect='auto', extent=[xmin, xmax, ymin, ymax])
        if i == 0:
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif i == 2:
            axs[i].ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

    fig.tight_layout()
    logger.info("Finished plotting.")
    return fig

@deprecated("Blank datashadr plot generator, using vispy + PyQT6.")
def BlankPlot(env):
    # unpacking
    lyl = pd.DataFrame({'x': [], 'y': []})
    map = env.plot_options.map
    lonmin = env.plot_options.lon_min
    lonmax = env.plot_options.lon_max
    latmin = env.plot_options.lat_min
    latmax = env.plot_options.lat_max
    
    imgs = []
    cvs = ds.Canvas(plot_width=1500, plot_height=150, y_range=(0 , 20000))
    agg = cvs.points(lyl, 'x', 'y')
    img = tf.set_background(tf.shade(agg, cmap=["white"]), "black")
    imgs.append((img, pd.Timestamp('2000-01-01 01:00:00'), pd.Timestamp('2000-01-01 02:00:00'), 0, 20))

    cvs = ds.Canvas(plot_width=1200, plot_height=150, x_range=(lonmin, lonmax), y_range=(0, 20000))
    agg = cvs.points(lyl, 'x', 'y')
    img = tf.set_background(tf.shade(agg, cmap=["white"]), "black")
    imgs.append((img, lonmin, lonmax, 0, 20))

    cvs = ds.Canvas(plot_width=150, plot_height=150, y_range=(0 , 20000))
    agg = cvs.points(lyl, 'x', 'y')
    img = tf.set_background(tf.shade(agg, cmap=["white"]), "black")
    imgs.append((img, 0, 1, 0, 20))
    
    cvs = ds.Canvas(plot_width=1200, plot_height=1200, x_range=(lonmin, lonmax), y_range=(latmin, latmax))
    agg = cvs.line(map, geometry="geometry")
    img = tf.set_background(tf.shade(agg, cmap=["white"]), "black")
    imgs.append((img, lonmin, lonmax, latmin, latmax))
    
    cvs = ds.Canvas(plot_width=150, plot_height=1200, x_range=(0, 20000), y_range=(latmin, latmax))
    agg = cvs.points(lyl, 'x', 'y')
    img = tf.set_background(tf.shade(agg, cmap=["white"]), "black")
    imgs.append((img, 0, 20, latmin, latmax))

    fig = plt.figure(figsize=(10, 12))

    gs = GridSpec(3, 2, height_ratios=[1, 1, 8], width_ratios=[8, 1])
    axs = []
    axs.append(fig.add_subplot(gs[0, :]))
    axs[0].name = 0
    axs.append(fig.add_subplot(gs[1, 0]))
    axs[1].name = 1
    axs.append(fig.add_subplot(gs[1, 1]))
    axs[2].name = 2
    axs.append(fig.add_subplot(gs[2, 0]))
    axs[3].name = 3
    axs.append(fig.add_subplot(gs[2, 1]))
    axs[4].name = 4
    
    for i in range(5):
        im, xmin, xmax, ymin, ymax = imgs[i]
        axs[i].imshow(im.to_pil(), aspect='auto', extent=[xmin, xmax, ymin, ymax])
        if i == 0:
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif i == 2:
            axs[i].ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

    fig.tight_layout()
    
    return fig

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
    