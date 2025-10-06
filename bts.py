import pandas as pd
import polars as pl
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from datetime import datetime
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import geopandas as gpd
import datashader as ds
import datashader.transfer_functions as tf
from joblib import Parallel, delayed
from tqdm import tqdm

# FIXME: why Plot? and QuickImage?


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
    lylout_read = Parallel(n_jobs=-5)(delayed(LyloutReader)(f, skiprows=skiprows) for f in tqdm(files, desc=f'{datetime.now().strftime("%b %d %H:%M:%S")} ⏳ Processing LYLOUT files', bar_format='{desc}: {n_fmt}/{total_fmt}.'))
    failed_files = []
    for i in range(len(lylout_read) - 1, -1, -1):
        if lylout_read[i] is None:
            failed_files.append(files[i])
            lylout_read.pop(i)
    if lylout_read:
        return pd.concat(lylout_read, ignore_index=True), failed_files, lma_stations
    else:
        return None, failed_files, lma_stations
    
def LyloutReader(file, skiprows = 55):
    try:
        tmp = pd.read_csv(file, skiprows = skiprows, header=None, names=['utc_sec', 'lat', 'lon', 'alt', 'chi', 'pdb', 'mask'], sep='\s+')
        tmp['number_stations'] = tmp['mask'].apply(lambda x: bin(int(x, 16)).count('1'))
        tmp_date = re.match(r'.*LYLOUT_(\d+)_\d+_0600\.dat', file).group(1)
        tmp['datetime'] = pd.to_datetime(tmp_date, format='%y%m%d') + pd.to_timedelta(tmp.utc_sec, unit='s')
        tmp = tmp[['datetime', 'lat', 'lon', 'alt', 'chi', 'pdb', 'number_stations', 'utc_sec', 'mask']]
        tmp.reset_index(inplace=True, drop=True)
        return tmp
    except:
        return None

def QuickImage(lyl, cvar, cmap, map, features, lma_stations, limits = (-98, -92, 27, 33, 0, 20)):
    cmap = plt.get_cmap(f"cet_{cmap}")
    
    lonmin, lonmax, latmin, latmax, altmin, altmax = limits

    imgs = []
    cvs = ds.Canvas(plot_width=1500, plot_height=150, y_range=(altmin * 1000, altmax * 1000))
    agg = cvs.points(lyl, 'utc_sec', 'alt', ds.mean(cvar))
    img = tf.set_background(tf.shade(agg, cmap=cmap), "white")
    imgs.append((img, lyl['datetime'].min().floor('N'), lyl['datetime'].max().floor('n'), altmin, altmax))

    cvs = ds.Canvas(plot_width=1200, plot_height=150, x_range=(lonmin, lonmax), y_range=(altmin * 1000, altmax * 1000))
    agg = cvs.points(lyl, 'lon', 'alt', ds.mean(cvar))
    img = tf.set_background(tf.shade(agg, cmap=cmap), "white")
    imgs.append((img, lonmin, lonmax, altmin, altmax))

    cvs = ds.Canvas(plot_width=150, plot_height=150, y_range=(altmin * 1000, altmax * 1000))
    counts, bin_edges = np.histogram(lyl["alt"], bins=10)
    hist = pd.DataFrame({'count': counts, 'edges': bin_edges[:-1]})
    agg = cvs.line(hist, 'count', 'edges')
    img = tf.set_background(tf.shade(agg, cmap="black"), "white")
    imgs.append((img, 0, hist['count'].max(), altmin, altmax))

    f_colors = {"roads": "brown", "rivers": "blue", "rails": "red", "urban": "sienna"}
    glyl = gpd.read_file(f"assets/maps/{map}/{map}.shp")
    cvs = ds.Canvas(plot_width=1200, plot_height=1200, x_range=(lonmin, lonmax), y_range=(latmin, latmax))
    agg = cvs.line(glyl, geometry="geometry")
    img = tf.shade(agg, cmap=["black"])
    for feature, fcolor in f_colors.items():
        f_index = list(f_colors.keys()).index(feature)
        if features[f_index] != 0:
            glyl_feat = gpd.read_file(f"assets/features/{feature}/{feature}.shp")
            agg_feat = cvs.line(glyl_feat, geometry="geometry")
            img_feat = tf.shade(agg_feat, cmap=[fcolor])
            img = tf.set_background(tf.stack(img, img_feat))
    cvs_dat = ds.Canvas(plot_width=1200, plot_height=1200, x_range=(lonmin, lonmax), y_range=(latmin, latmax))
    agg_dat = cvs_dat.points(lyl, 'lon', 'lat', ds.mean(cvar))
    img_dat = tf.shade(agg_dat, cmap=cmap)
    cvs_stat = ds.Canvas(plot_width=1200, plot_height=1200, x_range=(lonmin, lonmax), y_range=(latmin, latmax))
    agg_stat = cvs_stat.points(pd.DataFrame(lma_stations, columns=["lon","lat"]), 'lon', 'lat', ds.count())
    img_stat = tf.shade(agg_stat, cmap=["red"])
    img_stat = tf.spread(img_stat, px=3, shape='square')
    img = tf.set_background(tf.stack(img, img_dat, img_stat), "white")
    imgs.append((img, lonmin, lonmax, latmin, latmax))
    cvs = ds.Canvas(plot_width=150, plot_height=1200, x_range=(altmin * 1000, altmax * 1000), y_range=(latmin, latmax))
    agg = cvs.points(lyl, 'alt', 'lat', ds.mean(cvar))
    img = tf.set_background(tf.shade(agg, cmap=cmap), "white")
    imgs.append((img, altmin, altmax, latmin, latmax))

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

def BlankPlot():
    fig = plt.figure(figsize=(10, 12))
    return fig