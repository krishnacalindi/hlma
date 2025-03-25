import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import geopandas as gpd
import datashader as ds
import datashader.transfer_functions as tf

def OpenLylout(file):
    date = re.search(r"LYLOUT_(\d{6})_\d{6}_\d{4}\.dat", os.path.basename(file)).group(1)
    data_line = 0
    with open(file, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            if '*** data ***' in line:
                data_line = line_num
                break
    lyl = pd.read_csv(file, skiprows=data_line, delimiter=r'\s+', engine='python', names=['seconds', 'lat', 'lon', 'alt', 'chi', 'pdb', 'mask'], header=None)
    lyl['datetime'] = pd.to_datetime(date, format='%y%m%d') + pd.to_timedelta(lyl['seconds'], unit='s')
    return lyl[(lyl["alt"] < 20000) & (lyl["chi"] < 5)]

def Plot(imgs):
    fig = plt.figure(figsize=(10, 12))

    gs = GridSpec(3, 2, height_ratios=[1, 1, 8], width_ratios=[8, 1])
    axs = []
    axs.append(fig.add_subplot(gs[0, :]))
    axs.append(fig.add_subplot(gs[1, 0]))
    axs.append(fig.add_subplot(gs[1, 1]))
    axs.append(fig.add_subplot(gs[2, 0]))
    axs.append(fig.add_subplot(gs[2, 1]))
    
    for i in range(5):
        im, xmin, xmax, ymin, ymax = imgs[i]
        axs[i].imshow(im.to_pil(), aspect='auto', extent=[xmin, xmax, ymin, ymax])
        if i == 0:
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif i == 2:
            axs[i].ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

    fig.tight_layout()

    return FigureCanvasQTAgg(fig)

def QuickImage(lyl, cvar, cmap, map, features, subplot):
    cmap = plt.get_cmap(f"cet_{cmap}")
    df = pd.concat(lyl)
    lonmin, lonmax = -98.5, -91.5
    latmin, latmax = 26, 33
    altmin, altmax = 0, 20000
    if subplot == 0:
        cvs = ds.Canvas(plot_width=1500, plot_height=150, y_range=(altmin, altmax))
        agg = cvs.points(df, 'seconds', 'alt', ds.mean(cvar))
        img = tf.set_background(tf.shade(agg, cmap=cmap), "white")
        return (img, df['datetime'].min().floor('h'), df['datetime'].max().floor('h'), int(altmin/1000), int(altmax/1000))
    elif subplot == 1:
        cvs = ds.Canvas(plot_width=1200, plot_height=150, x_range=(lonmin, lonmax), y_range=(altmin, altmax))
        agg = cvs.points(df, 'lon', 'alt', ds.mean(cvar))
        img = tf.set_background(tf.shade(agg, cmap=cmap), "white")
        return (img, lonmin, lonmax, int(altmin/1000), int(altmax/1000))
    elif subplot == 2:
        cvs = ds.Canvas(plot_width=150, plot_height=150, y_range=(altmin, 20000))
        counts, bin_edges = np.histogram(df["alt"], bins=10)
        hist = pd.DataFrame({'count': counts, 'edges': bin_edges[:-1]})
        agg = cvs.line(hist, 'count', 'edges')
        img = tf.set_background(tf.shade(agg, cmap="black"), "white")
        return (img, 0, hist['count'].max(), int(altmin/1000), int(altmax/1000))
    elif subplot == 3:
        f_colors = {"roads": "brown",
                    "rivers": "blue",
                    "rails": "red",
                    "urban": "sienna"}
        gdf = gpd.read_file(f"assets/maps/{map}/{map}.shp")
        cvs = ds.Canvas(plot_width=1200, plot_height=1200, x_range=(lonmin, lonmax), y_range=(latmin, latmax))
        agg = cvs.line(gdf, geometry="geometry")
        img = tf.shade(agg, cmap=["black"])
        for feature, fcolor in f_colors.items():
            f_index = list(f_colors.keys()).index(feature)
            if features[f_index] != 0:
                gdf_feat = gpd.read_file(f"assets/features/{feature}/{feature}.shp")
                agg_feat = cvs.line(gdf_feat, geometry="geometry")
                img_feat = tf.shade(agg_feat, cmap=[fcolor])
                img = tf.set_background(tf.stack(img, img_feat))
        cvs_dat = ds.Canvas(plot_width=1200, plot_height=1200, x_range=(lonmin, lonmax), y_range=(latmin, latmax))
        agg_dat = cvs_dat.points(df, 'lon', 'lat', ds.mean(cvar))
        img_dat = tf.shade(agg_dat, cmap=cmap)
        img = tf.set_background(tf.stack(img, img_dat), "white")
        return (img, lonmin, lonmax, latmin, latmax)
    else:
        cvs = ds.Canvas(plot_width=150, plot_height=1200, x_range=(altmin, altmax), y_range=(latmin, latmax))
        agg = cvs.points(df, 'alt', 'lat', ds.mean(cvar))
        img = tf.set_background(tf.shade(agg, cmap=cmap), "white")
        return (img, int(altmin/1000), int(altmax/1000), latmin, latmax)

def BlankPlot():
    fig = plt.figure(figsize=(10, 12))
    return FigureCanvasQTAgg(fig)

class Nav(NavigationToolbar2QT):
    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ('Save', )]
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)

