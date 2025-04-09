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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import geopandas as gpd
import datashader as ds
import datashader.transfer_functions as tf

clicks = []
dots = []
lines = []

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
    return pl.from_pandas(lyl)

def Plot(imgs):
    fig = plt.figure(figsize=(10, 12))

    if imgs:
        print("Rendering images")
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

    def on_click(event):
        global clicks, lines, dots # There may be a better way, this was my first idea

        if event.inaxes: # Checks if inside a graph
            ax = event.inaxes
            x, y = event.xdata, event.ydata
            if event.button == 1: # Left click
                # print(f"Clicked on x={x}, y={y}") # Debugging statement

                dot, *_ = ax.plot(x, y, 'ro')
                dots.append(dot) # Grab the dot object
                clicks.append((x, y))

                if len(clicks) >= 2:
                    prev_x, prev_y = clicks[-2]
                    line, *_ = ax.plot([prev_x, x], [prev_y, y], 'r-') # Grab the Line2D object
                    lines.append(line)

                fig.canvas.draw()
            elif event.button == 3: # Right click
                if len(clicks) > 1:
                    # Handle shape stuff here, not really sure how to filter from here
                    # Probably shapely? then check if its inside the polygon?
                    first_x, first_y = clicks[0]
                    line, *_ = ax.plot([clicks[-1][0], first_x], [clicks[-1][1], first_y], 'r-') # This should close the figure
                    lines.append(line) 
                    
                    # Build polygon with lines
                    # For Shapely we can use polygon = Polygon(clicks)

                    # Clearing drawn points here
                    clicks.clear()
                    for line in lines:
                        line.remove()
                    for dot in dots:
                        dot.remove()
                    
                    lines.clear()
                    dots.clear()

                    fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    return FigureCanvasQTAgg(fig)

def QuickImage(lyl, cvar, cmap, map, features, extents):
    cmap = plt.get_cmap(f"cet_{cmap}")
    ogdf = pl.concat(lyl)
    
    timemin, timemax, lonmin, lonmax, latmin, latmax, altmin, altmax, chimin, chimax, pdbmin, pdbmax = extents
    
    if timemin != 'yyyy-mm-dd hh:mm:ss':
        timemin = datetime.strptime(timemin, "%Y-%m-%d %H:%M:%S")
    else:
        timemin = ogdf['datetime'].min()
    
    if timemax != 'yyyy-mm-dd hh:mm:ss':
        timemax = datetime.strptime(timemax, "%Y-%m-%d %H:%M:%S")
    else:
        timemax = ogdf['datetime'].max()
    
    df = ogdf.filter(
    (pl.col("datetime") >= timemin) & (pl.col("datetime") <= timemax) &
    (pl.col("lon") >= lonmin) & (pl.col("lon") <= lonmax) &
    (pl.col("lat") >= latmin) & (pl.col("lat") <= latmax) &
    (pl.col("alt") >= altmin * 1000) & (pl.col("alt") <= altmax * 1000) &
    (pl.col("chi") >= chimin) & (pl.col("chi") <= chimax) &
    (pl.col("pdb") >= pdbmin) & (pl.col("pdb") <= pdbmax)).to_pandas() # FIXME: is it a good idea to filter every plot?

    imgs = []
    if len(df) != 0:  
        cvs = ds.Canvas(plot_width=1500, plot_height=150, y_range=(altmin * 1000, altmax * 1000))
        agg = cvs.points(df, 'seconds', 'alt', ds.mean(cvar))
        img = tf.set_background(tf.shade(agg, cmap=cmap), "white")
        imgs.append((img, df['datetime'].min().floor('h'), df['datetime'].max().floor('h'), altmin, altmax))

        cvs = ds.Canvas(plot_width=1200, plot_height=150, x_range=(lonmin, lonmax), y_range=(altmin * 1000, altmax * 1000))
        agg = cvs.points(df, 'lon', 'alt', ds.mean(cvar))
        img = tf.set_background(tf.shade(agg, cmap=cmap), "white")
        imgs.append((img, lonmin, lonmax, altmin, altmax))

        cvs = ds.Canvas(plot_width=150, plot_height=150, y_range=(altmin * 1000, altmax * 1000))
        counts, bin_edges = np.histogram(df["alt"], bins=10)
        hist = pd.DataFrame({'count': counts, 'edges': bin_edges[:-1]})
        agg = cvs.line(hist, 'count', 'edges')
        img = tf.set_background(tf.shade(agg, cmap="black"), "white")
        imgs.append((img, 0, hist['count'].max(), altmin, altmax))

        f_colors = {"roads": "brown", "rivers": "blue", "rails": "red", "urban": "sienna"}
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
        imgs.append((img, lonmin, lonmax, latmin, latmax))

        cvs = ds.Canvas(plot_width=150, plot_height=1200, x_range=(altmin * 1000, altmax * 1000), y_range=(latmin, latmax))
        agg = cvs.points(df, 'alt', 'lat', ds.mean(cvar))
        img = tf.set_background(tf.shade(agg, cmap=cmap), "white")
        imgs.append((img, altmin, altmax, latmin, latmax))

    return imgs

def BlankPlot():
    fig = plt.figure(figsize=(10, 12))
    return FigureCanvasQTAgg(fig)

class Nav(NavigationToolbar2QT):
    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ('Save', )]
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)

