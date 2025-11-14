"""Setup module for UI and dataclass management.

This module handles UI setup, connects UI elements to back-end functions,
and defines essential dataclasses used throughout the application.

It provides the following functionality:
- Initialize and configure the main application window.
- Connect UI widgets to corresponding back-end logic.
- Define dataclasses for storing application state, plot options,
  animation options, and other utility.

Usage:
    Import this module at the start of your application to ensure
    that the UI is properly initialized and all event handlers
    are connected.
"""

import copy
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import colorcet  # noqa: F401
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from PyQt6.QtCore import QRegularExpression, Qt
from PyQt6.QtGui import (
    QAction,
    QDoubleValidator,
    QIcon,
    QIntValidator,
    QKeySequence,
    QRegularExpressionValidator,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from vispy import app, scene
from vispy.scene import AxisWidget, visuals

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("setup.py")
logger.setLevel(logging.DEBUG)

class LoadingDialog(QDialog):
    """UI blocking dialog for various functions.

    A simple modal loading dialog that displays a message while
    the application is performing a background task.

    This dialog is frameless, stays on top of other windows, and
    prevents user interaction with other windows until closed.

    Attributes:
        message (str): The message to display in the dialog.

    """

    def __init__(self, message: str) -> None:
        """Initialize the loading dialog with a given message.

        Args:
            message (str): The message to display in the dialog.

        """
        super().__init__()
        self.setWindowTitle("Please wait...")
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setFixedSize(300, 100)
        layout = QVBoxLayout()
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)

def setup_ui(obj: QMainWindow) -> SimpleNamespace:
    """Set up the main user interface for a QMainWindow instance.

    This function organizes the main application window into a left
    panel for filter, map, color, and animation options, and a right
    panel with multiple visualization canvases arranged in a grid.

    It initializes scene canvases, markers, line plots, menus, and
    associated actions. Filter, map, color, and animation widgets
    are all added to the left panel.

    Args:
        obj (QMainWindow): The main window of the application to which
            the UI elements will be attached.

    Returns:
        SimpleNamespace: A container object holding references to all
            created widgets, layouts, and visualization canvases for
            easy access and manipulation.

    """
    ui = SimpleNamespace()

    # Main spliter that splits UI into plots on the right and options on the left
    main_splitter = QSplitter()

    # Define the left panel containing option widgets
    option_widget = QWidget()
    option_layout = QVBoxLayout()
    option_widget.setLayout(option_layout)

    #  Define the right panel containing plots
    ui.view_widget = QWidget()
    grid_plot = QGridLayout()
    ui.view_widget.setLayout(grid_plot)

    def grid_view_axes(grid: scene.Grid) -> scene.ViewBox:
        """Configure axes and camera for a given grid and return the view.

        This function adds bottom and left axes to the grid, links them
        to the main view, sets rotation and size constraints, and
        configures the camera stretch.

        Args:
            grid (scene.Grid): The VisPy grid to which axes and view will be added.

        Returns:
            scene.ViewBox: The configured view object with linked axes and camera.

        """
        view = grid.add_view(0, 1)
        view.camera = scene.PanZoomCamera(aspect=None)
        x = AxisWidget(orientation="bottom", minor_tick_length=1, major_tick_length=3,
                       tick_font_size=5, tick_label_margin=10, axis_width=1)
        y = AxisWidget(orientation="left", minor_tick_length=1, major_tick_length=3,
                       tick_font_size=5, tick_label_margin=10, axis_width=1)
        grid.add_widget(x, 1, 1)
        grid.add_widget(y, 0, 0)
        x.link_view(view)
        y.link_view(view)
        y.axis._text.rotation = -90 # NOTE: makes y axis labels turn sideways
        y.width_max = 20
        x.height_max = 20
        view.stretch = (1, 1)
        return view

    ui.c0 = scene.SceneCanvas(keys=None, show=False, bgcolor="black")
    grid_plot.addWidget(ui.c0.native, 0, 0, 1, 2)
    grid = ui.c0.central_widget.add_grid()
    ui.v0 = grid_view_axes(grid)
    ui.s0 = visuals.Markers(spherical=True, edge_width=0,
                            light_position=(0, 0, 1), light_ambient=0.9)
    ui.v0.add(ui.s0)
    ui.pd0 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl0 = visuals.Line(color="red", width=1)
    ui.v0.add(ui.pl0)
    ui.gs0 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)
    ui.cc0 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)

    ui.c1 = scene.SceneCanvas(keys=None, show=False, bgcolor="black")
    grid_plot.addWidget(ui.c1.native, 1, 0)
    grid = ui.c1.central_widget.add_grid()
    ui.v1 = grid_view_axes(grid)
    ui.s1 = visuals.Markers(spherical=True, edge_width=0,
                            light_position=(0, 0, 1), light_ambient=0.9)
    ui.v1.add(ui.s1)
    ui.pd1 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl1 = visuals.Line(color="red", width=1)
    ui.v1.add(ui.pl1)
    ui.gs1 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)
    ui.cc1 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)


    ui.c2 = scene.SceneCanvas(keys=None, show=False, bgcolor="black")
    grid_plot.addWidget(ui.c2.native, 1, 1)
    grid = ui.c2.central_widget.add_grid()
    ui.v2 = grid_view_axes(grid)
    ui.v2.stretch = (1, 1)
    ui.hist = visuals.Line(color="white", width=1)
    ui.v2.add(ui.hist)
    ui.pd2 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl2 = visuals.Line(color="red", width=1)
    ui.v2.add(ui.pl2)

    ui.c3 = scene.SceneCanvas(keys=None, show=False, bgcolor="black")
    grid_plot.addWidget(ui.c3.native, 2, 0)
    grid = ui.c3.central_widget.add_grid()
    ui.v3 = grid_view_axes(grid)
    ui.map = visuals.Line(color="white", width=1)
    ui.v3.add(ui.map)
    ui.s3 = visuals.Markers(spherical=True, edge_width=0,
                            light_position=(0, 0, 1), light_ambient=0.9)
    ui.v3.add(ui.s3)
    ui.pd3 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl3 = visuals.Line(color="red", width=1)
    ui.v3.add(ui.pl3)
    ui.gs3 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)
    ui.cc3 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)

    ui.stats = visuals.Markers(spherical=True,
                               light_position=(0, 0, 1), light_ambient=0.9)
    ui.v3.add(ui.stats)

    ui.c4 = scene.SceneCanvas(keys=None, show=False, bgcolor="black")
    grid_plot.addWidget(ui.c4.native, 2, 1)
    grid = ui.c4.central_widget.add_grid()
    ui.v4 = grid_view_axes(grid)
    ui.s4 = visuals.Markers(spherical=True, edge_width=0,
                            light_position=(0, 0, 1), light_ambient=0.9)
    ui.v4.add(ui.s4)
    ui.pd4 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl4 = visuals.Line(color="red", width=1)
    ui.v4.add(ui.pl4)
    ui.gs4 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)
    ui.cc4 = visuals.Markers(spherical=True, edge_width=0,
                             light_position=(0, 0, 1), light_ambient=0.9)

    grid_plot.setRowStretch(0, 1)
    grid_plot.setRowStretch(1, 1)
    grid_plot.setRowStretch(2, 8)

    grid_plot.setColumnStretch(0, 8)
    grid_plot.setColumnStretch(1, 1)

    ui.v3.camera.link(ui.v1.camera, axis="x")
    ui.v3.camera.link(ui.v4.camera, axis="y")

    # Main UI layout setup
    center = QWidget()
    main = QHBoxLayout()
    main_splitter.addWidget(option_widget)
    main_splitter.addWidget(ui.view_widget)
    main_splitter.setSizes([1, 1])
    main.addWidget(main_splitter)
    center.setLayout(main)
    obj.setCentralWidget(center)

    # Defining menu bar and its options
    menubar = obj.menuBar()
    import_menu = menubar.addMenu("Import")
    export_menu = menubar.addMenu("Export")
    options_menu = menubar.addMenu("Options")
    flash_menu = menubar.addMenu("Flash")
    help_menu = menubar.addMenu("Help")
    filter_menu = menubar.addMenu("Filter")
    # Import menu containing various options for opening files
    ui.import_menu_lylout = QAction("LYLOUT", obj)
    ui.import_menu_lylout.setIcon(QIcon("assets/icons/lyl.svg"))
    import_menu.addAction(ui.import_menu_lylout)
    ui.import_menu_entln = QAction("ENTLN", obj)
    ui.import_menu_entln.setIcon(QIcon("assets/icons/entln.svg"))
    import_menu.addAction(ui.import_menu_entln)
    ui.import_menu_state = QAction("State", obj)
    ui.import_menu_state.setIcon(QIcon("assets/icons/state.svg"))
    import_menu.addAction(ui.import_menu_state)
    # Export menu containing various options for exporting files
    ui.export_menu_dat = QAction("DAT", obj)
    ui.export_menu_dat.setIcon(QIcon("assets/icons/dat.svg"))
    export_menu.addAction(ui.export_menu_dat)
    ui.export_menu_parquet = QAction("Parquet", obj)
    ui.export_menu_parquet.setIcon(QIcon("assets/icons/parquet.svg"))
    export_menu.addAction(ui.export_menu_parquet)
    ui.export_menu_state = QAction("State", obj)
    ui.export_menu_state.setIcon(QIcon("assets/icons/state.svg"))
    export_menu.addAction(ui.export_menu_state)
    ui.export_menu_image = QAction("Image", obj)
    ui.export_menu_image.setIcon(QIcon("assets/icons/image.svg"))
    export_menu.addAction(ui.export_menu_image)
    # Options menu containing various plot options
    ui.options_menu_draw = QAction("Animate", obj)
    ui.options_menu_draw.setIcon(QIcon("assets/icons/draw.svg"))
    ui.options_menu_draw.setShortcut(QKeySequence("Ctrl+D"))
    options_menu.addAction(ui.options_menu_draw)
    ui.options_menu_reset = QAction("Reset", obj)
    ui.options_menu_reset.setIcon(QIcon("assets/icons/reset.svg"))
    ui.options_menu_reset.setShortcut(QKeySequence("F5"))
    options_menu.addAction(ui.options_menu_reset)
    ui.options_menu_clear = QAction("Clear", obj)
    ui.options_menu_clear.setIcon(QIcon("assets/icons/clear.svg"))
    ui.options_menu_clear.setShortcut(QKeySequence("Delete"))
    options_menu.addAction(ui.options_menu_clear)
    # Flash menu containig two flash algorithms
    ui.flash_menu_dtd = QAction("Dot to Dot", obj)
    ui.flash_menu_dtd.setIcon(QIcon("assets/icons/dtd.svg"))
    flash_menu.addAction(ui.flash_menu_dtd)
    ui.flash_menu_mccaul = QAction("McCaul", obj)
    ui.flash_menu_mccaul.setIcon(QIcon("assets/icons/mcc.svg"))
    flash_menu.addAction(ui.flash_menu_mccaul)
    # Help menu allowing some websites with important info
    ui.help_menu_colors = QAction("Colors", obj)
    ui.help_menu_colors.setIcon(QIcon("assets/icons/color.svg"))
    help_menu.addAction(ui.help_menu_colors)
    ui.help_menu_about = QAction("About", obj)
    ui.help_menu_about.setIcon(QIcon("assets/icons/about.svg"))
    help_menu.addAction(ui.help_menu_about)
    ui.help_menu_contact = QAction("Contact", obj)
    ui.help_menu_contact.setIcon(QIcon("assets/icons/contact.svg"))
    help_menu.addAction(ui.help_menu_contact)
    # Filter menu to toggle polygon selections
    ui.filter_menu_keep = QAction("Keep", obj)
    ui.filter_menu_keep.setIcon(QIcon("assets/icons/keep.svg"))
    filter_menu.addAction(ui.filter_menu_keep)
    ui.filter_menu_remove = QAction("Remove", obj)
    ui.filter_menu_remove.setIcon(QIcon("assets/icons/remove.svg"))
    filter_menu.addAction(ui.filter_menu_remove)

    # Options menu on the left
    ui.cvar_dropdown = QComboBox()
    ui.cvar_dropdown.addItems(["Time", "Longitude", "Latitude", "Altitude", "Chi", "Receiving power", "Flash"])
    ui.map_dropdown = QComboBox()
    ui.map_dropdown.addItems(["State", "County", "NOAA CWAs", "Congressional Districts"])
    ui.cmap_dropdown = QComboBox()
    for cmap_name in [
        "bgy", "CET_D8", "bjy", "CET_CBD2", "blues", "bmw", "bmy",
        "CET_L10", "gray", "dimgray", "kbc", "gouldian", "kgy", "fire",
        "CET_CBL1", "CET_CBL3", "CET_CBL4", "kb", "kg", "kr",
        "CET_CBTL3", "CET_CBTL1", "CET_L19", "CET_L17", "CET_L18",
    ]:
        icon = QIcon(f"assets/colors/{cmap_name}.svg")
        ui.cmap_dropdown.addItem(icon, cmap_name)
    map_features = QHBoxLayout()
    ui.features = {}
    ui.features["roads"] = QCheckBox("Roads")
    ui.features["rivers"] = QCheckBox("Rivers")
    ui.features["rails"] = QCheckBox("Rails")
    ui.features["urban"] = QCheckBox("Urban area")
    map_features.addWidget(ui.features["roads"])
    map_features.addWidget(ui.features["rivers"])
    map_features.addWidget(ui.features["rails"])
    map_features.addWidget( ui.features["urban"])
    ui.avar_dropdown = QComboBox()
    ui.avar_dropdown.addItems(["Time", "Longitude", "Latitude",
                               "Altitude", "Chi", "Receiving power", "Flash"])

    # filters
    time_filter = QHBoxLayout()
    time_validator = QRegularExpressionValidator(QRegularExpression(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"))
    ui.timemin = QLineEdit()
    ui.timemin.setText("yyyy-mm-dd hh:mm:ss")
    ui.timemin.setValidator(time_validator)
    ui.timemax = QLineEdit()
    ui.timemax.setText("yyyy-mm-dd hh:mm:ss")
    ui.timemax.setValidator(time_validator)
    time_filter.addWidget(QLabel("Minimum time:"), 2)
    time_filter.addWidget(ui.timemin, 1)
    time_filter.addWidget(QLabel("Maximum time:"), 2)
    time_filter.addWidget(ui.timemax, 1)
    alt_filter = QHBoxLayout()
    ui.altmin = QLineEdit()
    ui.altmin.setText("0.0")
    ui.altmin.setValidator(QDoubleValidator())
    ui.altmax = QLineEdit()
    ui.altmax.setText("20.0")
    ui.altmax.setValidator(QDoubleValidator())
    alt_filter.addWidget(QLabel("Minimum altitude:"), 2)
    alt_filter.addWidget(ui.altmin, 1)
    alt_filter.addWidget(QLabel("Maximum altitude:"), 2)
    alt_filter.addWidget(ui.altmax, 1)
    chi_filter = QHBoxLayout()
    ui.chimin = QLineEdit()
    ui.chimin.setText("0.0")
    ui.chimin.setValidator(QDoubleValidator())
    ui.chimax = QLineEdit()
    ui.chimax.setText("2.0")
    ui.chimax.setValidator(QDoubleValidator())
    chi_filter.addWidget(QLabel("Minimum chi:"), 2)
    chi_filter.addWidget(ui.chimin, 1)
    chi_filter.addWidget(QLabel("Maximum chi:"), 2)
    chi_filter.addWidget(ui.chimax, 1)
    power_filter = QHBoxLayout()
    ui.powermin = QLineEdit()
    ui.powermin.setText("-60.0")
    ui.powermin.setValidator(QDoubleValidator())
    ui.powermax = QLineEdit()
    ui.powermax.setText("60.0")
    ui.powermax.setValidator(QDoubleValidator())
    power_filter.addWidget(QLabel("Minimum receiving power:"), 2)
    power_filter.addWidget(ui.powermin, 1)
    power_filter.addWidget(QLabel("Maximum receiving power:"), 2)
    power_filter.addWidget(ui.powermax, 1)
    stations_filter = QHBoxLayout()
    ui.stationsmin = QLineEdit()
    ui.stationsmin.setText("6")
    ui.stationsmin.setValidator(QIntValidator())
    stations_filter.addWidget(QLabel("Minimum number of stations:"), 2)
    stations_filter.addWidget(ui.stationsmin, 1)
    stations_filter.addStretch(3)

    # map options
    option_layout.addWidget(QLabel("<h1>Filter options</h1>"))
    option_layout.addLayout(time_filter)
    option_layout.addStretch(1)
    option_layout.addLayout(alt_filter)
    option_layout.addStretch(1)
    option_layout.addLayout(chi_filter)
    option_layout.addStretch(1)
    option_layout.addLayout(power_filter)
    option_layout.addStretch(1)
    option_layout.addLayout(stations_filter)
    option_layout.addStretch(2)
    option_layout.addWidget(QLabel("<h1>Map options</h1>"))
    option_layout.addWidget(QLabel("Map:"))
    option_layout.addWidget(ui.map_dropdown)
    option_layout.addStretch(1)
    option_layout.addWidget(QLabel("Features:"))
    option_layout.addLayout(map_features)
    option_layout.addStretch(3)

    # color options
    option_layout.addWidget(QLabel("<h1>Color options</h1>"))
    option_layout.addWidget(QLabel("Color by:"))
    option_layout.addWidget(ui.cvar_dropdown)
    option_layout.addStretch(1)
    option_layout.addWidget(QLabel("Color map:"))
    option_layout.addWidget(ui.cmap_dropdown)
    option_layout.addStretch(3)

    # animation options
    option_layout.addWidget(QLabel("<h1>Animation options</h1>"))
    option_layout.addWidget(QLabel("Animate by:"))
    option_layout.addWidget(ui.avar_dropdown)
    option_layout.addStretch(1)
    anim_duration = QHBoxLayout()
    ui.aduration = QLineEdit()
    ui.aduration.setText("5")
    ui.aduration.setValidator(QDoubleValidator())
    anim_duration.addWidget(QLabel("Animation duration: (seconds)"), 3)
    anim_duration.addWidget(ui.aduration, 1)
    option_layout.addLayout(anim_duration)
    option_layout.addStretch(3)
    option_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    # returning ui
    return ui

def connect_ui(obj: QMainWindow, ui: SimpleNamespace) -> None:
    """Connect UI widgets and menus to their corresponding back-end functions.

    This function wires signals and slots for the main application,
    including menu actions, filter inputs, plot options, animation
    controls, and feature checkboxes. It ensures that user interactions
    trigger the appropriate updates in the application state and
    visualization.

    Args:
        obj (QMainWindow): The main application window instance containing
            state, animation, and utility objects to be updated by UI actions.
        ui (SimpleNamespace): Namespace containing all UI widgets, layouts,
            menus, and visualization canvases created in the setup_ui function.

    Returns:
        None

    """
    # connections
    # menubar
    ui.import_menu_lylout.triggered.connect(obj.import_lylout)
    ui.import_menu_entln.triggered.connect(obj.import_entln)
    ui.import_menu_state.triggered.connect(obj.import_state)
    ui.export_menu_dat.triggered.connect(obj.export_dat)
    ui.export_menu_parquet.triggered.connect(obj.export_parquet)
    ui.export_menu_state.triggered.connect(obj.export_state)
    ui.export_menu_image.triggered.connect(obj.export_image)
    ui.options_menu_draw.triggered.connect(obj.animate)
    ui.options_menu_clear.triggered.connect(obj.options_clear)
    ui.options_menu_reset.triggered.connect(obj.options_reset)
    ui.flash_menu_dtd.triggered.connect(obj.flash_dtd)
    ui.flash_menu_mccaul.triggered.connect(obj.flash_mccaul)
    ui.help_menu_colors.triggered.connect(obj.help_color)
    ui.help_menu_about.triggered.connect(obj.help_about)
    ui.help_menu_contact.triggered.connect(obj.help_contact)
    ui.filter_menu_keep.triggered.connect(lambda _: obj.polyfilter.update_filter(new=False))
    ui.filter_menu_remove.triggered.connect(lambda _: obj.polyfilter.update_filter(new=True))

    # filters
    ui.timemin.editingFinished.connect(obj.filter)
    ui.timemax.editingFinished.connect(obj.filter)
    ui.altmin.editingFinished.connect(obj.filter)
    ui.altmax.editingFinished.connect(obj.filter)
    ui.chimin.editingFinished.connect(obj.filter)
    ui.chimax.editingFinished.connect(obj.filter)
    ui.powermin.editingFinished.connect(obj.filter)
    ui.powermax.editingFinished.connect(obj.filter)
    ui.stationsmin.editingFinished.connect(obj.filter)

    # plot options
    ui.cvar_dropdown.currentIndexChanged.connect(lambda index: obj.state.update(cvar=obj.util.cvars[index]))
    ui.cmap_dropdown.currentIndexChanged.connect(lambda index: obj.state.update(cmap=obj.util.cmaps[index]))
    ui.map_dropdown.currentIndexChanged.connect(lambda index: obj.state.update(map=obj.util.maps[index]))
    ui.avar_dropdown.currentIndexChanged.connect(lambda index: obj.anim.update(var=obj.util.avars[index]))
    ui.aduration.editingFinished.connect(lambda: obj.anim.update(duration=float(ui.aduration.text())))
    # features
    for chk in ui.features.values():
        chk.stateChanged.connect(
            lambda _: obj.state.update(features={
                feat_name: {"gdf": obj.util.features[feat_name]["gdf"],
                            "color": obj.util.features[feat_name]["color"]}
                for feat_name, checkbox in ui.features.items() if checkbox.isChecked()
            }),
        )

def setup_folders() -> None:
    """Create necessary application folders if they do not already exist.

    This function ensures that the 'state' and 'output' directories
    are present for storing application state and output files.

    Returns:
        None

    """
    Path("state").mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(parents=True, exist_ok=True)

def setup_utility() -> SimpleNamespace:
    """Set up utility data and return a container with commonly used variables.

    This function initializes a SimpleNamespace containing:
    - Available color maps (cmaps) loaded from Matplotlib/CET.
    - Variables for plotting and animation (cvars, avars).
    - Geospatial features loaded from parquet files.
    - Preloaded map data for states, counties, CWAs, and congressional districts.

    Returns:
        SimpleNamespace: Namespace containing cmaps, cvars, avars, features, and maps.

    """
    util = SimpleNamespace()

    cmap_options = ["bgy", "CET_D8", "bjy", "CET_CBD2", "blues", "bmw", "bmy", "CET_L10", "gray", "dimgray", "kbc", "gouldian", "kgy", "fire", "CET_CBL1", "CET_CBL3", "CET_CBL4", "kb", "kg", "kr", "CET_CBTL3", "CET_CBTL1", "CET_L19", "CET_L17", "CET_L18"]

    util.cvars = ["seconds", "lon", "lat", "alt", "chi", "pdb", "flash_id"]
    util.avars = ["seconds", "lon", "lat", "alt", "chi", "pdb", "flash_id"]
    util.features = {
    "roads":  {"file": "assets/features/roads.parquet",  "color": "orange"},
    "rivers": {"file": "assets/features/rivers.parquet", "color": "blue"},
    "rails":  {"file": "assets/features/rails.parquet",  "color": "darkgray"},
    "urban":  {"file": "assets/features/urban.parquet",  "color": "red"}}
    for value in util.features.values():
        value["gdf"] = gpd.read_parquet(value["file"])
    util.cmaps = []
    for cmap in cmap_options:
        util.cmaps.append(plt.get_cmap(f"cet_{cmap}"))
    util.maps = []
    for file in ["assets/maps/state.parquet", "assets/maps/county.parquet", "assets/maps/cw.parquet", "assets/maps/cd.parquet"]:
        util.maps.append(gpd.read_parquet(file))

    return util

@dataclass(order=False)
class Animate:
    """Dataclass for managing animation state and control.

    Attributes:
        start_time (float): The start time of the animation in seconds.
        duration (float): Duration of the animation in seconds.
        active (bool): Whether the animation is currently active.
        timer (app.Timer): VisPy timer controlling the animation updates.
        var (str): The variable to animate.

    """

    start_time: float = field(default=0)
    duration: float = field(default=5.0)
    active: bool = field(default=False)
    timer: app.Timer = field(default_factory=lambda: app.Timer(interval="auto", start=False))
    var: str = field(default = "utc_sec")

    def update(self, **kwargs: object) -> None:
        """Update attributes of the animation and start it.

        This method accepts keyword arguments corresponding to
        attributes of the Animate dataclass. If a valid attribute
        is provided, it updates its value, logs the animation start,
        sets the start time, marks it as active, and starts the timer.

        Args:
            **kwargs: Arbitrary keyword arguments matching Animate
                attribute names and their new values.

        Returns:
            None

        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                self.__dict__[k] = v

        logger.info("Starting animation.")
        self.start_time = time.perf_counter()
        self.active = True
        self.timer.start()

@dataclass(order=False)
class PlotOptions:
    """Dataclass for managing plot options and visualization settings.

    Attributes:
        cvar (str): Variable used for coloring the plot.
        cmap (ListedColormap): Colormap used for the plot.
        features (dict): Dictionary of features to display on the map.
        map (gpd.GeoDataFrame): Geospatial map data for plotting.

    """

    cvar: str = field(default = "seconds")
    cmap: ListedColormap = field(default_factory = lambda: plt.get_cmap("cet_bgy"))
    features: dict = field(default_factory = dict)
    map: gpd.GeoDataFrame = field(default_factory=lambda: gpd.read_parquet("assets/maps/state.parquet"))

    def update(self, **kwargs: object) -> None:
        """Update attributes of the PlotOptions dataclass.

        Accepts keyword arguments corresponding to the attributes of
        the dataclass. Only valid attributes are updated.

        Args:
            **kwargs: Arbitrary keyword arguments matching PlotOptions
                attribute names and their new values.

        Returns:
            None

        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

@dataclass(order=False)
class State:
    """Dataclass for managing the application state, including data, plot options, and history.

    Attributes:
        all (pd.DataFrame): Main dataset used for plotting and analysis.
        stations (list[tuple]): List of station coordinates or identifiers.
        plot (pd.Series): Current plot data.
        plot_options (PlotOptions): Visualization options for plotting.
        replot (callable): Function to trigger a replot of the current state.
        history (deque): Circular buffer storing past states for undo functionality.
        future (deque): Circular buffer storing future states for redo functionality.
        _initialized (bool): Internal flag indicating if post-initialization is complete.
        gsd (pd.DataFrame): DataFrame containing ground strike data.
        gsd_mask (pd.Series): Mask applied to the ground strike data.

    """

    all: pd.DataFrame = field(default_factory = pd.DataFrame)
    stations: list[tuple] = field(default_factory = list)
    plot: pd.Series = field(default_factory = pd.Series)
    plot_options: PlotOptions = field(default_factory=PlotOptions)
    replot: callable = field(default=None, repr=False)
    history: deque = field(default=None)
    future: deque = field(default=None)
    _initialized: bool = field(init=False, default=False, repr=False)
    gsd: pd.DataFrame = field(default_factory = lambda: pd.DataFrame(columns=["seconds, lat, lon, alt, type"]))
    gsd_mask: pd.Series = field(default_factory=pd.Series)

    def __post_init__(self) -> None:
        """Initialize history and future buffers after dataclass fields are set.

        Returns:
            None

        """
        self.history = deque(maxlen=20)
        self.future = deque(maxlen=20)
        self._initialized = True

    def __copy__(self) -> Self:
        """Create a deep copy of the current state, preserving references for certain attributes.

        Returns:
            State: A copy of the current State instance.

        """
        new = self.__class__.__new__(self.__class__)
        logger.info("State was copied")
        for k, v in self.__dict__.items():
            if k in {"replot", "history", "future"}:
                new.__dict__[k] = v
            else:
                new.__dict__[k] = copy.deepcopy(v)
        return new

    def update(self, **kwargs: object) -> None:
        """Update attributes of the state or plot options and trigger a replot.

        Keyword arguments can correspond to any attribute of State or PlotOptions.
        If the lengths of `all` and `plot` match, the previous state is saved in
        `history` for undo functionality, and `future` is cleared.

        Args:
            **kwargs: Arbitrary keyword arguments matching State or PlotOptions
                attributes and their new values.

        Returns:
            None

        """
        if len(self.all) == len(self.plot):
            self.history.append(copy.copy(self))
            self.future.clear()

        for k, v in kwargs.items():
            if hasattr(self, k):
                self.__dict__[k] = v
            elif hasattr(self.plot_options, k):
                self.plot_options.update(**{k: v})

        self.replot()

    def __setattr__(self, name: object, value: object) -> None:
        """Override setattr to track changes in state for undo/redo functionality.

        Args:
            name (str): Attribute name to set.
            value (object): Value to assign to the attribute.

        Returns:
            None

        """
        if getattr(self, "_initialized", False) and len(self.all) == len(self.plot):
            self.history.append(copy.copy(self))
            self.future.clear()

        super().__setattr__(name, value)
