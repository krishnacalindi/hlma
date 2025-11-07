# global imports
import os

# pyqt
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget,  QLabel, QSplitter, QComboBox, QCheckBox, QLineEdit, QGridLayout, QDialogButtonBox, QPushButton, QDialog
from PyQt6.QtGui import QIcon, QAction, QDoubleValidator, QRegularExpressionValidator, QIntValidator, QKeySequence
from PyQt6.QtCore import Qt, QRegularExpression

# data imports
from dataclasses import dataclass, field
from types import SimpleNamespace
import geopandas as gpd
import pandas as pd
import copy
from collections import deque

# plot imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from vispy import scene
from vispy.scene import visuals, AxisWidget

# setting up logging
import logging
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("setup.py")
logger.setLevel(logging.DEBUG)

class LoadingDialog(QDialog):
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle('Please wait...')
        self.setModal(True) 
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(300, 100)
        layout = QVBoxLayout()
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)

def UI(obj):
    # UI
    ui = SimpleNamespace()
    
    # main splitter
    main_splitter = QSplitter()
    
    # left: options
    option_widget = QWidget()
    option_layout = QVBoxLayout()
    option_widget.setLayout(option_layout)

    # right: views
    ui.view_widget = QWidget()
    grid_plot = QGridLayout()
    ui.view_widget.setLayout(grid_plot)
    
    def grid_view_axes(grid):
        view = grid.add_view(0, 1)
        view.camera = scene.PanZoomCamera(aspect=None)
        x = AxisWidget(orientation='bottom', minor_tick_length=1, major_tick_length=3, tick_font_size=5, tick_label_margin=10, axis_width=1)
        y = AxisWidget(orientation='left', minor_tick_length=1, major_tick_length=3, tick_font_size=5, tick_label_margin=10, axis_width=1)
        grid.add_widget(x, 1, 1)
        grid.add_widget(y, 0, 0)
        x.link_view(view)
        y.link_view(view)
        y.axis._text.rotation = -90 # NOTE: makes y axis labels turn sideways
        y.width_max = 20
        x.height_max = 20
        view.stretch = (1, 1)
        return view
        
    ui.c0 = scene.SceneCanvas(keys=None, show=False, bgcolor='black')
    grid_plot.addWidget(ui.c0.native, 0, 0, 1, 2)
    grid = ui.c0.central_widget.add_grid()
    ui.v0 = grid_view_axes(grid)
    ui.s0 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.v0.add(ui.s0)
    ui.pd0 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl0 = visuals.Line(color='red', width=1)
    ui.v0.add(ui.pl0)
    ui.gs0 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)

    ui.c1 = scene.SceneCanvas(keys=None, show=False, bgcolor='black')
    grid_plot.addWidget(ui.c1.native, 1, 0)
    grid = ui.c1.central_widget.add_grid()
    ui.v1 = grid_view_axes(grid)
    ui.s1 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.v1.add(ui.s1)
    ui.pd1 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl1 = visuals.Line(color='red', width=1)
    ui.v1.add(ui.pl1)
    ui.gs1 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    

    ui.c2 = scene.SceneCanvas(keys=None, show=False, bgcolor='black')
    grid_plot.addWidget(ui.c2.native, 1, 1)
    grid = ui.c2.central_widget.add_grid()
    ui.v2 = grid_view_axes(grid)
    ui.v2.stretch = (1, 1)
    ui.hist = visuals.Line(color='white', width=1)
    ui.v2.add(ui.hist)
    ui.pd2 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl2 = visuals.Line(color='red', width=1)
    ui.v2.add(ui.pl2)
    
    ui.c3 = scene.SceneCanvas(keys=None, show=False, bgcolor='black')
    grid_plot.addWidget(ui.c3.native, 2, 0)
    grid = ui.c3.central_widget.add_grid()
    ui.v3 = grid_view_axes(grid)
    ui.map = visuals.Line(color='white', width=1)
    ui.v3.add(ui.map)
    ui.s3 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.v3.add(ui.s3)
    ui.pd3 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl3 = visuals.Line(color='red', width=1)
    ui.v3.add(ui.pl3)
    ui.gs3 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    
    
    ui.c4 = scene.SceneCanvas(keys=None, show=False, bgcolor='black')
    grid_plot.addWidget(ui.c4.native, 2, 1)
    grid = ui.c4.central_widget.add_grid()
    ui.v4 = grid_view_axes(grid)
    ui.s4 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.v4.add(ui.s4)
    ui.pd4 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.pl4 = visuals.Line(color='red', width=1)
    ui.v4.add(ui.pl4)
    ui.gs4 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    
    
    grid_plot.setRowStretch(0, 1)
    grid_plot.setRowStretch(1, 1)
    grid_plot.setRowStretch(2, 8)
    
    grid_plot.setColumnStretch(0, 8)
    grid_plot.setColumnStretch(1, 1)
    
    ui.v3.camera.link(ui.v1.camera, axis='x')
    ui.v3.camera.link(ui.v4.camera, axis='y')
    
    # main UI
    center = QWidget()
    main = QHBoxLayout()
    main_splitter.addWidget(option_widget)
    main_splitter.addWidget(ui.view_widget)
    main_splitter.setSizes([1, 1])
    main.addWidget(main_splitter)
    center.setLayout(main)
    obj.setCentralWidget(center)
    
    # menu bar
    menubar = obj.menuBar()
    import_menu = menubar.addMenu('Import')
    export_menu = menubar.addMenu('Export')
    options_menu = menubar.addMenu('Options')
    flash_menu = menubar.addMenu('Flash')
    help_menu = menubar.addMenu('Help')
    filter_menu = menubar.addMenu('Filter')
    # import menu
    ui.import_menu_lylout = QAction('LYLOUT', obj)
    ui.import_menu_lylout.setIcon(QIcon('assets/icons/lyl.svg'))
    import_menu.addAction(ui.import_menu_lylout)
    ui.import_menu_entln = QAction('ENTLN', obj)
    ui.import_menu_entln.setIcon(QIcon('assets/icons/entln.svg'))
    import_menu.addAction(ui.import_menu_entln) 
    ui.import_menu_state = QAction('State', obj)
    ui.import_menu_state.setIcon(QIcon('assets/icons/state.svg'))
    import_menu.addAction(ui.import_menu_state)
    # export menu
    ui.export_menu_dat = QAction('DAT', obj)
    ui.export_menu_dat.setIcon(QIcon('assets/icons/dat.svg'))
    export_menu.addAction(ui.export_menu_dat)
    ui.export_menu_parquet = QAction('Parquet', obj)
    ui.export_menu_parquet.setIcon(QIcon('assets/icons/parquet.svg'))
    export_menu.addAction(ui.export_menu_parquet)
    ui.export_menu_state = QAction('State', obj)
    ui.export_menu_state.setIcon(QIcon('assets/icons/state.svg'))
    export_menu.addAction(ui.export_menu_state)
    ui.export_menu_image = QAction('Image', obj)
    ui.export_menu_image.setIcon(QIcon('assets/icons/image.svg'))
    export_menu.addAction(ui.export_menu_image)
    # options menu
    ui.options_menu_draw = QAction('Draw', obj)
    ui.options_menu_draw.setIcon(QIcon('assets/icons/draw.svg'))
    ui.options_menu_draw.setShortcut(QKeySequence("Ctrl+D"))
    options_menu.addAction(ui.options_menu_draw)
    ui.options_menu_reset = QAction('Reset', obj)
    ui.options_menu_reset.setIcon(QIcon('assets/icons/reset.svg'))
    ui.options_menu_reset.setShortcut(QKeySequence("F5"))
    options_menu.addAction(ui.options_menu_reset)
    ui.options_menu_clear = QAction('Clear', obj)
    ui.options_menu_clear.setIcon(QIcon('assets/icons/clear.svg'))
    ui.options_menu_clear.setShortcut(QKeySequence("Delete"))
    options_menu.addAction(ui.options_menu_clear)
    # flash menu
    ui.flash_menu_dtd = QAction('Dot to Dot', obj)
    ui.flash_menu_dtd.setIcon(QIcon('assets/icons/dtd.svg'))
    flash_menu.addAction(ui.flash_menu_dtd)
    ui.flash_menu_mccaul = QAction('McCaul', obj)
    ui.flash_menu_mccaul.setIcon(QIcon('assets/icons/mcc.svg'))
    flash_menu.addAction(ui.flash_menu_mccaul)
    # help menu
    ui.help_menu_colors = QAction('Colors', obj)
    ui.help_menu_colors.setIcon(QIcon('assets/icons/color.svg'))
    help_menu.addAction(ui.help_menu_colors)
    ui.help_menu_about = QAction('About', obj)
    ui.help_menu_about.setIcon(QIcon('assets/icons/about.svg'))
    help_menu.addAction(ui.help_menu_about)
    ui.help_menu_contact = QAction('Contact', obj)
    ui.help_menu_contact.setIcon(QIcon('assets/icons/contact.svg'))
    help_menu.addAction(ui.help_menu_contact)
    # filter menu
    ui.filter_menu_keep = QAction('Keep', obj)
    ui.filter_menu_keep.setIcon(QIcon('assets/icons/keep.svg'))
    filter_menu.addAction(ui.filter_menu_keep)
    ui.filter_menu_remove = QAction('Remove', obj)
    ui.filter_menu_remove.setIcon(QIcon('assets/icons/remove.svg'))
    filter_menu.addAction(ui.filter_menu_remove)

    # options
    ui.cvar_dropdown = QComboBox()
    ui.cvar_dropdown.addItems(["Time", "Longitude", "Latitude", "Altitude", "Chi", "Receiving power", "Flash"])
    ui.map_dropdown = QComboBox()
    ui.map_dropdown.addItems(["State", "County", "NOAA CWAs", "Congressional Districts"])
    ui.cmap_dropdown = QComboBox()
    for cmap_name in ["bgy", "CET_D8", "bjy", "CET_CBD2", "blues", "bmw", "bmy", "CET_L10", "gray", "dimgray", "kbc", "gouldian", "kgy", "fire", "CET_CBL1", "CET_CBL3", "CET_CBL4", "kb", "kg", "kr", "CET_CBTL3", "CET_CBTL1", "CET_L19", "CET_L17", "CET_L18"]:
        icon = QIcon(f'assets/colors/{cmap_name}.svg')
        ui.cmap_dropdown.addItem(icon, cmap_name)
    map_features = QHBoxLayout()
    ui.features = {}
    ui.features['roads'] = QCheckBox('Roads')
    ui.features['rivers'] = QCheckBox('Rivers')
    ui.features['rails'] = QCheckBox('Rails')
    ui.features['urban'] = QCheckBox('Urban area')
    map_features.addWidget(ui.features['roads'])
    map_features.addWidget(ui.features['rivers'])
    map_features.addWidget(ui.features['rails'])
    map_features.addWidget( ui.features['urban'])
    
    # filters
    time_filter = QHBoxLayout()
    time_validator = QRegularExpressionValidator(QRegularExpression(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'))
    ui.timemin = QLineEdit()
    ui.timemin.setText('yyyy-mm-dd hh:mm:ss')
    ui.timemin.setValidator(time_validator)
    ui.timemax = QLineEdit()
    ui.timemax.setText('yyyy-mm-dd hh:mm:ss')
    ui.timemax.setValidator(time_validator)
    time_filter.addWidget(QLabel('Minimum time:'), 2)
    time_filter.addWidget(ui.timemin, 1)
    time_filter.addWidget(QLabel("Maximum time:"), 2)
    time_filter.addWidget(ui.timemax, 1)
    alt_filter = QHBoxLayout()
    ui.altmin = QLineEdit()
    ui.altmin.setText('0.0')
    ui.altmin.setValidator(QDoubleValidator())
    ui.altmax = QLineEdit()
    ui.altmax.setText('20.0')
    ui.altmax.setValidator(QDoubleValidator())
    alt_filter.addWidget(QLabel('Minimum altitude:'), 2)
    alt_filter.addWidget(ui.altmin, 1)
    alt_filter.addWidget(QLabel('Maximum altitude:'), 2)
    alt_filter.addWidget(ui.altmax, 1)
    chi_filter = QHBoxLayout()
    ui.chimin = QLineEdit()
    ui.chimin.setText('0.0')
    ui.chimin.setValidator(QDoubleValidator())
    ui.chimax = QLineEdit()
    ui.chimax.setText('2.0')
    ui.chimax.setValidator(QDoubleValidator())
    chi_filter.addWidget(QLabel('Minimum chi:'), 2)
    chi_filter.addWidget(ui.chimin, 1)
    chi_filter.addWidget(QLabel('Maximum chi:'), 2)
    chi_filter.addWidget(ui.chimax, 1)
    power_filter = QHBoxLayout()
    ui.powermin = QLineEdit()
    ui.powermin.setText('-60.0')
    ui.powermin.setValidator(QDoubleValidator())
    ui.powermax = QLineEdit()
    ui.powermax.setText('60.0')
    ui.powermax.setValidator(QDoubleValidator())
    power_filter.addWidget(QLabel('Minimum receiving power:'), 2)
    power_filter.addWidget(ui.powermin, 1)
    power_filter.addWidget(QLabel('Maximum receiving power:'), 2)
    power_filter.addWidget(ui.powermax, 1)
    stations_filter = QHBoxLayout()
    ui.stationsmin = QLineEdit()
    ui.stationsmin.setText('6')
    ui.stationsmin.setValidator(QIntValidator())
    stations_filter.addWidget(QLabel('Minimum number of stations:'), 2)
    stations_filter.addWidget(ui.stationsmin, 1)
    stations_filter.addStretch(3)
    
    # options layout
    option_layout.addWidget(QLabel('<h1>Filter options</h1>'))             
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
    option_layout.addWidget(QLabel('<h1>Map options</h1>'))    
    option_layout.addWidget(QLabel('Map:'))
    option_layout.addWidget(ui.map_dropdown)
    option_layout.addStretch(1)
    option_layout.addWidget(QLabel('Features:'))
    option_layout.addLayout(map_features)
    option_layout.addStretch(2)
    option_layout.addWidget(QLabel('<h1>Color options</h1>'))  
    option_layout.addWidget(QLabel('Color by:'))
    option_layout.addWidget(ui.cvar_dropdown)
    option_layout.addStretch(1)
    option_layout.addWidget(QLabel('Color map:'))
    option_layout.addWidget(ui.cmap_dropdown)
    option_layout.addStretch(2)
    option_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    
    # returning ui
    return ui
  
def Connections(obj, ui: SimpleNamespace):
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
    ui.filter_menu_keep.triggered.connect(lambda val: obj.polyfilter.update_filter(False))
    ui.filter_menu_remove.triggered.connect(lambda val: obj.polyfilter.update_filter(True))
    
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
    # features
    for _, chk in ui.features.items():
        chk.stateChanged.connect(
            lambda _: obj.state.update(features={
                feat_name: {'gdf': obj.util.features[feat_name]['gdf'],
                            'color': obj.util.features[feat_name]['color']}
                for feat_name, checkbox in ui.features.items() if checkbox.isChecked()
            })
        )

def Folders():
    os.makedirs('state', exist_ok=True)
    os.makedirs('output', exist_ok=True)

def Utility():
    util = SimpleNamespace()
    
    cmap_options = ["bgy", "CET_D8", "bjy", "CET_CBD2", "blues", "bmw", "bmy", "CET_L10", "gray", "dimgray", "kbc", "gouldian", "kgy", "fire", "CET_CBL1", "CET_CBL3", "CET_CBL4", "kb", "kg", "kr", "CET_CBTL3", "CET_CBTL1", "CET_L19", "CET_L17", "CET_L18"]
    
    util.cvars = ["seconds", "lon", "lat", "alt", "chi", "pdb", "flash_id"]
    util.features = {
    'roads':  {'file': 'assets/features/roads.parquet',  'color': 'orange'},
    'rivers': {'file': 'assets/features/rivers.parquet', 'color': 'blue'},
    'rails':  {'file': 'assets/features/rails.parquet',  'color': 'darkgray'},
    'urban':  {'file': 'assets/features/urban.parquet',  'color': 'red'}}
    for _, value in util.features.items():
        value['gdf'] = gpd.read_parquet(value['file'])
    util.cmaps = []
    for cmap in cmap_options:
        util.cmaps.append(plt.get_cmap(f'cet_{cmap}'))
    util.maps = []
    for file in ['assets/maps/state.parquet', 'assets/maps/county.parquet', 'assets/maps/cw.parquet', 'assets/maps/cd.parquet']:
        util.maps.append(gpd.read_parquet(file))
    
    return util

@dataclass(order=False)
class PlotOptions():
    cvar: str = field(default = "seconds")
    cmap: ListedColormap = field(default_factory = lambda: plt.get_cmap("cet_bgy"))
    features: dict = field(default_factory = dict)
    map: gpd.GeoDataFrame = field(default_factory=lambda: gpd.read_parquet('assets/maps/state.parquet'))
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

@dataclass(order=False)
class State:
    all: pd.DataFrame = field(default_factory = pd.DataFrame)
    stations: list[tuple] = field(default_factory = list)
    plot: pd.Series = field(default_factory = pd.Series)
    plot_options: PlotOptions = field(default_factory=PlotOptions)
    replot: callable = field(default=None, repr=False)
    history: deque = field(default=None)
    future: deque = field(default=None)
    _initialized: bool = field(init=False, default=False, repr=False)
    gsd: pd.DataFrame = field(default_factory = pd.DataFrame)
    gsd_mask: pd.Series = field(default_factory=pd.Series)

    def __post_init__(self):
        self.history = deque(maxlen=20)
        self.future = deque(maxlen=20)
        self._initialized = True
    
    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        logger.info(f"State was copied")
        for k, v in self.__dict__.items():
            if k in {"replot", "history", "future"}:
                new.__dict__[k] = v
            else:
                new.__dict__[k] = copy.deepcopy(v)
        return new
    
    def update(self, **kwargs):
        if len(self.all) == len(self.plot):  
            self.history.append(copy.copy(self))
            self.future.clear()
        
        for k, v in kwargs.items():
            if hasattr(self, k):
                self.__dict__[k] = v
            elif hasattr(self.plot_options, k):
                self.plot_options.update(**{k: v})
        
        self.replot()
    
    def __setattr__(self, name, value):
        if getattr(self, "_initialized", False):
            if len(self.all) == len(self.plot):  
                self.history.append(copy.copy(self))
                self.future.clear()
        
        super().__setattr__(name, value)
