# global imports
import os

# pyqt
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget,  QLabel, QSplitter, QComboBox, QCheckBox, QLineEdit, QGridLayout
from PyQt6.QtGui import QIcon, QAction, QDoubleValidator, QRegularExpressionValidator, QIntValidator, QKeySequence
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt6.QtCore import Qt, QRegularExpression

# data imports
from dataclasses import dataclass, field
from types import SimpleNamespace
import geopandas as gpd
import pandas as pd

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
        x = AxisWidget(orientation='bottom', minor_tick_length=1, major_tick_length=3, tick_font_size=8, tick_label_margin=10, axis_width=1)
        y = AxisWidget(orientation='left', minor_tick_length=1, major_tick_length=3, tick_font_size=8, tick_label_margin=10, axis_width=1)
        grid.add_widget(x, 1, 1)
        grid.add_widget(y, 0, 0)
        x.link_view(view)
        y.link_view(view)
        y.axis._text.rotation = -90 # NOTE: makes y axis labels turn sideways
        y.width_max = 20
        x.height_max = 20
        view.stretch = (1, 1)
        return view
        
    canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor='black')
    grid_plot.addWidget(canvas.native, 0, 0, 1, 2)
    grid = canvas.central_widget.add_grid()
    ui.v0 = grid_view_axes(grid)
    ui.s0 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.v0.add(ui.s0)

    canvas.events.mouse_press.connect(lambda ev: on_click(ev, view_index=0))

    canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor='black')
    grid_plot.addWidget(canvas.native, 1, 0)
    grid = canvas.central_widget.add_grid()
    ui.v1 = grid_view_axes(grid)
    ui.s1 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.v1.add(ui.s1)
    
    canvas.events.mouse_press.connect(lambda ev: on_click(ev, view_index=1))

    canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor='black')
    grid_plot.addWidget(canvas.native, 1, 1)
    grid = canvas.central_widget.add_grid()
    ui.v2 = grid_view_axes(grid)
    ui.v2.stretch = (1, 1)
    ui.hist = visuals.Line(color='white', width=1)
    ui.v2.add(ui.hist)

    canvas.events.mouse_press.connect(lambda ev: on_click(ev, view_index=2))
    
    canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor='black')
    grid_plot.addWidget(canvas.native, 2, 0)
    grid = canvas.central_widget.add_grid()
    ui.v3 = grid_view_axes(grid)
    ui.map = visuals.Line(color='white', width=1)
    ui.v3.add(ui.map)
    ui.s3 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.v3.add(ui.s3)

    canvas.events.mouse_press.connect(lambda ev: on_click(ev, view_index=3))

    canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor='black')
    grid_plot.addWidget(canvas.native, 2, 1)
    grid = canvas.central_widget.add_grid()
    ui.v4 = grid_view_axes(grid)
    ui.s4 = visuals.Markers(spherical=True, edge_width=0, light_position=(0, 0, 1), light_ambient=0.9)
    ui.v4.add(ui.s4)

    canvas.events.mouse_press.connect(lambda ev: on_click(ev, view_index=4))
    
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
    ui.options_menu_clear = QAction('Clear', obj)
    ui.options_menu_clear.setIcon(QIcon('assets/icons/clear.svg'))
    options_menu.addAction(ui.options_menu_clear)
    ui.options_menu_reset = QAction('Reset', obj)
    ui.options_menu_reset.setIcon(QIcon('assets/icons/reset.svg'))
    ui.options_menu_reset.setShortcut(QKeySequence("F5"))
    options_menu.addAction(ui.options_menu_reset)
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
    lon_filter = QHBoxLayout()
    ui.lonmin = QLineEdit()
    ui.lonmin.setText('-98.5')
    ui.lonmin.setValidator(QDoubleValidator())
    ui.lonmax = QLineEdit()
    ui.lonmax.setText('-91.5')
    ui.lonmax.setValidator(QDoubleValidator())
    lon_filter.addWidget(QLabel('Minimum longitude:'), 2)
    lon_filter.addWidget(ui.lonmin, 1)
    lon_filter.addWidget(QLabel('Maximum longitude:'), 2)
    lon_filter.addWidget(ui.lonmax, 1)
    lat_filter = QHBoxLayout()
    ui.latmin = QLineEdit()
    ui.latmin.setText('26.0')
    ui.latmin.setValidator(QDoubleValidator())
    ui.latmax = QLineEdit()
    ui.latmax.setText('33.0')
    ui.latmax.setValidator(QDoubleValidator())
    lat_filter.addWidget(QLabel('Minimum latitude:'), 2)
    lat_filter.addWidget(ui.latmin, 1)
    lat_filter.addWidget(QLabel('Maximum latitude:'), 2)
    lat_filter.addWidget(ui.latmax, 1)
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
    option_layout.addLayout(lon_filter)
    option_layout.addStretch(1)
    option_layout.addLayout(lat_filter)
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
    # ui.import_menu_entln.triggered.connect(obj.to_be_implemented)
    ui.import_menu_state.triggered.connect(obj.import_state)
    ui.export_menu_dat.triggered.connect(obj.export_dat)
    ui.export_menu_parquet.triggered.connect(obj.export_parquet)
    ui.export_menu_state.triggered.connect(obj.export_state)
    ui.export_menu_image.triggered.connect(obj.export_image)
    ui.options_menu_clear.triggered.connect(obj.options_clear)
    ui.options_menu_reset.triggered.connect(obj.options_reset)
    ui.flash_menu_dtd.triggered.connect(obj.flash_dtd)
    ui.flash_menu_mccaul.triggered.connect(obj.flash_mccaul)
    ui.help_menu_colors.triggered.connect(obj.help_color)
    ui.help_menu_about.triggered.connect(obj.help_about)
    ui.help_menu_contact.triggered.connect(obj.help_contact)
    
    # filters
    ui.timemin.editingFinished.connect(obj.filter)
    ui.timemax.editingFinished.connect(obj.filter)
    ui.lonmin.editingFinished.connect(obj.filter)
    ui.lonmax.editingFinished.connect(obj.filter)
    ui.latmin.editingFinished.connect(obj.filter)
    ui.latmax.editingFinished.connect(obj.filter)
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

def on_click(ui, event, view_index):
    pos = event.pos
    view = ui.__getattribute__(f'v{view_index}')
    transform = view.scene.transform
    x, y = transform.imap(pos)[:2]

    print(f"Clicked on view {view_index}: x={x}, y={y}")
    if event.button == 1: # Left click
        if view_index == 0:
            handle_click_0(event)
        elif view_index == 1:
            handle_click_1(event)
        elif view_index == 2:
            pass # As long as this is the sources graph, there should be no graphing on it
        elif view_index == 3:
            handle_click_3(event)
        elif view_index == 4:
            handle_click_4(event)
        else:
            # uh oh
            pass
    elif event.button == 3:
        if len(self.clicks) > 1:
            if view_index == 3:
                # Close the polygon with a final line
                first_x, first_y = self.clicks[0]
                last_x, last_y = self.clicks[-1]
                line = visuals.Line(
                    pos=np.array([[last_x, last_y], [first_x, first_y]]),
                    color='green',
                    width=1,
                    method='gl'
                )
                view.add(line)
                self.lines.append(line)

            # Turn all visuals green
            for line in self.lines:
                if hasattr(line, "set_data"):
                    line.set_data(pos=line._pos, color='green')
            for dot in self.dots:
                dot.set_data(pos=dot._data['a_position'], face_color='green')

            # Store the current axis index for use in polygon logic
            self.prev_ax = view_index

            # Simulate dialog (replace later with real Qt or vispy UI)
            choice = prompt_polygon_action()

            if choice == 1:  # Keep
                self.remove = False
                self.polygon(self.prev_ax)
            elif choice == 2:  # Remove
                self.remove = True
                self.polygon(self.prev_ax)
            # 3 is not needed since zoom is now interactive
            elif choice == 4:  # Cancel
                self.plot()

            self.prev_ax = None
            clear_polygon_visuals(view)

def clear_polygon_visuals(self, view):
    for line in self.lines:
        if line.parent is not None:
            view.remove(line)
    for dot in self.dots:
        if dot.parent is not None:
            view.remove(dot)

    self.lines.clear()
    self.dots.clear()
    self.clicks.clear()

def prompt_polygon_action(self):
    # This only uses terminal, but it will suffice until we get a UI for this
    print("\nPolygon completed. Choose action:")
    print("1: Keep")
    print("2: Remove")
    print("3: Zoom")
    print("4: Cancel")
    while True:
        try:
            choice = int(input("Enter choice [1-4]: "))
            if choice in (1, 2, 3, 4):
                return choice
        except ValueError:
            pass
        print("Invalid input, try again.")


def handle_click_0(self, x, view):
    if len(self.clicks) < 2:
        _, _, y0, y1 = get_view_bounds(view)
        line = visuals.Line(
            pos=np.array([[x, y0], [x, y1]]),
            color='red',
            width=1,
            method='gl'
        )

        view.add(line)
        self.lines.append(line)

        # Convert x to datetime
        clicked_time = num2date(x)
        self.clicks.append(clicked_time)


def handle_click_1(self, x, view):
    if len(self.clicks) < 2:
        _, _, y0, y1 = get_view_bounds(view)
        line = visuals.Line(
            pos=np.array([[x, y0], [x, y1]]),
            color='red',
            width=1,
            method='gl'
        )

        view.add(line)
        self.lines.append(line)
        self.clicks.append((x, y0))

    
def handle_click_3(self, x, y, view):
    dot = visuals.Markers()
    dot.set_data(pos=np.array([[x, y]]), face_color='red', size=5)
    view.add(dot)
    self.dots.append(dot)
    self.clicks.append((x, y))

    if len(self.clicks) >= 2:
        prev_x, prev_y = self.clicks[-2]
        line = visuals.Line(
            pos=np.array([[prev_x, prev_y], [x, y]]),
            color='red',
            width=1,
            method='gl'
        )

        view.add(line)
        self.lines.append(line)

def handle_click_4(self, y, view):
    x0, x1, _, _ = get_view_bounds(view)
    if len(self.clicks) < 2:
        line = visuals.Line(
            pos=np.array([[x0, y], [x1, y]]),
            color='red',
            width=1,
            method='gl'
        )

        view.add(line)
        self.lines.append(line)
        self.clicks.append([x0, y])


def get_view_bounds(view):
    cam = view.camera
    x0, y0, width, height = cam.get_state()['rect']
    return x0, x0 + width, y0, y0 + height


def Folders():
    os.makedirs('state', exist_ok=True)
    os.makedirs('output', exist_ok=True)

def Utility():
    util = SimpleNamespace()
    
    cmap_options = ["bgy", "CET_D8", "bjy", "CET_CBD2", "blues", "bmw", "bmy", "CET_L10", "gray", "dimgray", "kbc", "gouldian", "kgy", "fire", "CET_CBL1", "CET_CBL3", "CET_CBL4", "kb", "kg", "kr", "CET_CBTL3", "CET_CBTL1", "CET_L19", "CET_L17", "CET_L18"]
    
    util.cvars = ["utc_sec", "lon", "lat", "alt", "chi", "pdb", "flash_id"]
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
    for file in ['assets/maps/cd.parquet', 'assets/maps/county.parquet', 'assets/maps/cw.parquet', 'assets/maps/state.parquet']:
        util.maps.append(gpd.read_parquet(file))
    
    return util

@dataclass(order=False)
class PlotOptions():
    cvar: str = field(default = "utc_sec")
    cmap: ListedColormap = field(default_factory = lambda: plt.get_cmap("cet_bgy"))
    lon_max: float = field(default = -92.0)
    lon_min: float = field(default = -98.0)
    lat_min: float = field(default = 27.0)
    lat_max: float = field(default = 33.0)
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
    canvas: FigureCanvasQTAgg = field(default_factory=lambda: FigureCanvasQTAgg(plt.figure(figsize=(10, 12))))
    plot_options: PlotOptions = field(default_factory=PlotOptions)
    
    replot: callable = field(default=None, repr=False)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif hasattr(self.plot_options, k):
                self.plot_options.update(**{k: v})
        self.replot()
