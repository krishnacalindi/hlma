import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget,  QStatusBar, QLabel, QSplitter, QComboBox, QCheckBox, QLineEdit, QDialog, QPushButton, QDialogButtonBox
from PyQt6.QtGui import QIcon, QAction, QDoubleValidator, QRegularExpressionValidator
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QRegularExpression
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from shapely import Polygon, vectorized
from matplotlib.dates import num2date
import webbrowser
from bts import OpenLylout, Plot, QuickImage, BlankPlot, Nav
import warnings
warnings.filterwarnings("ignore")

class OpenWorker(QThread):
    finished = pyqtSignal(object)
    def __init__(self, files):
        super().__init__()
        self.files = files
    def run(self):
        with ProcessPoolExecutor(max_workers=10) as executor:
            results = list(tqdm(executor.map(OpenLylout, self.files), total=len(self.files), desc="LYLOUT files processed: "))
        self.finished.emit(pd.concat(results, ignore_index=True))

class ImageWorker(QThread):
    finished = pyqtSignal(object)
    def __init__(self, lyl, cvar, cmap, map, features, extents):
        super().__init__()
        self.lyl = lyl
        self.cvar = cvar
        self.cmap = cmap
        self.map = map
        self.features = features
        self.extents = extents
    def run(self):
        imgs = QuickImage(self.lyl, self.cvar, self.cmap, self.map, self.features, self.extents)
        self.finished.emit(imgs)

class PolygonDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polygon")
        self.setWindowIcon(QIcon('assets/icons/keep.svg'))
        self.setModal(True)  # blocks all bg tasks while this is true!
        layout = QVBoxLayout()
        label = QLabel("Choose an option:")
        layout.addWidget(label)
        button_box = QDialogButtonBox(self)
        button_box.setOrientation(Qt.Orientation.Vertical)
        self.keep_button = QPushButton("Keep")
        self.remove_button = QPushButton("Remove")
        self.zoom_button = QPushButton("Zoom")
        self.cancel_button = QPushButton("Cancel")
        button_box.addButton(self.keep_button, QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.addButton(self.remove_button, QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.addButton(self.zoom_button, QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.addButton(self.cancel_button, QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.keep_button.clicked.connect(lambda: self.close(1))
        self.remove_button.clicked.connect(lambda: self.close(2))
        self.zoom_button.clicked.connect(lambda: self.close(3))
        self.cancel_button.clicked.connect(lambda: self.close(4))
    def close(self, choice):
        self.choice = choice
        self.accept()
    def get_choice(self):
        return self.choice
        
class HLMA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HLMA')
        self.setWindowIcon(QIcon('assets/icons/hlma.svg'))
        
        # data holders
        self.lyl = None
        self.cmap = ["bgy", "CET_D8", "bjy", "CET_CBD2", "blues", "bmw", 
                     "bmy", "CET_L10", "gray", "dimgray", "kbc", "gouldian", 
                     "kgy", "fire", "CET_CBL1", "CET_CBL3", "CET_CBL4", "kb", 
                     "kg", "kr", "CET_CBTL3", "CET_CBTL1", 
                     "CET_L19", "CET_L17", "CET_L18"]
        self.cvar = ['seconds', 'lat', 'lon', 'alt', 'chi', 'pdb']
        self.map = ["state", "county", "cw", "cd"]
        self.imgs = None
        
        self.layout = QHBoxLayout()
        splitter = QSplitter()
        
        self.option_layout = QVBoxLayout()
        self.option_widget = QWidget()
        self.option_widget.setLayout(self.option_layout)

        self.view_layout = QVBoxLayout()
        self.view_widget = QWidget()
        self.view_widget.setLayout(self.view_layout)

        splitter.addWidget(self.option_widget)
        splitter.addWidget(self.view_widget)
        splitter.setSizes([1, 1])
    
        self.layout.addWidget(splitter)
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)
        
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.status_label = QLabel('Ready')
        self.statusbar.addWidget(self.status_label, 1)

        self.menubar = self.menuBar()

        file_menu = self.menubar.addMenu('File')
        help_menu = self.menubar.addMenu('Help')
        
        open_action = QAction('Open', self)
        open_action.setIcon(QIcon('assets/icons/open.svg'))
        open_action.triggered.connect(self.do_open)
        file_menu.addAction(open_action)
        
        clear_action = QAction('Clear', self)
        clear_action.setIcon(QIcon('assets/icons/clear.svg'))
        clear_action.triggered.connect(self.do_clear)
        file_menu.addAction(clear_action)
        
        color_action = QAction('Colors', self)
        color_action.setIcon(QIcon('assets/icons/color.svg'))
        color_action.triggered.connect(self.do_color)
        help_menu.addAction(color_action)

        about_action = QAction('About', self)
        about_action.setIcon(QIcon('assets/icons/about.svg'))
        about_action.triggered.connect(self.do_about)
        help_menu.addAction(about_action)
        
        contact_action = QAction('Contact', self)
        contact_action.setIcon(QIcon('assets/icons/contact.svg'))
        contact_action.triggered.connect(self.do_contact)
        help_menu.addAction(contact_action)
        
        self.cvar_label = QLabel("Color by:")
        self.cvar_dropdown = QComboBox()
        self.cvar_dropdown.addItems(['Time', 'Longitude', 'Latitude', 'Altitude', 'Chi', 'Receiving power'])

        self.cmap_label = QLabel("Color map:")
        self.cmap_dropdown = QComboBox()
        for cmap_name in self.cmap:
            icon = QIcon(f"assets/colors/{cmap_name}.svg")
            self.cmap_dropdown.addItem(icon, cmap_name)
            
        self.map_label = QLabel("Map:")
        self.map_dropdown = QComboBox()
        self.map_dropdown.addItems(['State', 'County', 'NOAA County Warning Areas', '116 Congressional Districts'])
        
        self.features_layout = QHBoxLayout()
        self.features_label = QLabel('Features:')
        self.roads = QCheckBox('Roads')
        self.rivers = QCheckBox('Rivers')
        self.rails = QCheckBox('Rails')
        self.urban = QCheckBox('Urban area')
        self.roads.stateChanged.connect(self.redraw)
        self.rivers.stateChanged.connect(self.redraw)
        self.rails.stateChanged.connect(self.redraw)
        self.urban.stateChanged.connect(self.redraw)
        self.features_layout.addWidget(self.roads)
        self.features_layout.addWidget(self.rivers)
        self.features_layout.addWidget(self.rails)
        self.features_layout.addWidget(self.urban)
        
        self.cvar_dropdown.currentIndexChanged.connect(self.redraw)
        self.cmap_dropdown.currentIndexChanged.connect(self.redraw)
        self.map_dropdown.currentIndexChanged.connect(self.redraw)
        
        self.time_layout = QHBoxLayout()
        time_regex = QRegularExpression(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        time_validator = QRegularExpressionValidator(time_regex)
        self.timemin_label = QLabel("Minimum time:")
        self.timemin = QLineEdit()
        self.timemin.setText("yyyy-mm-dd hh:mm:ss")
        self.timemin.setValidator(time_validator)
        self.timemax_label = QLabel("Minimum time:")
        self.timemax = QLineEdit()
        self.timemax.setText("yyyy-mm-dd hh:mm:ss")
        self.timemax.setValidator(time_validator)
        self.time_layout.addWidget(self.timemin_label, 1)
        self.time_layout.addWidget(self.timemin)
        self.time_layout.addWidget(self.timemax_label, 1)
        self.time_layout.addWidget(self.timemax)
        
        self.lon_layout = QHBoxLayout()
        self.lonmin_label = QLabel("Minimum longitude:")
        self.lonmin = QLineEdit()
        self.lonmin.setText("-98.5")
        self.lonmin.setValidator(QDoubleValidator())
        self.lonmax_label = QLabel("Maximum longitude:")
        self.lonmax = QLineEdit()
        self.lonmax.setText("-91.5")
        self.lonmax.setValidator(QDoubleValidator())
        self.lon_layout.addWidget(self.lonmin_label, 1)
        self.lon_layout.addWidget(self.lonmin)
        self.lon_layout.addWidget(self.lonmax_label, 1)
        self.lon_layout.addWidget(self.lonmax)

        self.lat_layout = QHBoxLayout()
        self.latmin_label = QLabel("Minimum latitude:")
        self.latmin = QLineEdit()
        self.latmin.setText("26.0")
        self.latmin.setValidator(QDoubleValidator())
        self.latmax_label = QLabel("Maximum latitude:")
        self.latmax = QLineEdit()
        self.latmax.setText("33.0")
        self.latmax.setValidator(QDoubleValidator())
        self.lat_layout.addWidget(self.latmin_label, 1)
        self.lat_layout.addWidget(self.latmin)
        self.lat_layout.addWidget(self.latmax_label, 1)
        self.lat_layout.addWidget(self.latmax)

        self.alt_layout = QHBoxLayout()
        self.altmin_label = QLabel("Minimum altitude:")
        self.altmin = QLineEdit()
        self.altmin.setText("0.0")
        self.altmin.setValidator(QDoubleValidator())
        self.altmax_label = QLabel("Maximum altitude:")
        self.altmax = QLineEdit()
        self.altmax.setText("20.0")
        self.altmax.setValidator(QDoubleValidator())
        self.alt_layout.addWidget(self.altmin_label, 1)
        self.alt_layout.addWidget(self.altmin)
        self.alt_layout.addWidget(self.altmax_label, 1)
        self.alt_layout.addWidget(self.altmax)

        self.chi_layout = QHBoxLayout()
        self.chimin_label = QLabel("Minimum chi:")
        self.chimin = QLineEdit()
        self.chimin.setText("0.0")
        self.chimin.setValidator(QDoubleValidator())
        self.chimax_label = QLabel("Maximum chi:")
        self.chimax = QLineEdit()
        self.chimax.setText("2.0")
        self.chimax.setValidator(QDoubleValidator())
        self.chi_layout.addWidget(self.chimin_label, 1)
        self.chi_layout.addWidget(self.chimin)
        self.chi_layout.addWidget(self.chimax_label, 1)
        self.chi_layout.addWidget(self.chimax)

        self.pdb_layout = QHBoxLayout()
        self.pdbmin_label = QLabel("Minimum receiving power:")
        self.pdbmin = QLineEdit()
        self.pdbmin.setText("-60.0")
        self.pdbmin.setValidator(QDoubleValidator())
        self.pdbmax_label = QLabel("Maximum receiving power:")
        self.pdbmax = QLineEdit()
        self.pdbmax.setText("60.0")
        self.pdbmax.setValidator(QDoubleValidator())
        self.pdb_layout.addWidget(self.pdbmin_label, 1)
        self.pdb_layout.addWidget(self.pdbmin)
        self.pdb_layout.addWidget(self.pdbmax_label, 1)
        self.pdb_layout.addWidget(self.pdbmax)
        
        self.timemin.editingFinished.connect(self.redraw)
        self.timemax.editingFinished.connect(self.redraw)
        self.lonmin.editingFinished.connect(self.redraw)
        self.lonmax.editingFinished.connect(self.redraw)
        self.latmin.editingFinished.connect(self.redraw)
        self.latmax.editingFinished.connect(self.redraw)
        self.altmin.editingFinished.connect(self.redraw)
        self.altmax.editingFinished.connect(self.redraw)
        self.chimin.editingFinished.connect(self.redraw)
        self.chimax.editingFinished.connect(self.redraw)
        self.pdbmin.editingFinished.connect(self.redraw)
        self.pdbmax.editingFinished.connect(self.redraw)
        
        self.option_layout.addWidget(self.cvar_label)
        self.option_layout.addWidget(self.cvar_dropdown)
        self.option_layout.addWidget(self.cmap_label)
        self.option_layout.addWidget(self.cmap_dropdown)
        self.option_layout.addWidget(self.map_label)
        self.option_layout.addWidget(self.map_dropdown)
        self.option_layout.addWidget(self.features_label)
        self.option_layout.addLayout(self.features_layout)
        self.option_layout.addLayout(self.time_layout)
        self.option_layout.addLayout(self.lon_layout)
        self.option_layout.addLayout(self.lat_layout)
        self.option_layout.addLayout(self.alt_layout)
        self.option_layout.addLayout(self.chi_layout)
        self.option_layout.addLayout(self.pdb_layout)
        self.option_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        canvas = BlankPlot()
        toolbar =  Nav(canvas, self)
        self.view_layout.addWidget(toolbar)
        self.view_layout.addWidget(canvas)
        
        self.showMaximized()

    def update_status(self, text):
        self.status_label.setText(text)
    
    def do_open(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select LMA LYLOUT files", "", "Dat files (*.dat)")
        if files:
            self.update_status("Opening files...")
            self.worker = OpenWorker(files)
            self.worker.finished.connect(self.do_plot)
            self.worker.start()
    
    def do_clear(self):
        self.update_status("Ready")
        self.lyl = None
        for i in reversed(range(self.view_layout.count())):
            item = self.view_layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def do_about(self):
        webbrowser.open("https://lightning.tamu.edu/hlma/")
    
    def do_contact(self):
        webbrowser.open("https://artsci.tamu.edu/atmos-science/contact/profiles/timothy-logan.html")
    
    def do_color(self):
        webbrowser.open("https://colorcet.holoviz.org/user_guide/Continuous.html#linear-sequential-colormaps-for-plotting-magnitudes")
    
    def raise_flag(self, action):
        if action == "keep":
            self.remove = False
        elif action == "remove":
            self.remove = True
        
    def do_plot(self, lyl):
        self.lyl = lyl
        self.fdf = self.lyl
        self.masks = []
        self.popped_masks = []
        self.clicks = []
        self.dots = []
        self.lines = []
        self.prev_ax = None
        self.remove = False
        self.do_update(self.lyl)

    def do_update(self, lyl):
        self.update_status("Drawing images...")
        self.worker = ImageWorker(lyl, self.cvar[self.cvar_dropdown.currentIndex()], self.cmap[self.cmap_dropdown.currentIndex()], self.map[self.map_dropdown.currentIndex()], [int(self.roads.isChecked()),int(self.rivers.isChecked()), int(self.rails.isChecked()),int(self.urban.isChecked())], (self.timemin.text(), self.timemax.text(), float(self.lonmin.text()), float(self.lonmax.text()), float(self.latmin.text()), float(self.latmax.text()), float(self.altmin.text()), float(self.altmax.text()), float(self.chimin.text()), float(self.chimax.text()), float(self.pdbmin.text()), float(self.pdbmax.text())))
        self.worker.finished.connect(self.do_show)
        self.worker.start()

    def do_show(self, imgs):
        def on_click(event):

            # 0 is time-alt
            # 1 is lon-alt
            # 2 is sources
            # 3 is lon-lat
            # 4 is lat-alt

            # 0, 1 need vertical lines on click
            # 3 needs lines between points
            # 4 needs horizontal line on click
            ax = event.inaxes

            if event.inaxes and (self.prev_ax is None or self.prev_ax == event.inaxes.name): # Checks if inside a graph
                x, y = event.xdata, event.ydata
                if event.button == 1: # Left click
                    if event.inaxes.name == 0:
                        if len(self.clicks) < 2:
                            limit = event.inaxes.get_ylim() # get the ylimit for the line
                            line, *_ = ax.plot([x, x], limit, 'r--')
                            self.lines.append(line)

                            # Convert x to datetime
                            clicked_time = num2date(x)
                            self.clicks.append(clicked_time)
                            
                    if event.inaxes.name == 1:
                        if len(self.clicks) < 2:
                            limit = event.inaxes.get_ylim()
                            line, *_ = ax.plot([x, x], limit, 'r--')
                            self.lines.append(line)
                            self.clicks.append((x, limit[0]))

                    if event.inaxes.name == 3:
                        dot, *_ = ax.plot(x, y, 'ro', markersize=2)
                        self.dots.append(dot) # Grab the dot object
                        self.clicks.append((x, y))
                        if len(self.clicks) >= 2:
                            prev_x, prev_y = self.clicks[-2]
                            line, *_ = ax.plot([prev_x, x], [prev_y, y], 'r--') # Grab the Line2D object
                            self.lines.append(line)

                    if event.inaxes.name == 4:
                        if len(self.clicks) < 2:
                            limit = event.inaxes.get_xlim()
                            line, *_ = ax.plot(limit, [y,y], 'r--')
                            self.lines.append(line)
                            self.clicks.append([limit[0], y])

                    canvas.draw()
                    self.prev_ax = ax.name
                elif event.button == 3: # Right click
                    if len(self.clicks) > 1:
                        if self.prev_ax == 3:
                            first_x, first_y = self.clicks[0]
                            line, *_ = ax.plot([self.clicks[-1][0], first_x], [self.clicks[-1][1], first_y], 'g--') # This should close the figure
                        
                        for line in self.lines:
                            line.set_color('green')
                        for point in self.dots:
                            point.set_color('green')
                        self.lines.append(line) 
                        canvas.draw()
                        pd = PolygonDialog()
                        pd.exec()
                        print(pd.get_choice())
                        # pd.get_choice() will return the 1-4 for the thingy (1:keep,2:remove,3:zoom,4:cancel)
                        if pd.get_choice() == 1: # Keep
                            self.remove = False
                            self.polygon(self.prev_ax)
                        elif pd.get_choice() == 2: # Remove
                            self.remove = True
                            self.polygon(self.prev_ax)
                        elif pd.get_choice() == 3: # Zoom
                            self.remove = False
                            self.polygon(self.prev_ax)
                        # Implicit cancel means nothing is done just need to clear lines

                        self.prev_ax = None
                        # Clearing drawn points here
                        self.clicks.clear()
                        for line in self.lines:
                            self.lines.remove(line)
                        for dot in self.dots:
                            self.dots.remove(dot)
                        
                        self.lines.clear()
                        self.dots.clear()              
                        canvas.draw()            

            canvas.draw()

            canvas.draw()
        self.imgs = imgs
        if not imgs:
            self.update_status("No data to plot")
        else:
            self.update_status("Ready")
            for i in reversed(range(self.view_layout.count())):
                item = self.view_layout.itemAt(i)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            canvas = Plot(imgs)
            canvas.mpl_connect('button_press_event', on_click)
            toolbar =  Nav(canvas, self)
            self.view_layout.addWidget(toolbar)
            self.view_layout.addWidget(canvas)

    def redraw(self):
        if self.lyl:
            self.do_plot(self.lyl) 
            
    def polygon(self, num):
        self.do_show(self.imgs)
        if not hasattr(self, "fdf"):
            self.fdf = self.lyl.copy()
        if not hasattr(self, "masks"):
            self.masks = []

        if num == 0:
            print(self.clicks)
            min_x = min(self.clicks).replace(tzinfo=None)
            max_x = max(self.clicks).replace(tzinfo=None)

            mask = (self.fdf['datetime'] > min_x) & (self.fdf['datetime'] < max_x)
        if num == 1:
            x_values = [pt[0] for pt in self.clicks]  
            min_x = min(x_values)
            max_x = max(x_values)

            mask = (self.fdf['lon'] > min_x) & (self.fdf["lon"] < max_x)
        elif num == 3:
            polygon = Polygon(self.clicks)
            lon = self.fdf['lon'].to_numpy()
            lat = self.fdf['lat'].to_numpy()

            mask = vectorized.contains(polygon, lon, lat)
        elif num == 4:
            y_values = [pt[1] for pt in self.clicks]
            min_y = min(y_values)
            max_y = max(y_values)

            mask = (self.fdf['lat'] > min_y) & (self.fdf['lat'] < max_y)

        if self.remove:
            mask = ~mask

        self.fdf = self.fdf[mask]
        # and then when we call plots/etc we can check to see if the fdf is not none else we can send it in or sum ting else.

        if len(mask) < len(self.lyl):
            # Pad mask with false for later undo operations
            mask = np.pad(mask, (0, len(self.lyl) - len(mask)), constant_values=False)
        if not self.fdf.empty:
            self.masks.append(mask)
            print(self.masks)
            self.do_update(self.fdf)
        else:
            self.update_status("Polygon failed")

    def undo_filter(self):
        if self.masks:
            mask = self.masks.pop()
            self.popped_masks.append(mask)
            self.apply_filters()
        else:
            self.update_status("No filter to undo")
        
    def redo_filter(self):
        if self.popped_masks:
            redo_mask = self.popped_masks.pop()
            self.masks.append(redo_mask)
            self.apply_filters()
        else:
            self.update_status("No filter to redo")

    def apply_filters(self):
        if not self.masks:
            self.fdf = self.lyl.copy()
        else:
            stacked_masks = np.stack(self.masks)
            # Make a single mask to apply all at once
            combined_masks = np.logical_and.reduce(stacked_masks, axis=0)
            self.fdf = self.lyl[combined_masks]
        
        self.do_update(self.fdf)
        
        
if __name__ == "__main__": 
    app = QApplication(sys.argv)
    window = HLMA()
    window.show()
    sys.exit(app.exec())
