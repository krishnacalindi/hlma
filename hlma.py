import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget,  QStatusBar, QLabel, QSplitter, QComboBox, QCheckBox
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
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
        self.finished.emit(results)

class ImageWorker(QThread):
    finished = pyqtSignal(object)
    def __init__(self, lyl, cvar, cmap, map, features):
        super().__init__()
        self.lyl = lyl
        self.cvar = cvar
        self.cmap = cmap
        self.map = map
        self.features = features
    def run(self):
        imgs = []
        for i in tqdm(range(5), desc="Images rendered: "):
            imgs.append(QuickImage(self.lyl, self.cvar, self.cmap, self.map, self.features, i))
        self.finished.emit(imgs)
        
class XLMA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('XLMA')
        self.setWindowIcon(QIcon('assets/icons/xlma.svg'))
        
        # data holders
        self.lyl = None
        self.cmap = ["bgy", "CET_D8", "bjy", "CET_CBD2", "blues", "bmw", 
                     "bmy", "CET_L10", "gray", "dimgray", "kbc", "gouldian", 
                     "kgy", "fire", "CET_CBL1", "CET_CBL3", "CET_CBL4", "kb", 
                     "kg", "kr", "CET_CBTL3", "CET_CBTL1", 
                     "CET_L19", "CET_L17", "CET_L18"]
        self.cvar = ['seconds', 'lat', 'lon', 'alt', 'chi', 'pdb']
        self.map = ["state", "county", "cw", "cd"]
        
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
        
        self.option_layout.addWidget(self.cvar_label)
        self.option_layout.addWidget(self.cvar_dropdown)
        self.option_layout.addWidget(self.cmap_label)
        self.option_layout.addWidget(self.cmap_dropdown)
        self.option_layout.addWidget(self.map_label)
        self.option_layout.addWidget(self.map_dropdown)
        self.option_layout.addWidget(self.features_label)
        self.option_layout.addLayout(self.features_layout)
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
    
    def do_plot(self, lyl):
        self.lyl = lyl
        self.update_status("Drawing images...")
        self.worker = ImageWorker(self.lyl, self.cvar[self.cvar_dropdown.currentIndex()], self.cmap[self.cmap_dropdown.currentIndex()],
                                  self.map[self.map_dropdown.currentIndex()], [int(self.roads.isChecked()),int(self.rivers.isChecked()),
                                                                               int(self.rails.isChecked()),int(self.urban.isChecked())])
        self.worker.finished.connect(self.do_show)
        self.worker.start()
    
    def do_show(self, imgs):
        self.update_status("Ready")
        for i in reversed(range(self.view_layout.count())):
            item = self.view_layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        canvas = Plot(imgs)
        toolbar =  Nav(canvas, self)
        self.view_layout.addWidget(toolbar)
        self.view_layout.addWidget(canvas)
    
    def redraw(self):
        if self.lyl:
            self.do_plot(self.lyl) 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XLMA()
    window.show()
    sys.exit(app.exec())
