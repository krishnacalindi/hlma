import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget,  QStatusBar, QLabel, QSplitter, QComboBox, QCheckBox, QLineEdit, QFrame
from PyQt6.QtGui import QIcon, QAction, QDoubleValidator, QRegularExpressionValidator, QIntValidator
from PyQt6.QtCore import Qt, QRegularExpression
import webbrowser
import warnings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
warnings.filterwarnings('ignore')

from bts import OpenLylout, QuickImage, BlankPlot

class HLMA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HLMA')
        self.setWindowIcon(QIcon('assets/icons/hlma.svg'))
        
        # data holders
        self.og = None
        self.lyl = None
        self.cmap = ['bgy', 'CET_D8', 'bjy', 'CET_CBD2', 'blues', 'bmw', 
                     'bmy', 'CET_L10', 'gray', 'dimgray', 'kbc', 'gouldian', 
                     'kgy', 'fire', 'CET_CBL1', 'CET_CBL3', 'CET_CBL4', 'kb', 
                     'kg', 'kr', 'CET_CBTL3', 'CET_CBTL1', 
                     'CET_L19', 'CET_L17', 'CET_L18']
        self.cvar = ['utc_sec', 'lon', 'lat', 'alt', 'chi', 'pdb']
        self.map = ['state', 'county', 'cw', 'cd']
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
        
        self.cvar_label = QLabel('Color by:')
        self.cvar_dropdown = QComboBox()
        self.cvar_dropdown.addItems(['Time', 'Longitude', 'Latitude', 'Altitude', 'Chi', 'Receiving power'])

        self.cmap_label = QLabel('Color map:')
        self.cmap_dropdown = QComboBox()
        for cmap_name in self.cmap:
            icon = QIcon(f'assets/colors/{cmap_name}.svg')
            self.cmap_dropdown.addItem(icon, cmap_name)
            
        self.map_label = QLabel('Map:')
        self.map_dropdown = QComboBox()
        self.map_dropdown.addItems(['State', 'County', 'NOAA County Warning Areas', '116 Congressional Districts'])
        
        self.features_layout = QHBoxLayout()
        self.features_label = QLabel('Features:')
        self.roads = QCheckBox('Roads')
        self.rivers = QCheckBox('Rivers')
        self.rails = QCheckBox('Rails')
        self.urban = QCheckBox('Urban area')
        self.roads.stateChanged.connect(self.do_filter)
        self.rivers.stateChanged.connect(self.do_filter)
        self.rails.stateChanged.connect(self.do_filter)
        self.urban.stateChanged.connect(self.do_filter)
        self.features_layout.addWidget(self.roads)
        self.features_layout.addWidget(self.rivers)
        self.features_layout.addWidget(self.rails)
        self.features_layout.addWidget(self.urban)
        
        self.cvar_dropdown.currentIndexChanged.connect(self.do_filter)
        self.cmap_dropdown.currentIndexChanged.connect(self.do_filter)
        self.map_dropdown.currentIndexChanged.connect(self.do_filter)
        
        self.time_layout = QHBoxLayout()
        time_regex = QRegularExpression(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
        time_validator = QRegularExpressionValidator(time_regex)
        self.timemin_label = QLabel('Minimum time:')
        self.timemin = QLineEdit()
        self.timemin.setText('yyyy-mm-dd hh:mm:ss')
        self.timemin.setValidator(time_validator)
        self.timemax_label = QLabel('Minimum time:')
        self.timemax = QLineEdit()
        self.timemax.setText('yyyy-mm-dd hh:mm:ss')
        self.timemax.setValidator(time_validator)
        self.time_layout.addWidget(self.timemin_label, 2)
        self.time_layout.addWidget(self.timemin, 1)
        self.time_layout.addWidget(self.timemax_label, 2)
        self.time_layout.addWidget(self.timemax, 1)
        
        self.lon_layout = QHBoxLayout()
        self.lonmin_label = QLabel('Minimum longitude:')
        self.lonmin = QLineEdit()
        self.lonmin.setText('-98.5')
        self.lonmin.setValidator(QDoubleValidator())
        self.lonmax_label = QLabel('Maximum longitude:')
        self.lonmax = QLineEdit()
        self.lonmax.setText('-91.5')
        self.lonmax.setValidator(QDoubleValidator())
        self.lon_layout.addWidget(self.lonmin_label, 2)
        self.lon_layout.addWidget(self.lonmin, 1)
        self.lon_layout.addWidget(self.lonmax_label, 2)
        self.lon_layout.addWidget(self.lonmax, 1)

        self.lat_layout = QHBoxLayout()
        self.latmin_label = QLabel('Minimum latitude:')
        self.latmin = QLineEdit()
        self.latmin.setText('26.0')
        self.latmin.setValidator(QDoubleValidator())
        self.latmax_label = QLabel('Maximum latitude:')
        self.latmax = QLineEdit()
        self.latmax.setText('33.0')
        self.latmax.setValidator(QDoubleValidator())
        self.lat_layout.addWidget(self.latmin_label, 2)
        self.lat_layout.addWidget(self.latmin, 1)
        self.lat_layout.addWidget(self.latmax_label, 2)
        self.lat_layout.addWidget(self.latmax, 1)

        self.alt_layout = QHBoxLayout()
        self.altmin_label = QLabel('Minimum altitude:')
        self.altmin = QLineEdit()
        self.altmin.setText('0.0')
        self.altmin.setValidator(QDoubleValidator())
        self.altmax_label = QLabel('Maximum altitude:')
        self.altmax = QLineEdit()
        self.altmax.setText('20.0')
        self.altmax.setValidator(QDoubleValidator())
        self.alt_layout.addWidget(self.altmin_label, 2)
        self.alt_layout.addWidget(self.altmin, 1)
        self.alt_layout.addWidget(self.altmax_label, 2)
        self.alt_layout.addWidget(self.altmax, 1)

        self.chi_layout = QHBoxLayout()
        self.chimin_label = QLabel('Minimum chi:')
        self.chimin = QLineEdit()
        self.chimin.setText('0.0')
        self.chimin.setValidator(QDoubleValidator())
        self.chimax_label = QLabel('Maximum chi:')
        self.chimax = QLineEdit()
        self.chimax.setText('2.0')
        self.chimax.setValidator(QDoubleValidator())
        self.chi_layout.addWidget(self.chimin_label, 2)
        self.chi_layout.addWidget(self.chimin, 1)
        self.chi_layout.addWidget(self.chimax_label, 2)
        self.chi_layout.addWidget(self.chimax, 1)

        self.pdb_layout = QHBoxLayout()
        self.pdbmin_label = QLabel('Minimum receiving power:')
        self.pdbmin = QLineEdit()
        self.pdbmin.setText('-60.0')
        self.pdbmin.setValidator(QDoubleValidator())
        self.pdbmax_label = QLabel('Maximum receiving power:')
        self.pdbmax = QLineEdit()
        self.pdbmax.setText('60.0')
        self.pdbmax.setValidator(QDoubleValidator())
        self.pdb_layout.addWidget(self.pdbmin_label, 2)
        self.pdb_layout.addWidget(self.pdbmin, 1)
        self.pdb_layout.addWidget(self.pdbmax_label, 2)
        self.pdb_layout.addWidget(self.pdbmax, 1)
        
        self.statnum_layout = QHBoxLayout()
        self.statnummin_label = QLabel('Minimum number of stations:')
        self.statnumin = QLineEdit()
        self.statnumin.setText('6')
        self.statnumin.setValidator(QIntValidator())
        self.statnum_layout.addWidget(self.statnummin_label, 2)
        self.statnum_layout.addWidget(self.statnumin, 1)
        self.statnum_layout.addStretch(3)
                
        self.timemin.editingFinished.connect(self.do_filter)
        self.timemax.editingFinished.connect(self.do_filter)
        self.lonmin.editingFinished.connect(self.do_filter)
        self.lonmax.editingFinished.connect(self.do_filter)
        self.latmin.editingFinished.connect(self.do_filter)
        self.latmax.editingFinished.connect(self.do_filter)
        self.altmin.editingFinished.connect(self.do_filter)
        self.altmax.editingFinished.connect(self.do_filter)
        self.chimin.editingFinished.connect(self.do_filter)
        self.chimax.editingFinished.connect(self.do_filter)
        self.pdbmin.editingFinished.connect(self.do_filter)
        self.pdbmax.editingFinished.connect(self.do_filter)
        self.statnumin.editingFinished.connect(self.do_filter)
        
        self.option_layout.addWidget(QLabel('<h1>Filter options</h1>'))             
        self.option_layout.addLayout(self.time_layout)
        self.option_layout.addStretch(1)
        self.option_layout.addLayout(self.lon_layout)
        self.option_layout.addStretch(1)
        self.option_layout.addLayout(self.lat_layout)
        self.option_layout.addStretch(1)
        self.option_layout.addLayout(self.alt_layout)
        self.option_layout.addStretch(1)
        self.option_layout.addLayout(self.chi_layout)
        self.option_layout.addStretch(1)
        self.option_layout.addLayout(self.pdb_layout)
        self.option_layout.addStretch(1)
        self.option_layout.addLayout(self.statnum_layout)
        self.option_layout.addStretch(2)
        
        self.option_layout.addWidget(QLabel('<h1>Map options</h1>'))    
        self.option_layout.addWidget(self.map_label)
        self.option_layout.addWidget(self.map_dropdown)
        self.option_layout.addStretch(1)
        self.option_layout.addWidget(self.features_label)
        self.option_layout.addLayout(self.features_layout)
        self.option_layout.addStretch(2)
        
        self.option_layout.addWidget(QLabel('<h1>Color options</h1>'))  
        self.option_layout.addWidget(self.cvar_label)
        self.option_layout.addWidget(self.cvar_dropdown)
        self.option_layout.addStretch(1)
        self.option_layout.addWidget(self.cmap_label)
        self.option_layout.addWidget(self.cmap_dropdown)
        self.option_layout.addStretch(2)
        
        self.option_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.do_blank()
        self.view_widget.setFocus()
        self.showMaximized()

    def update_status(self, text):
        self.status_label.setText(text)
    
    def do_open(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select LYLOUT files', '', 'Dat files (*.dat)')
        if files:
            self.update_status('Opening files...')
            self.og, failed_files = OpenLylout(files)
            if self.og is None:
                print('❌ All LYLOUT files were not processed due to errors.')
            elif failed_files:
                print('❌ Following LYLOUT files were not processed due to errors:')
                for f in failed_files:
                    print(f)
            else:
                print('✅ All LYLOUT files were opened successfully.')
            self.timemin.setText(self.og['datetime'].min().floor('s').strftime('%Y-%m-%d %H:%M:%S'))
            self.timemax.setText(self.og['datetime'].max().ceil('s').strftime('%Y-%m-%d %H:%M:%S'))
            self.do_filter()

    def do_clear(self):
        self.update_status('Ready')
        for i in reversed(range(self.view_layout.count())):
            item = self.view_layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def do_about(self):
        webbrowser.open('https://lightning.tamu.edu/hlma/')
    
    def do_contact(self):
        webbrowser.open('https://artsci.tamu.edu/atmos-science/contact/profiles/timothy-logan.html')
    
    def do_color(self):
        webbrowser.open('https://colorcet.holoviz.org/user_guide/Continuous.html#linear-sequential-colormaps-for-plotting-magnitudes')
    
    def do_blank(self):
        fig = BlankPlot()
        canvas = FigureCanvasQTAgg(fig)
        self.view_layout.addWidget(canvas)

    def do_filter(self):
        print('⏳ Filtering data.')
        if self.og is None:
            return
        tm_min, tm_max = self.timemin.text(), self.timemax.text()
        lon_min, lon_max = float(self.lonmin.text()), float(self.lonmax.text())
        lat_min, lat_max = float(self.latmin.text()), float(self.latmax.text())
        alt_min, alt_max = float(self.altmin.text()) * 1000, float(self.altmax.text()) * 1000
        chi_min, chi_max = float(self.chimin.text()), float(self.chimax.text())
        pdb_min, pdb_max = float(self.pdbmin.text()), float(self.pdbmax.text())
        stat_min = int(self.statnumin.text())
        query = (
            f"(datetime >= '{tm_min}') & (datetime <= '{tm_max}') & "
            f"(lon >= {lon_min}) & (lon <= {lon_max}) & "
            f"(lat >= {lat_min}) & (lat <= {lat_max}) & "
            f"(alt >= {alt_min}) & (alt <= {alt_max}) & "
            f"(chi >= {chi_min}) & (chi <= {chi_max}) & "
            f"(pdb >= {pdb_min}) & (pdb <= {pdb_max}) &"
            f"(number_stations >= {stat_min})"
        )
        self.lyl = self.og.query(query) 
        self.do_plot()
        
    def do_plot(self):
        self.update_status('Drawing images...')
        print('⏳ Drawing images.')
        self.do_clear()

        fig = QuickImage(self.lyl, self.cvar[self.cvar_dropdown.currentIndex()], self.cmap[self.cmap_dropdown.currentIndex()], self.map[self.map_dropdown.currentIndex()], [int(self.roads.isChecked()),int(self.rivers.isChecked()), int(self.rails.isChecked()),int(self.urban.isChecked())])
        canvas = FigureCanvasQTAgg(fig)
        self.view_layout.addWidget(canvas)
        print('✅ Images drawn.')
            
if __name__ == '__main__': 
    app = QApplication(sys.argv)
    window = HLMA()
    window.show()
    sys.exit(app.exec())
