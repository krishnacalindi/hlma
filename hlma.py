import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget,  QStatusBar, QLabel, QSplitter, QComboBox, QCheckBox, QLineEdit, QFrame
from PyQt6.QtGui import QIcon, QAction, QDoubleValidator, QRegularExpressionValidator, QIntValidator
from PyQt6.QtCore import Qt, QRegularExpression
import webbrowser
import warnings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import json
warnings.filterwarnings('ignore')

from bts import OpenLylout, QuickImage, BlankPlot

class HLMA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HLMA')
        self.setWindowIcon(QIcon('assets/icons/hlma.svg'))
        
        # data holders
        
        # state variable
        self.state = {
            'all_lylouts': None, 
            'plot_lylouts': None,
            'lma_stations': None
            }

        # other stuff
        self.cmap = json.load(open('assets/vars/cmap.json'))
        self.cvar = json.load(open('assets/vars/cvar.json'))
        self.map = json.load(open('assets/vars/map.json'))
        
        layout = QHBoxLayout()
        splitter = QSplitter()
        
        option_layout = QVBoxLayout()
        option_widget = QWidget()
        option_widget.setLayout(option_layout)

        self.view_layout = QVBoxLayout()
        view_widget = QWidget()
        view_widget.setLayout(self.view_layout)

        splitter.addWidget(option_widget)
        splitter.addWidget(view_widget)
        splitter.setSizes([1, 1])
    
        layout.addWidget(splitter)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        menubar = self.menuBar()

        file_menu = menubar.addMenu('File')
        state_menu = menubar.addMenu('State')
        help_menu = menubar.addMenu('Help')
        
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
        
        save_action = QAction('Save', self)
        save_action.setIcon(QIcon('assets/icons/down.svg'))
        save_action.triggered.connect(self.save_state)
        load_action = QAction('Load', self)
        load_action.setIcon(QIcon('assets/icons/up.svg'))
        load_action.triggered.connect(self.load_state)
        state_menu.addAction(save_action)
        state_menu.addAction(load_action)
        
        cvar_label = QLabel('Color by:')
        self.cvar_dropdown = QComboBox()
        self.cvar_dropdown.addItems(['Time', 'Longitude', 'Latitude', 'Altitude', 'Chi', 'Receiving power'])

        cmap_label = QLabel('Color map:')
        self.cmap_dropdown = QComboBox()
        for cmap_name in self.cmap:
            icon = QIcon(f'assets/colors/{cmap_name}.svg')
            self.cmap_dropdown.addItem(icon, cmap_name)
            
        map_label = QLabel('Map:')
        self.map_dropdown = QComboBox()
        self.map_dropdown.addItems(['State', 'County', 'NOAA County Warning Areas', '116 Congressional Districts'])
        
        features_layout = QHBoxLayout()
        features_label = QLabel('Features:')
        self.roads = QCheckBox('Roads')
        self.rivers = QCheckBox('Rivers')
        self.rails = QCheckBox('Rails')
        self.urban = QCheckBox('Urban area')
        self.roads.stateChanged.connect(self.do_filter)
        self.rivers.stateChanged.connect(self.do_filter)
        self.rails.stateChanged.connect(self.do_filter)
        self.urban.stateChanged.connect(self.do_filter)
        features_layout.addWidget(self.roads)
        features_layout.addWidget(self.rivers)
        features_layout.addWidget(self.rails)
        features_layout.addWidget(self.urban)
        
        self.cvar_dropdown.currentIndexChanged.connect(self.do_filter)
        self.cmap_dropdown.currentIndexChanged.connect(self.do_filter)
        self.map_dropdown.currentIndexChanged.connect(self.do_filter)
        
        time_layout = QHBoxLayout()
        time_regex = QRegularExpression(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
        time_validator = QRegularExpressionValidator(time_regex)
        timemin_label = QLabel('Minimum time:')
        self.timemin = QLineEdit()
        self.timemin.setText('yyyy-mm-dd hh:mm:ss')
        self.timemin.setValidator(time_validator)
        timemax_label = QLabel('Minimum time:')
        self.timemax = QLineEdit()
        self.timemax.setText('yyyy-mm-dd hh:mm:ss')
        self.timemax.setValidator(time_validator)
        time_layout.addWidget(timemin_label, 2)
        time_layout.addWidget(self.timemin, 1)
        time_layout.addWidget(timemax_label, 2)
        time_layout.addWidget(self.timemax, 1)
        
        lon_layout = QHBoxLayout()
        lonmin_label = QLabel('Minimum longitude:')
        self.lonmin = QLineEdit()
        self.lonmin.setText('-98.5')
        self.lonmin.setValidator(QDoubleValidator())
        lonmax_label = QLabel('Maximum longitude:')
        self.lonmax = QLineEdit()
        self.lonmax.setText('-91.5')
        self.lonmax.setValidator(QDoubleValidator())
        lon_layout.addWidget(lonmin_label, 2)
        lon_layout.addWidget(self.lonmin, 1)
        lon_layout.addWidget(lonmax_label, 2)
        lon_layout.addWidget(self.lonmax, 1)

        lat_layout = QHBoxLayout()
        latmin_label = QLabel('Minimum latitude:')
        self.latmin = QLineEdit()
        self.latmin.setText('26.0')
        self.latmin.setValidator(QDoubleValidator())
        latmax_label = QLabel('Maximum latitude:')
        self.latmax = QLineEdit()
        self.latmax.setText('33.0')
        self.latmax.setValidator(QDoubleValidator())
        lat_layout.addWidget(latmin_label, 2)
        lat_layout.addWidget(self.latmin, 1)
        lat_layout.addWidget(latmax_label, 2)
        lat_layout.addWidget(self.latmax, 1)

        alt_layout = QHBoxLayout()
        altmin_label = QLabel('Minimum altitude:')
        self.altmin = QLineEdit()
        self.altmin.setText('0.0')
        self.altmin.setValidator(QDoubleValidator())
        altmax_label = QLabel('Maximum altitude:')
        self.altmax = QLineEdit()
        self.altmax.setText('20.0')
        self.altmax.setValidator(QDoubleValidator())
        alt_layout.addWidget(altmin_label, 2)
        alt_layout.addWidget(self.altmin, 1)
        alt_layout.addWidget(altmax_label, 2)
        alt_layout.addWidget(self.altmax, 1)

        chi_layout = QHBoxLayout()
        chimin_label = QLabel('Minimum chi:')
        self.chimin = QLineEdit()
        self.chimin.setText('0.0')
        self.chimin.setValidator(QDoubleValidator())
        chimax_label = QLabel('Maximum chi:')
        self.chimax = QLineEdit()
        self.chimax.setText('2.0')
        self.chimax.setValidator(QDoubleValidator())
        chi_layout.addWidget(chimin_label, 2)
        chi_layout.addWidget(self.chimin, 1)
        chi_layout.addWidget(chimax_label, 2)
        chi_layout.addWidget(self.chimax, 1)

        pdb_layout = QHBoxLayout()
        pdbmin_label = QLabel('Minimum receiving power:')
        self.pdbmin = QLineEdit()
        self.pdbmin.setText('-60.0')
        self.pdbmin.setValidator(QDoubleValidator())
        pdbmax_label = QLabel('Maximum receiving power:')
        self.pdbmax = QLineEdit()
        self.pdbmax.setText('60.0')
        self.pdbmax.setValidator(QDoubleValidator())
        pdb_layout.addWidget(pdbmin_label, 2)
        pdb_layout.addWidget(self.pdbmin, 1)
        pdb_layout.addWidget(pdbmax_label, 2)
        pdb_layout.addWidget(self.pdbmax, 1)
        
        statnum_layout = QHBoxLayout()
        statnummin_label = QLabel('Minimum number of stations:')
        self.statnumin = QLineEdit()
        self.statnumin.setText('6')
        self.statnumin.setValidator(QIntValidator())
        statnum_layout.addWidget(statnummin_label, 2)
        statnum_layout.addWidget(self.statnumin, 1)
        statnum_layout.addStretch(3)
                
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
        
        option_layout.addWidget(QLabel('<h1>Filter options</h1>'))             
        option_layout.addLayout(time_layout)
        option_layout.addStretch(1)
        option_layout.addLayout(lon_layout)
        option_layout.addStretch(1)
        option_layout.addLayout(lat_layout)
        option_layout.addStretch(1)
        option_layout.addLayout(alt_layout)
        option_layout.addStretch(1)
        option_layout.addLayout(chi_layout)
        option_layout.addStretch(1)
        option_layout.addLayout(pdb_layout)
        option_layout.addStretch(1)
        option_layout.addLayout(statnum_layout)
        option_layout.addStretch(2)
        
        option_layout.addWidget(QLabel('<h1>Map options</h1>'))    
        option_layout.addWidget(map_label)
        option_layout.addWidget(self.map_dropdown)
        option_layout.addStretch(1)
        option_layout.addWidget(features_label)
        option_layout.addLayout(features_layout)
        option_layout.addStretch(2)
        
        option_layout.addWidget(QLabel('<h1>Color options</h1>'))  
        option_layout.addWidget(cvar_label)
        option_layout.addWidget(self.cvar_dropdown)
        option_layout.addStretch(1)
        option_layout.addWidget(cmap_label)
        option_layout.addWidget(self.cmap_dropdown)
        option_layout.addStretch(2)
        
        option_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.do_blank()
        view_widget.setFocus()
        self.showMaximized()
    
    def save_state(self):
        import pickle
        with open('state.pkl', 'wb') as file:
            pickle.dump(self.state, file)
        print('✅ Saved state in state.pkl.')
        
    def load_state(self):
        import pickle
        with open('state.pkl', 'rb') as file:
            self.state = pickle.load(file)
        self.do_plot()
        print('✅ Loaded state in state.pkl.')
    
    def do_open(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select LYLOUT files', '', 'Dat files (*.dat)')
        if files:
            self.state['all_lylout'], failed_files, self.state['lma_stations'] = OpenLylout(files)
            if self.state['all_lylout'] is None:
                print('❌ All LYLOUT files were not processed due to errors.')
            elif failed_files:
                print('❌ Following LYLOUT files were not processed due to errors:')
                for f in failed_files:
                    print(f)
            else:
                print('✅ All LYLOUT files were opened successfully.')
            self.timemin.setText(self.state['all_lylout']['datetime'].min().floor('s').strftime('%Y-%m-%d %H:%M:%S'))
            self.timemax.setText(self.state['all_lylout']['datetime'].max().ceil('s').strftime('%Y-%m-%d %H:%M:%S'))
            self.do_filter()

    def do_clear(self):
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
        if self.state['all_lylout'] is None:
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
        self.state['plot_lylout'] = self.state['all_lylout'].query(query) 
        self.do_plot()
        
    def do_plot(self):
        print('⏳ Drawing images.')
        self.do_clear()

        fig = QuickImage(self.state['plot_lylout'], self.cvar[self.cvar_dropdown.currentIndex()], self.cmap[self.cmap_dropdown.currentIndex()], self.map[self.map_dropdown.currentIndex()], [int(self.roads.isChecked()),int(self.rivers.isChecked()), int(self.rails.isChecked()),int(self.urban.isChecked())], self.state['lma_stations'])
        canvas = FigureCanvasQTAgg(fig)
        self.view_layout.addWidget(canvas)
        print('✅ Images drawn.')
            
if __name__ == '__main__': 
    app = QApplication(sys.argv)
    window = HLMA()
    window.show()
    sys.exit(app.exec())
