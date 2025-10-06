import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget,  QLabel, QSplitter, QComboBox, QCheckBox, QLineEdit, QDialog, QPushButton, QDialogButtonBox
from PyQt6.QtGui import QIcon, QAction, QDoubleValidator, QRegularExpressionValidator, QIntValidator
from PyQt6.QtCore import Qt, QRegularExpression, QSettings
from matplotlib.dates import num2date
import webbrowser
import warnings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import json
warnings.filterwarnings('ignore')
from pandas import date_range
from datetime import datetime, timedelta
from polygon import polygon, undo_filter, redo_filter, apply_filters
from bts import OpenLylout, QuickImage, BlankPlot

class PolygonDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polygon")
        self.setWindowIcon(QIcon('assets/icons/keep.svg'))
        self.setModal(True)
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

class LoadingDialog(QDialog):
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle('Please wait...')
        self.setModal(True) 
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(200, 100)
        layout = QVBoxLayout()
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)
    

class HLMA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HLMA')
        self.setWindowIcon(QIcon('assets/icons/hlma.svg'))
        self.settings = QSettings('HLMA', 'LAt')
        
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
        self.clicks = []
        self.lines = []
        self.remove = False
        self.dots = []
        self.polygon = polygon
        self.undo_filter = undo_filter
        self.redo_filter = redo_filter
        self.apply_filters = apply_filters
        
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

        import_menu = menubar.addMenu('Import')
        export_menu = menubar.addMenu('Export')
        options_menu = menubar.addMenu('Options')
        help_menu = menubar.addMenu('Help')
        
        import_lylout_action = QAction('LYLOUT', self)
        import_lylout_action.triggered.connect(self.do_open)
        import_menu.addAction(import_lylout_action)
        
        import_entln_action = QAction('ENTLN', self)
        # import_lylout_action.triggered.connect(self.do_open)
        import_menu.addAction(import_entln_action)
        
        import_state_action = QAction('State', self)
        import_state_action.triggered.connect(self.load_state)
        import_menu.addAction(import_state_action)
        
        export_dat_action = QAction('DAT', self)
        export_dat_action.triggered.connect(self.do_dat)
        export_menu.addAction(export_dat_action)
        
        export_parquet_action = QAction('Parquet', self)
        export_parquet_action.triggered.connect(self.save_parquet)
        export_menu.addAction(export_parquet_action)
        
        export_state_action = QAction('State', self)
        export_state_action.triggered.connect(self.save_state)
        export_menu.addAction(export_state_action)
    
        clear_action = QAction('Clear', self)
        clear_action.setIcon(QIcon('assets/icons/clear.svg'))
        clear_action.triggered.connect(self.do_clear)
        options_menu.addAction(clear_action)
        
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
        timemax_label = QLabel("Maximum time:")
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
        try:
            with open('state/state.pkl', 'wb') as file:
                pickle.dump(self.state, file)
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ Saved state in state.pkl.')
        except:
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ An unexpected error occured while saving state.')
        
    def load_state(self):
        import pickle
        try:
            with open('state/state.pkl', 'rb') as file:
                self.state = pickle.load(file)
            self.do_plot()
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ Loaded state in state.pkl.')
        except:
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ An unexpected error occured while loading state.')
    
    def save_parquet(self):
        try:
            if not self.state['plot_lylouts'].empty:
                self.state['plot_lylouts'].to_parquet('output/lylout.parquet', index=False)
                print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ Saved paruqet in output/lylout.parquet.')
        except:
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ An unexpected error occurred while saving as parquet.')
            
    
    def do_open(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select LYLOUT files', self.settings.value('lylout_folder', ''), 'Dat files (*.dat)')
        if files:
            dialog = LoadingDialog('Opening selected LYLOUT files...')
            dialog.show()
            QApplication.processEvents()
            self.settings.setValue('lylout_folder', os.path.dirname(files[0]))
            self.state['all_lylouts'], failed_files, self.state['lma_stations'] = OpenLylout(files)
            if self.state['all_lylouts'] is None:
                print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ All LYLOUT files were not processed due to errors.')
            elif failed_files:
                print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ Following LYLOUT files were not processed due to errors:')
                for f in failed_files:
                    print(f)
            else:
                print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ All LYLOUT files were opened successfully.')
            dialog.close()
            self.timemin.setText(self.state['all_lylouts']['datetime'].min().floor('s').strftime('%Y-%m-%d %H:%M:%S'))
            self.timemax.setText(self.state['all_lylouts']['datetime'].max().ceil('s').strftime('%Y-%m-%d %H:%M:%S'))
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
        print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ⏳ Filtering data.')
        if self.state['all_lylouts'] is None:
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
        self.state['plot_lylouts'] = self.state['all_lylouts'].query(query) 
        self.do_plot()
        
    def do_plot(self):
        print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ⏳ Drawing images.')
        self.do_clear()

        dialog = LoadingDialog('Rendering images...')
        dialog.show()
        QApplication.processEvents()
        fig = QuickImage(self.state['plot_lylouts'], self.cvar[self.cvar_dropdown.currentIndex()], self.cmap[self.cmap_dropdown.currentIndex()], self.map[self.map_dropdown.currentIndex()], [int(self.roads.isChecked()),int(self.rivers.isChecked()), int(self.rails.isChecked()),int(self.urban.isChecked())], self.state['lma_stations'])
        dialog.close()
        
        canvas = FigureCanvasQTAgg(fig)
        self.view_layout.addWidget(canvas)
        self.prev_ax = None
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
                        # pd.get_choice() will return the 1-4 for the thingy (1:keep,2:remove,3:zoom,4:cancel)
                        if pd.get_choice() == 1: # Keep
                            self.remove = False
                            self.polygon(self, self.prev_ax)
                        elif pd.get_choice() == 2: # Remove
                            self.remove = True
                            self.polygon(self, self.prev_ax)
                        elif pd.get_choice() == 3: # Zoom
                            self.remove = False
                            self.polygon(self, self.prev_ax)
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
        
        canvas.mpl_connect('button_press_event', on_click)
        canvas.draw()
        print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ Images drawn.')
    
    def do_dat(self):
        from pathlib import Path
        start = self.state['plot_lylouts']['datetime'].min().floor('10min')
        end = self.state['plot_lylouts']['datetime'].max().ceil('10min')

        bins = date_range(start, end, freq='10min')

        for i, chunk in enumerate(bins):
            if i == len(bins) - 1:
                continue

            filename = f"LYLOUT_{datetime.strftime(chunk, '%y%m%d_%H%M%S')}_0600"
            start = bins[i]
            end = bins[i+1]
            df_chunk = self.state['plot_lylouts'][(self.state['plot_lylouts']['datetime'] >= start) & (self.state['plot_lylouts']['datetime'] < end)]
            df_chunk = df_chunk[['utc_sec', 'lat', 'lon', 'alt', 'chi', 'number_stations', 'pdb', 'mask']]
            beginning_stuff = f"""Houston A&M Lightning Mapping System -- Selected Data
                                When exported: {datetime.now().ctime()}
                                Original data file: {Path.home()}
                                Data start time: {start}
                                Location: LYLOUT
                                Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, # of stations contributed, P(dBW), mask
                                Data format: f15.9 f11.6 f11.6 f8.1 f6.2 2i e11.4 4x
                                Number of events:       {len(df_chunk)}
                                Flash stats: not saved
                                ***data***\n"""

            with open(f'./output/{filename}.dat', 'w', newline='') as file:
                file.write(beginning_stuff)
                for _, row in df_chunk.iterrows():
                    line = (
                        f"{row.utc_sec:15.9f} "
                        f"{row.lat:11.6f} "
                        f"{row.lon:11.6f} "
                        f"{row.alt:8.1f} "
                        f"{row.chi:6.2f} "
                        f"{int(row.number_stations):2d} "
                        f"{row.pdb:11.4e} "
                        f"{int(row['mask'], 16):04x}\n"
                    )
                    file.write(line)
            
        print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ Exporting complete.')

if __name__ == "__main__": 
    app = QApplication(sys.argv)
    import os
    os.makedirs('state', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    window = HLMA()
    window.show()
    sys.exit(app.exec())
