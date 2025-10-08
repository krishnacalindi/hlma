# global imports
import sys
import warnings
import os
import webbrowser
from pathlib import Path

# pyqt
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QDialog, QPushButton, QDialogButtonBox
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QSettings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# special imports
from matplotlib.dates import num2date
from pandas import date_range
from datetime import datetime
import pickle

# manual functions
from bts import OpenLylout, QuickImage, BlankPlot
from setup import UI, Connections, Folders, Utility, State

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
        
        # setting up
        # folders
        Folders()
        # utiilty
        self.util = Utility()
        # ui
        self.ui = UI(self)
        # connections
        Connections(self, self.ui)
        # state
        self.state = State()
        self.state.replot = self.plot # connecting replot function
        
        # # Used by polygonning tools
        # self.clicks = [] 
        # self.lines = []
        # self.remove = False
        # self.dots = []
        # self.polygon = polygon
        # self.undo_filter = undo_filter
        # self.redo_filter = redo_filter
        # self.apply_filters = apply_filters
        # self.zoom_to_polygon = zoom_to_polygon
        # self.prev_ax = None
        
        # go!
        self.do_blank()
        self.ui.view_widget.setFocus()
        self.showMaximized()
    
    def import_lylout(self):
        # resetting default zoom (#FIXME: unzoom option?)
        self.state.plot_options.lon_min = -98
        self.state.plot_options.lon_max = -92
        self.state.plot_options.lat_min = 27
        self.state.plot_options.lat_max = 33
        files, _ = QFileDialog.getOpenFileNames(self, 'Select LYLOUT files', self.settings.value('lylout_folder', ''), 'Dat files (*.dat)')
        if files:
            dialog = LoadingDialog('Opening selected LYLOUT files...')
            dialog.show()
            QApplication.processEvents()
            self.settings.setValue('lylout_folder', os.path.dirname(files[0]))
            self.state.all, failed_files, self.state.stations = OpenLylout(files)
            if self.state.all is None:
                print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ All LYLOUT files were not processed due to errors.')
            elif failed_files:
                print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ Following LYLOUT files were not processed due to errors:')
                for f in failed_files:
                    print(f)
            else:
                print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ All LYLOUT files were opened successfully.')
            dialog.close()
            self.ui.timemin.setText(self.state.all['datetime'].min().floor('s').strftime('%Y-%m-%d %H:%M:%S'))
            self.ui.timemax.setText(self.state.all['datetime'].max().ceil('s').strftime('%Y-%m-%d %H:%M:%S'))
            self.filter()
    
    def import_state(self):
        try:
            with open('state/state.pkl', 'rb') as file:
                save_state = pickle.load(file)
                self.state.update(all = save_state['all'], stations = self.state.stations, plot = save_state['plot'], plot_options = save_state['plot_options'])
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ Loaded state in state.pkl.')
        except:
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ An unexpected error occured while loading state.')
        
    def export_dat(self):
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
    
    def export_parquet(self):
        try:
            if not self.state.all.empty:
                self.state.all[self.state.plot].to_parquet('output/lylout.parquet', index=False)
                print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ Saved paruqet in output/lylout.parquet.')
        except:
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ An unexpected error occurred while saving as parquet.')
    
    def export_state(self):
        try:
            with open('state/state.pkl', 'wb') as file:
                save_state = {'all': self.state.all, 'stations': self.state.stations, 'plot': self.state.plot, 'plot_options': self.state.plot_options, }
                pickle.dump(save_state, file)
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ Saved state in state.pkl.')
        except Exception as e:
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} {e} ❌ An unexpected error occured while saving state.')
    
    def export_image(self):
        try:
            self.state.canvas.print_figure('output/image.pdf', dpi=500, bbox_inches='tight')
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ✅ Saved image in output/image.pdf.')
        except:
            print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ❌ An unexpected error occurred while saving image.')         

    def options_clear(self):
        for i in reversed(range(self.ui.view.count())):
            item = self.ui.view.itemAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def flash_dtd(self):
        return
    
    def flash_mccaul(self):
        return
    
    def help_about(self):
        webbrowser.open('https://lightning.tamu.edu/hlma/')
    
    def help_contact(self):
        webbrowser.open('https://artsci.tamu.edu/atmos-science/contact/profiles/timothy-logan.html')
    
    def help_color(self):
        webbrowser.open('https://colorcet.holoviz.org/user_guide/Continuous.html#linear-sequential-colormaps-for-plotting-magnitudes')
    
    def do_blank(self):
        fig = BlankPlot()
        self.state.canvas = FigureCanvasQTAgg(fig)
        self.ui.view.addWidget(self.state.canvas)

    def on_click(self, event):
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

                    self.state['canvas'].draw()
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
                        self.state['canvas'].draw()
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
                            if len(self.clicks) <= 4:
                                self.min_x, self.max_x, self.min_y, self.max_y = self.zoom_to_polygon(self)
                            self.do_paint()
                        elif pd.get_choice() == 4:
                            self.do_paint()

                        self.prev_ax = None
                        # Clearing drawn points here
                        self.clicks.clear()
                        for line in self.lines:
                            self.lines.remove(line)
                        for dot in self.dots:
                            self.dots.remove(dot)
                        
                        self.lines.clear()
                        self.dots.clear()              
                        self.state['canvas'].draw()            

            self.state['canvas'].draw()

    def filter(self):
        print(f'{datetime.now().strftime("%b %d %H:%M:%S")} ⏳ Filtering data.')
        if self.state.all is None:
            return
        query = (
            f"(datetime >= '{self.ui.timemin.text()}') & (datetime <= '{self.ui.timemax.text()}') & "
            f"(lon >= {self.ui.lonmin.text()}) & (lon <= {self.ui.lonmax.text()}) & "
            f"(lat >= {self.ui.latmin.text()}) & (lat <= {self.ui.latmax.text()}) & "
            f"(alt >= {float(self.ui.altmin.text()) * 1000}) & (alt <= {float(self.ui.altmax.text()) * 1000}) & "
            f"(chi >= {self.ui.chimin.text()}) & (chi <= {self.ui.chimax.text()}) & "
            f"(pdb >= {self.ui.powermin.text()}) & (pdb <= {self.ui.powermax.text()}) & "
            f"(number_stations >= {self.ui.stationsmin.text()})"
        )
        self.state.update(plot=self.state.all.eval(query))
    
    def plot(self):
        print(f'{datetime.now()} Starting to print')
        self.options_clear()
        dialog = LoadingDialog('Rendering images...')
        dialog.show()
        QApplication.processEvents()
        fig = QuickImage(self.state)
        self.state.canvas = FigureCanvasQTAgg(fig)
        dialog.close()
        self.ui.view.addWidget(self.state.canvas)

if __name__ == "__main__": 
    app = QApplication(sys.argv)
    warnings.filterwarnings('ignore')
    window = HLMA()
    window.show()
    sys.exit(app.exec())
