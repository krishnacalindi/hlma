# global imports
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import webbrowser
from pathlib import Path
import logging
import time

# pyqt
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QDialog, QPushButton, QDialogButtonBox
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QSettings

# special imports
from matplotlib import colors as mcolors
import colorcet as cc
from pandas import date_range
import numpy as np
from datetime import datetime
import pickle
from deprecated import deprecated

# manual functions
from bts import OpenLylout, OpenEntln, DotToDot, McCaul
from setup import UI, Connections, Folders, Utility, State, LoadingDialog
from polygon import PolygonFilter

# logging
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("hlma.py")

class HLMA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('HLMA', 'LAt')
        
        # setting up
        # state
        self.state = State()
        self.state.__dict__['replot'] = self.visplot
        
        # folders
        Folders()
        # utiilty
        self.util = Utility()
        # ui
        self.ui = UI(self)

        # Polygonning tool
        self.polyfilter = PolygonFilter(self)
        
        # connections
        Connections(self, self.ui)
        
        # undo and redo
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self.undo)
        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self.redo)

        # go!
        self.ui.view_widget.setFocus()
        logger.info("Application running.")
        self.showMaximized()

    def import_lylout(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select LYLOUT files', self.settings.value('lylout_folder', ''), 'Dat files (*.dat *.dat.gz)')
        if files:
            dialog = LoadingDialog('Opening selected LYLOUT files...')
            dialog.show()
            QApplication.processEvents()
            self.settings.setValue('lylout_folder', os.path.dirname(files[0]))
            # following syntax ensures one call to set_attr to appropriately track history
            self.state.__dict__['all'], self.state.__dict__['stations'] = OpenLylout(files)
            logger.info("All LYLOUT files opened.")
            self.ui.timemin.setText(self.state.all['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'))
            self.ui.timemax.setText(self.state.all['datetime'].max().strftime('%Y-%m-%d %H:%M:%S'))
            self.filter()
            dialog.close()

    def import_entln(self):
        if not len(self.state.all) > 0:
            logger.warning(f"No LYLOUT data, cannot plot ENTLN")
            return 
        
        files, _ = QFileDialog.getOpenFileNames(self, 'Select ENTLN files', self.settings.value('entln_folder', ''), 'CSV files (*.csv)')
        if files:
            dialog = LoadingDialog('Opening selected ENTLN files...')
            dialog.show()
            QApplication.processEvents()
            self.settings.setValue('entln_folder', os.path.dirname(files[0]))
            # See import_lylout for syntactic reasoning
            temp = OpenEntln(files, self.state.all['datetime'].min())
            self.state.gsd = temp
            colors = [(1.0, 0.0, 0.0, 1.0) if pc >= 0 else (0.0, 0.0, 1.0, 1.0) for pc in temp['peakcurrent'].to_numpy()]
            self.state.__dict__['gsd']['colors'] = colors
            logger.info("All ENTLN files opened.")
            dialog.close()

            if not self.state.gsd.empty:
                self.ui.v0.add(self.ui.gs0)
                self.ui.v1.add(self.ui.gs1)
                self.ui.v3.add(self.ui.gs3)
                self.ui.v4.add(self.ui.gs4)
                sym = 'triangle_up'

                positions = np.column_stack([temp['utc_sec'].to_numpy(dtype=np.float32),temp['alt'].to_numpy(dtype=np.float32)])
                self.ui.gs0.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

                positions = temp[['lon', 'alt']].to_numpy().astype(np.float32)
                self.ui.gs1.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

                positions = temp[['lon', 'lat']].to_numpy().astype(np.float32)
                self.ui.gs3.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

                positions = temp[['alt', 'lat']].to_numpy().astype(np.float32)
                self.ui.gs4.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

                # Induce change for state saving
                self.state.__dict__['gsd_mask'] = np.ones([temp.shape[0]], dtype=bool)
    
    def import_state(self):
        try:
            with open('state/state.pkl', 'rb') as file:
                save_state = pickle.load(file)
                self.state.update(all = save_state['all'], stations = self.state.stations, plot = save_state['plot'], plot_options = save_state['plot_options'])
            logger.info("Loaded state in state/state.pkl.")
        except Exception as e:
            logger.warning(f"Could not load state/state.pkl due to {e}.")
        
    def export_dat(self):
        temp = self.state.all[self.state.plot]
        start = temp['datetime'].min().floor('10min')
        end = temp['datetime'].max().ceil('10min')
        bins = date_range(start, end, freq='10min', inclusive='left')
        for i, chunk in enumerate(bins):
            filename = f"LYLOUT_{datetime.strftime(chunk, '%y%m%d_%H%M%S')}_0600"
            start = bins[i]
            end = bins[i+1]
            df_chunk = temp[(temp['datetime'] >= start) & (temp['datetime'] < end)]
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
        logger.info("Saved files in output.")
    
    def export_parquet(self):
        try:
            if not self.state.all.empty:
                self.state.all[self.state.plot].to_parquet('output/lylout.parquet', index=False)
                logger.info("Saved file in output/lylout.parquet.")
        except Exception as e:
            logger.warning(f"Could not save file in output/lylout.parquet due to {e}.")
    
    def export_state(self):
        try:
            with open('state/state.pkl', 'wb') as file:
                save_state = {'all': self.state.all, 'stations': self.state.stations, 'plot': self.state.plot, 'plot_options': self.state.plot_options, }
                pickle.dump(save_state, file)
            logger.info("Saved state in state/state.pkl.")
        except Exception as e:
            logger.warning(f"Could not save state in state/state.pkl due to {e}")
    
    def export_image(self):
        try:
            pixmap = self.ui.view_widget.grab()
            pixmap.save("output/image.pdf")
            logger.info("Saved image in output/image.pdf")
        except Exception as e:
            logger.warning(f"Could not save image in output/image.pdf due to {e}")     

    def options_clear(self):
        self.ui.s0.set_data(np.empty((0, 2)))
        self.ui.s1.set_data(np.empty((0, 2)))
        self.ui.s3.set_data(np.empty((0, 2)))
        self.ui.s4.set_data(np.empty((0, 2)))
        self.ui.hist.set_data(np.empty((0, 2))) 
    
    def options_reset(self):
        self.ui.v0.camera.reset()
        self.ui.v1.camera.reset()
        self.ui.v2.camera.reset()
        self.ui.v3.camera.reset()
        self.ui.v4.camera.reset()
    
    def flash_dtd(self):
        dialog = LoadingDialog('Running dot to dot flash algorithm..')
        dialog.show()
        QApplication.processEvents()
        DotToDot(self.state)
        dialog.close()
    
    def flash_mccaul(self):
        dialog = LoadingDialog('Running McCaul flash algorithm..')
        dialog.show()
        QApplication.processEvents()
        McCaul(self.state)
        dialog.close()
    
    def help_about(self):
        webbrowser.open('https://lightning.tamu.edu/hlma/')
    
    def help_contact(self):
        webbrowser.open('https://artsci.tamu.edu/atmos-science/contact/profiles/timothy-logan.html')
    
    def help_color(self):
        webbrowser.open('https://colorcet.holoviz.org/user_guide/Continuous.html#linear-sequential-colormaps-for-plotting-magnitudes')
        
    def filter(self):
        if self.state.all is None:
            return
        query = (
            f"(datetime >= '{self.ui.timemin.text()}') & (datetime <= '{self.ui.timemax.text()}') & "
            f"(alt >= {float(self.ui.altmin.text()) * 1000}) & (alt <= {float(self.ui.altmax.text()) * 1000}) & "
            f"(chi >= {self.ui.chimin.text()}) & (chi <= {self.ui.chimax.text()}) & "
            f"(pdb >= {self.ui.powermin.text()}) & (pdb <= {self.ui.powermax.text()}) & "
            f"(number_stations >= {self.ui.stationsmin.text()})"
        )
        self.state.__dict__['plot'] = self.state.all.eval(query)
        self.polyfilter.inc_mask = np.zeros(np.count_nonzero(self.state.plot), dtype=bool)
        self.state.replot()
    
    def animate(self):
        logger.info("Starting animation.")
        temp = self.state.all[self.state.plot]
        temp.alt /= 1000
        cvar = self.state.plot_options.cvar
        cmap = self.state.plot_options.cmap
        arr = temp[cvar].to_numpy()
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        colors = cmap(norm)
        # NOTE: hardcooding 30 chunks
        breaks = [i * (len(temp) // 30 )for i in range(1, 31)]
        
        for idx, bp in enumerate(breaks):
            self.anim_plot(temp[:bp], colors[:bp])
            QApplication.processEvents()
            time.sleep(0.1) 
        
        logger.info("Finished animation.")
    
    def anim_plot(self, temp, colors):
        positions = np.column_stack([temp['seconds'].to_numpy(dtype=np.float32),temp['alt'].to_numpy(dtype=np.float32)])
        self.ui.s0.set_data(pos=positions, face_color=colors, size=1, edge_width=0, edge_color='green')
        
        positions = temp[['lon', 'alt']].to_numpy().astype(np.float32)
        self.ui.s1.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        
        bins = 200
        counts, edges = np.histogram(temp['alt'], bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        line_data = np.column_stack([counts, centers])
        self.ui.hist.set_data(pos=line_data, color=(1, 1, 1, 1), width=1)
        
        positions = temp[['lon', 'lat']].to_numpy().astype(np.float32)
        self.ui.s3.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        
        positions = temp[['alt', 'lat']].to_numpy().astype(np.float32)
        self.ui.s4.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        
    def visplot(self):
        logger.info("Starting vis.py plotting.")
        temp = self.state.all[self.state.plot]
        temp.alt /= 1000
        cvar = self.state.plot_options.cvar
        cmap = self.state.plot_options.cmap
        arr = temp[cvar].to_numpy()
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        colors = cmap(norm)

        positions = np.column_stack([temp['seconds'].to_numpy(dtype=np.float32),temp['alt'].to_numpy(dtype=np.float32)])
        self.ui.s0.set_data(pos=positions, face_color=colors, size=1, edge_width=0, edge_color='green')
        self.ui.v0.camera.set_range(x=(positions[:,0].min(), positions[:,0].max()), y=(0, 20))
        self.ui.v0.camera.set_default_state()

        positions = temp[['lon', 'alt']].to_numpy().astype(np.float32)
        self.ui.s1.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        self.ui.v1.camera.set_range(x=(positions[:,0].min(), positions[:,0].max()), y=(0, positions[:,1].max()))
        self.ui.v1.camera.set_default_state()
        
        bins = 200
        counts, edges = np.histogram(temp['alt'], bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        line_data = np.column_stack([counts, centers])
        self.ui.hist.set_data(pos=line_data, color=(1, 1, 1, 1), width=1)
        self.ui.v2.camera.set_range(x=(0, counts.max()), y=(0, 20))
        self.ui.v2.camera.set_default_state()

        positions_list = []
        colors_list = []
        minx, maxx = temp.lon.min(), temp.lon.max()
        miny, maxy = temp.lat.min(), temp.lat.max()
        map_gdf = self.state.plot_options.map.cx[minx:maxx, miny:maxy]
        if not map_gdf.empty:
            pos_map = np.vstack(map_gdf.geometry.boundary.explode(index_parts=False).apply(lambda p: np.append(np.array(p.coords, np.float32),np.array([[np.nan, np.nan]]),axis=0)).values)
            color_map = np.tile(np.array([1, 1, 1, 1.0], dtype=np.float32), (pos_map.shape[0], 1))
            positions_list.append(pos_map)
            colors_list.append(color_map)
        for fdict in self.state.plot_options.features.values():
            gdf = fdict['gdf'].cx[minx:maxx, miny:maxy]
            if gdf.empty:
                continue
            pos_array = np.vstack(gdf.geometry.explode(index_parts=False).apply(lambda p: np.append(np.array(p.coords, np.float32),np.array([[np.nan, np.nan]]), axis=0)).values)
            colors_array = np.tile(np.array(mcolors.to_rgba(fdict['color']), dtype=np.float32), (pos_array.shape[0], 1))
            positions_list.append(pos_array)
            colors_list.append(colors_array)
        if positions_list:
            map_positions = np.vstack(positions_list)
            map_colors = np.vstack(colors_list)
        else:
            map_positions = np.empty((0, 2), dtype=np.float32)
            map_colors = np.empty((0, 4), dtype=np.float32)
        self.ui.map.set_data(pos=map_positions, color=map_colors)
        positions = temp[['lon', 'lat']].to_numpy().astype(np.float32)
        self.ui.s3.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        self.ui.v3.camera.set_range(x=(positions[:,0].min(), positions[:,0].max()), y=(positions[:,1].min(), positions[:,1].max()))
        self.ui.v3.camera.set_default_state()
               
        positions = temp[['alt', 'lat']].to_numpy().astype(np.float32)
        self.ui.s4.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        self.ui.v4.camera.set_range(x=(0, positions[:,0].max()), y=(positions[:,1].min(), positions[:,1].max()))
        self.ui.v4.camera.set_default_state()

        if self.state.gsd.empty or len(self.state.gsd[self.state.gsd_mask]) == 0:
            # If our ground strike data is empty, or if the filtered df is empty we need to remove the visuals
            if self.ui.gs0 in self.ui.v0.scene.children:
                self.ui.v0.scene.children.remove(self.ui.gs0)
                self.ui.gs0.parent = None
            if self.ui.gs1 in self.ui.v1.scene.children:
                self.ui.v1.scene.children.remove(self.ui.gs1)
                self.ui.gs1.parent = None
            if self.ui.gs3 in self.ui.v3.scene.children:
                self.ui.v3.scene.children.remove(self.ui.gs3)
                self.ui.gs3.parent = None
            if self.ui.gs4 in self.ui.v4.scene.children:
                self.ui.v4.scene.children.remove(self.ui.gs4)
                self.ui.gs4.parent = None
        else:
            colors = np.stack(self.state.gsd[self.state.gsd_mask]['colors'].to_numpy())
            sym = 'triangle_up'

            if self.ui.gs0 not in self.ui.v0.scene.children:
                self.ui.v0.add(self.ui.gs0)
            if self.ui.gs1 not in self.ui.v1.scene.children:
                self.ui.v1.add(self.ui.gs1)
            if self.ui.gs3 not in self.ui.v3.scene.children:
                self.ui.v3.add(self.ui.gs3)
            if self.ui.gs4 not in self.ui.v4.scene.children:
                self.ui.v4.add(self.ui.gs4)

            positions = np.column_stack([self.state.gsd[self.state.gsd_mask]['utc_sec'].to_numpy(dtype=np.float32),self.state.gsd[self.state.gsd_mask]['alt'].to_numpy(dtype=np.float32)])
            self.ui.gs0.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

            positions = self.state.gsd[self.state.gsd_mask][['lon', 'alt']].to_numpy().astype(np.float32)
            self.ui.gs1.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

            positions = self.state.gsd[self.state.gsd_mask][['lon', 'lat']].to_numpy().astype(np.float32)
            self.ui.gs3.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

            positions = self.state.gsd[self.state.gsd_mask][['alt', 'lat']].to_numpy().astype(np.float32)
            self.ui.gs4.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

        logger.info("Finished vis.py plotting.")
    
    def undo(self):
        logger.info("Undo called")
        if len(self.polyfilter.clicks) > 0:
            self.polyfilter.clicks.pop()
            self.polyfilter.handle_poly_plot()
        elif self.state.history:
            self.state.future.append(self.state)
            self.state = self.state.history.pop()
            logger.info(f"Set state to {len(self.state.plot)}")
            self.state.replot()
    
    def redo(self):
        logger.info("Redo called")
        if self.state.future:
            self.state.history.append(self.state)
            self.state = self.state.future.pop()
            self.state.replot()

      
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = HLMA()
    window.setWindowTitle('Aggie XLMA')
    window.setWindowIcon(QIcon('assets/icons/hlma.svg'))
    window.show()
    sys.exit(app.exec())
