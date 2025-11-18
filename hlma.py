"""Main GUI window for the HLMA application.

Handles plotting, visualization, and analysis of lightning data,
including animations and interactive polygon-based selections.
Connects user actions to underlying data processing and the FLASH
algorithm, serving as the primary entry point for the application.

Attributes
----------
settings : QSettings
Stores application-specific settings.
state : State
Application state containing loaded data and plot configuration.
anim : Animate
Animation controller for temporal lightning visualizations.
ui : SimpleNamespace
Container for all UI widgets.
polyfilter : PolygonFilter
Tool for interactive polygon-based selection.

"""

import logging
import pickle
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import colors as mcolors
from pandas import date_range
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow

from bts import dot_to_dot, mc_caul, open_entln, open_lylout
from polygon import PolygonFilter
from setup import (
    Animate,
    LoadingDialog,
    State,
    connect_ui,
    setup_folders,
    setup_ui,
    setup_utility,
)

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("hlma.py")

class HLMA(QMainWindow):
    """Main GUI window for the HLMA application.

    Handles plotting, visualization, and analysis of lightning data,
    including animations and interactive polygon-based selections.
    Connects user actions to underlying data processing and the FLASH
    algorithm, serving as the primary entry point for the application.

    Attributes
    ----------
    settings : QSettings
        Stores application-specific settings.
    state : State
        Application state containing loaded data and plot configuration.
    anim : Animate
        Animation controller for temporal lightning visualizations.
    ui : SimpleNamespace
        Container for all UI widgets.
    polyfilter : PolygonFilter
        Tool for interactive polygon-based selection.

    """

    def __init__(self) -> None:
        """Initialize the HLMA main GUI application window.

        Sets up application folders, utility data, UI components, and
        connections. Initializes the animation controller, state object,
        and polygon selection tool. Also sets up undo/redo shortcuts.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        super().__init__()
        self.settings = QSettings("HLMA", "LAt")

        # setting up
        # state
        self.state = State()
        self.state.__dict__["replot"] = self.visplot
        self.anim = Animate()
        self.anim.timer.connect(self._animate_step)

        # folders
        setup_folders()
        # utiilty
        self.util = setup_utility()
        # ui
        self.ui = setup_ui(self)

        # Polygonning tool
        self.polyfilter = PolygonFilter(self)

        # connections
        connect_ui(self, self.ui)

        # undo and redo
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self.undo)
        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self.redo)

        # go!
        self.ui.view_widget.setFocus()
        logger.info("Application running.")
        self.showMaximized()

    def import_lylout(self) -> None:
        """Open and load one or more LYLOUT files selected by the user.

        Opens files using a file dialog, loads data into the state object,
        updates time range and station statistics, and applies current filters.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        files, _ = QFileDialog.getOpenFileNames(self, "Select LYLOUT files", self.settings.value("lylout_folder", ""), "Dat files (*.dat *.dat.gz)")
        if files:
            dialog = LoadingDialog("Opening selected LYLOUT files...")
            dialog.show()
            QApplication.processEvents()
            self.settings.setValue("lylout_folder", str(Path(files[0]).parent))
            # following syntax ensures one call to set_attr to appropriately track history
            self.state.__dict__["all"], self.state.__dict__["stations"] = open_lylout(files)
            logger.info("All LYLOUT files opened.")
            self.ui.timemin.setText(self.state.all["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"))
            self.ui.timemax.setText(self.state.all["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"))
            self.ui.stats.set_data(self.state.stations, face_color=None, size=5, edge_width=1, edge_color="red", symbol="square")
            self.filter()
            dialog.close()

    def import_entln(self) -> None:
        """Open and load one or more ENTLN files, updating lightning strike data.

        Loads ENTLN files selected by the user, updates the ground strike
        dataset (`self.state.gsd`), assigns colors and symbols for CG and CC
        strikes, and refreshes the UI plots.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if not len(self.state.all) > 0:
            logger.warning("No LYLOUT data, cannot plot ENTLN")
            return

        files, _ = QFileDialog.getOpenFileNames(self, "Select ENTLN files", self.settings.value("entln_folder", ""), "CSV files (*.csv)")
        if files:
            dialog = LoadingDialog("Opening selected ENTLN files...")
            dialog.show()
            QApplication.processEvents()
            self.settings.setValue("entln_folder", str(Path(files[0]).parent))
            # See import_lylout for syntactic reasoning
            temp = open_entln(files, self.state.all["datetime"].min())
            self.state.gsd = temp
            colors = [(1,0,0,1) if pc >= 0 else (0,0,1,1) for pc in temp["peakcurrent"]]
            self.state.__dict__["gsd"]["colors"] = colors
            logger.info("All ENTLN files opened.")
            dialog.close()

            if not self.state.gsd.empty:
                # CG strikes
                self.ui.v0.add(self.ui.gs0)
                self.ui.v1.add(self.ui.gs1)
                self.ui.v3.add(self.ui.gs3)
                self.ui.v4.add(self.ui.gs4)
                # CC strikes
                self.ui.v0.add(self.ui.cc0)
                self.ui.v1.add(self.ui.cc1)
                self.ui.v3.add(self.ui.cc3)
                self.ui.v4.add(self.ui.cc4)
                self.state.gsd["symbol"] =  ["triangle_up" if (val == 0) or (val == 40) else "x" for val in temp["type"]]

                gs_data = temp[(temp["type"] == 0) | (temp["type"] == 40)]
                cc_data = temp[temp["type"] == 1]

                # Statements were becoming too long
                if not gs_data.empty:
                    self.ui.gs0.set_data(
                        pos=np.column_stack([gs_data["utc_sec"].to_numpy(dtype=np.float32), gs_data["alt"].to_numpy(dtype=np.float32)]),
                        face_color=gs_data["colors"].to_list(),
                        edge_color=gs_data["colors"].to_list(),
                        size=5,
                        symbol=gs_data["symbol"],
                    )
                if not cc_data.empty:
                    self.ui.cc0.set_data(
                        pos=np.column_stack([cc_data["utc_sec"].to_numpy(dtype=np.float32), cc_data["alt"].to_numpy(dtype=np.float32)]),
                        face_color=cc_data["colors"].to_list(),
                        edge_color=cc_data["colors"].to_list(),
                        size=5,
                        symbol=cc_data["symbol"],
                    )

                coords = [["lon", "alt"], ["lon", "lat"], ["alt", "lat"]]
                targets = [self.ui.gs1, self.ui.gs3, self.ui.gs4]
                cc_targets = [self.ui.cc1, self.ui.cc3, self.ui.cc4]

                # Same logic as before, just condensed to a loop
                for (x, y), gs_plot, cc_plot in zip(coords, targets, cc_targets, strict=True):
                    gs_data = temp[(temp["type"] == 0) | (temp["type"] == 40)]
                    cc_data = temp[temp["type"] == 1]

                    if not gs_data.empty:
                        gs_plot.set_data(
                            pos=gs_data[[x, y]].to_numpy(dtype=np.float32),
                            face_color=gs_data["colors"].to_list(),
                            edge_color=gs_data["colors"].to_list(),
                            size=5,
                            symbol=gs_data["symbol"],
                        )

                    if not cc_data.empty:
                        cc_plot.set_data(
                            pos=cc_data[[x, y]].to_numpy(dtype=np.float32),
                            face_color=cc_data["colors"].to_list(),
                            edge_color=cc_data["colors"].to_list(),
                            size=5,
                            symbol=cc_data["symbol"],
                        )


                # Induce change for state saving
                self.state.__dict__["gsd_mask"] = np.ones([temp.shape[0]], dtype=bool)

    def import_state(self) -> None:
        """Load a previously saved application state from 'state/state.pkl'.

        Reads a pickled state file and updates the internal `State` object
        including datasets, plot options, and stations.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        try:
            with Path.open("state/state.pkl", "rb") as file:
                save_state = pickle.load(file)
                self.state.update(all = save_state["all"], stations = self.state.stations, plot = save_state["plot"], plot_options = save_state["plot_options"])
            logger.info("Loaded state in state/state.pkl.")
        except Exception as e:
            logger.warning("Could not load state/state.pkl due to %s.", e)

    def export_dat(self) -> None:
        """Export selected LYLOUT data to 10-minute interval .dat files.

        Groups the currently plotted data into 10-minute bins and writes
        formatted .dat files to the output folder with relevant metadata.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        temp = self.state.all[self.state.plot]
        start = temp["datetime"].min().floor("10min")
        end = temp["datetime"].max().ceil("10min")
        bins = date_range(start, end, freq="10min", inclusive="left")
        for i, chunk in enumerate(bins):
            filename = f"LYLOUT_{datetime.strftime(chunk, '%y%m%d_%H%M%S')}_0600"
            start = chunk
            end = bins[i+1]
            df_chunk = temp[(temp["datetime"] >= start) & (temp["datetime"] < end)]
            df_chunk = df_chunk[["utc_sec", "lat", "lon", "alt", "chi", "number_stations", "pdb", "mask"]]
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
            with Path.open(f"./output/{filename}.exported.dat", "w", newline="") as file:
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

    def export_parquet(self) -> None:
        """Export the currently plotted LYLOUT data to a Parquet file.

        Writes the plotted dataset to 'output/lylout.parquet' for efficient
        storage and later use.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        try:
            if not self.state.all.empty:
                self.state.all[self.state.plot].to_parquet("output/lylout.parquet", index=False)
                logger.info("Saved file in output/lylout.parquet.")
        except Exception as e:
            logger.warning("Could not save file in output/lylout.parquet due to %s.", e)

    def export_state(self) -> None:
        """Save the current application state to 'state/state.pkl'.

        Serializes the state attributes including datasets, plot options,
        and stations, enabling later restoration.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        try:
            with Path.open("state/state.pkl", "wb") as file:
                save_state = {"all": self.state.all, "stations": self.state.stations, "plot": self.state.plot, "plot_options": self.state.plot_options }
                pickle.dump(save_state, file)
            logger.info("Saved state in state/state.pkl.")
        except Exception as e:
            logger.warning("Could not save state in state/state.pkl due to %s", e)

    def export_image(self) -> None:
        """Capture the current view widget and save it as a PDF image.

        Grabs the content of the main view widget and writes it to
        'output/image.pdf'.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        try:
            pixmap = self.ui.view_widget.grab()
            pixmap.save("output/image.pdf")
            logger.info("Saved image in output/image.pdf")
        except Exception as e:
            logger.warning("Could not save image in output/image.pdf due to %s", e)

    def options_clear(self) -> None:
        """Clear all plotted data from plots and histograms.

        Resets the scatter plots (`s0`, `s1`, `s3`, `s4`) and histogram (`hist`)
        in the UI to empty arrays, effectively clearing visualizations.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.ui.s0.set_data(np.empty((0, 2)))
        self.ui.s1.set_data(np.empty((0, 2)))
        self.ui.s3.set_data(np.empty((0, 2)))
        self.ui.s4.set_data(np.empty((0, 2)))
        self.ui.hist.set_data(np.empty((0, 2)))

    def options_reset(self) -> None:
        """Reset the camera views of all main visualization widgets.

        Restores the default orientation and zoom for the cameras of view
        widgets `v0` through `v4`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.ui.v0.camera.reset()
        self.ui.v1.camera.reset()
        self.ui.v2.camera.reset()
        self.ui.v3.camera.reset()
        self.ui.v4.camera.reset()

    def flash_dtd(self) -> None:
        """Run the dot-to-dot (DTD) flash detection algorithm.

        Executes the `dot_to_dot` function on the current state, updating
        flash detection results while showing a loading dialog.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        dialog = LoadingDialog("Running dot to dot flash algorithm..")
        dialog.show()
        QApplication.processEvents()
        dot_to_dot(self.state)
        dialog.close()

    def flash_mccaul(self) -> None:
        """Run the McCaul flash detection algorithm.

        Executes the `mc_caul` function on the current state, updating
        flash detection results while showing a loading dialog.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        dialog = LoadingDialog("Running McCaul flash algorithm..")
        dialog.show()
        QApplication.processEvents()
        mc_caul(self.state)
        dialog.close()

    def help_about(self) -> None:
        """Open the HLMA project About webpage.

        Launches the HLMA About page in the default web browser.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        webbrowser.open("https://lightning.tamu.edu/hlma/")

    def help_contact(self) -> None:
        """Open the HLMA contact webpage.

        Launches the contact page for Timothy Logan in the default web browser.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        webbrowser.open("https://artsci.tamu.edu/atmos-science/contact/profiles/timothy-logan.html")

    def help_color(self) -> None:
        """Open the Colorcet colormap user guide.

        Launches the Colorcet documentation for linear sequential colormaps.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        webbrowser.open("https://colorcet.holoviz.org/user_guide/Continuous.html#linear-sequential-colormaps-for-plotting-magnitudes")

    def filter(self) -> None:
        """Apply user-defined filters to the loaded lightning data.

        Reads filter criteria from UI elements, evaluates the query against
        `self.state.all`, updates `self.state.plot`, resets polygon selection
        mask, and triggers a replot.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if self.state.all is None:
            return
        query = (
            f"(datetime >= '{self.ui.timemin.text()}') & (datetime <= '{self.ui.timemax.text()}') & "
            f"(alt >= {float(self.ui.altmin.text()) * 1000}) & (alt <= {float(self.ui.altmax.text()) * 1000}) & "
            f"(chi >= {self.ui.chimin.text()}) & (chi <= {self.ui.chimax.text()}) & "
            f"(pdb >= {self.ui.powermin.text()}) & (pdb <= {self.ui.powermax.text()}) & "
            f"(number_stations >= {self.ui.stationsmin.text()})"
        )
        self.state.__dict__["plot"] = self.state.all.eval(query)
        self.polyfilter.inc_mask = np.zeros(np.count_nonzero(self.state.plot), dtype=bool)
        self.state.replot()

    def animate(self) -> None:
        """Start animating the currently loaded lightning data.

        Initializes animation timing, sets the active flag, and starts the
        animation timer.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if self.state.all.empty:
            logger.info("No data to animate.")
            return

        logger.info("Starting animation.")

        self.anim.start_time = time.perf_counter()
        self.anim.active = True
        self.anim.timer.start()

    def _animate_step(self, _: object) -> None:
        """Perform a single step of the lightning data animation.

        Updates visualization plots incrementally based on elapsed time and
        stops the animation when complete.

        Parameters
        ----------
        _ : object
            Timer event passed by the animation timer.

        Returns
        -------
        None

        """
        if not self.anim.active:
            return
        n = np.count_nonzero(self.state.plot)
        elapsed = time.perf_counter() - self.anim.start_time
        progress = min(1.0, elapsed / self.anim.duration)
        n_vis = int(progress * n)

        # FIXME: should find a way to do this without having to recalculate this every time
        temp = self.state.all[self.state.plot].sort_values(by=self.anim.var).iloc[:n_vis]
        temp.alt /= 1000
        cvar = self.state.plot_options.cvar
        cmap = self.state.plot_options.cmap
        arr = temp[cvar].to_numpy()
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        colors = cmap(norm)

        positions = np.column_stack([temp["seconds"].to_numpy(dtype=np.float32),temp["alt"].to_numpy(dtype=np.float32)])
        self.ui.s0.set_data(pos=positions, face_color=colors, size=1, edge_width=0)

        positions = temp[["lon", "alt"]].to_numpy().astype(np.float32)
        self.ui.s1.set_data(pos=positions, face_color=colors, size=1, edge_width=0)

        bins = 200
        counts, edges = np.histogram(temp["alt"], bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        line_data = np.column_stack([counts, centers])
        self.ui.hist.set_data(pos=line_data, color=(1, 1, 1, 1), width=1)

        positions = temp[["lon", "lat"]].to_numpy().astype(np.float32)
        self.ui.s3.set_data(pos=positions, face_color=colors, size=1, edge_width=0)

        positions = temp[["alt", "lat"]].to_numpy().astype(np.float32)
        self.ui.s4.set_data(pos=positions, face_color=colors, size=1, edge_width=0)

        if progress >= 1.0:
            logger.info("Finished animation.")
            self.anim.timer.stop()
            self.anim.active = False

    def visplot(self) -> None:
        """Update all visualization plots with the current lightning and map data.

        Refreshes scatter plots (`s0`, `s1`, `s3`, `s4`), histogram (`hist`),
        and map layers using the currently selected data (`self.state.plot`) and
        map features (`self.state.plot_options.map` and `features`). Handles
        coloring based on the selected variable and adds or removes CG and CC
        flash visuals depending on data availability.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        logger.info("Starting vis.py plotting.")
        temp = self.state.all[self.state.plot]
        temp.alt /= 1000
        cvar = self.state.plot_options.cvar
        cmap = self.state.plot_options.cmap
        arr = temp[cvar].to_numpy()
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        colors = cmap(norm)

        positions = np.column_stack([temp["seconds"].to_numpy(dtype=np.float32),temp["alt"].to_numpy(dtype=np.float32)])
        self.ui.s0.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        self.ui.v0.camera.set_range(x=(positions[:,0].min(), positions[:,0].max()), y=(0, 20))
        self.ui.v0.camera.set_default_state()

        positions = temp[["lon", "alt"]].to_numpy().astype(np.float32)
        self.ui.s1.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        self.ui.v1.camera.set_range(x=(positions[:,0].min(), positions[:,0].max()), y=(0, positions[:,1].max()))
        self.ui.v1.camera.set_default_state()

        bins = 200
        counts, edges = np.histogram(temp["alt"], bins=bins)
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
            gdf = fdict["gdf"].cx[minx:maxx, miny:maxy]
            if gdf.empty:
                continue
            pos_array = np.vstack(gdf.geometry.explode(index_parts=False).apply(lambda p: np.append(np.array(p.coords, np.float32),np.array([[np.nan, np.nan]]), axis=0)).values)
            colors_array = np.tile(np.array(mcolors.to_rgba(fdict["color"]), dtype=np.float32), (pos_array.shape[0], 1))
            positions_list.append(pos_array)
            colors_list.append(colors_array)
        if positions_list:
            map_positions = np.vstack(positions_list)
            map_colors = np.vstack(colors_list)
        else:
            map_positions = np.empty((0, 2), dtype=np.float32)
            map_colors = np.empty((0, 4), dtype=np.float32)
        self.ui.map.set_data(pos=map_positions, color=map_colors)
        positions = temp[["lon", "lat"]].to_numpy().astype(np.float32)
        self.ui.s3.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        self.ui.v3.camera.set_range(x=(positions[:,0].min(), positions[:,0].max()), y=(positions[:,1].min(), positions[:,1].max()))
        self.ui.v3.camera.set_default_state()

        positions = temp[["alt", "lat"]].to_numpy().astype(np.float32)
        self.ui.s4.set_data(pos=positions, face_color=colors, size=1, edge_width=0)
        self.ui.v4.camera.set_range(x=(0, positions[:,0].max()), y=(positions[:,1].min(), positions[:,1].max()))
        self.ui.v4.camera.set_default_state()

        temp = self.state.gsd[self.state.gsd_mask]
        if self.state.gsd.empty:
            gs_empty = True
            cc_empty = True
        else:
            gs_mask = self.state.gsd["type"].isin([0, 40])
            cc_mask = self.state.gsd["type"] == 1

            gs_empty = len(self.state.gsd[gs_mask & self.state.gsd_mask]) == 0
            cc_empty = len(self.state.gsd[cc_mask & self.state.gsd_mask]) == 0

        # Handle when only gs data is empty
        if gs_empty and not cc_empty:
            # Remove only gs visuals
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

        # Handle when only cc data is empty
        elif cc_empty and not gs_empty:
            # Remove only cc visuals
            if self.ui.cc0 in self.ui.v0.scene.children:
                self.ui.v0.scene.children.remove(self.ui.cc0)
                self.ui.cc0.parent = None
            if self.ui.cc1 in self.ui.v1.scene.children:
                self.ui.v1.scene.children.remove(self.ui.cc1)
                self.ui.cc1.parent = None
            if self.ui.cc3 in self.ui.v3.scene.children:
                self.ui.v3.scene.children.remove(self.ui.cc3)
                self.ui.cc3.parent = None
            if self.ui.cc4 in self.ui.v4.scene.children:
                self.ui.v4.scene.children.remove(self.ui.cc4)
                self.ui.cc4.parent = None

        # Handle when both are empty
        elif gs_empty and cc_empty:
            # Remove both gs and cc visuals
            targets = [
                (self.ui.v0, self.ui.gs0, self.ui.cc0),
                (self.ui.v1, self.ui.gs1, self.ui.cc1),
                (self.ui.v3, self.ui.gs3, self.ui.cc3),
                (self.ui.v4, self.ui.gs4, self.ui.cc4),
            ]
            for v, gs, cc in targets:
                if gs in v.scene.children:
                    v.scene.children.remove(gs)
                    gs.parent = None
                if cc in v.scene.children:
                    v.scene.children.remove(cc)
                    cc.parent = None
        # Both are not empty
        else:
            if self.ui.gs0 not in self.ui.v0.scene.children:
                self.ui.v0.add(self.ui.gs0)
            if self.ui.gs1 not in self.ui.v1.scene.children:
                self.ui.v1.add(self.ui.gs1)
            if self.ui.gs3 not in self.ui.v3.scene.children:
                self.ui.v3.add(self.ui.gs3)
            if self.ui.gs4 not in self.ui.v4.scene.children:
                self.ui.v4.add(self.ui.gs4)

            if self.ui.cc0 not in self.ui.v0.scene.children:
                self.ui.v0.add(self.ui.cc0)
            if self.ui.cc1 not in self.ui.v1.scene.children:
                self.ui.v1.add(self.ui.cc1)
            if self.ui.cc3 not in self.ui.v3.scene.children:
                self.ui.v3.add(self.ui.cc3)
            if self.ui.cc4 not in self.ui.v4.scene.children:
                self.ui.v4.add(self.ui.cc4)

            gs_data = temp[(temp["type"] == 0) | (temp["type"] == 40)]
            cc_data = temp[temp["type"] == 1]

            # Statements were becoming too long
            if not gs_data.empty:
                self.ui.gs0.set_data(
                    pos=np.column_stack([gs_data["utc_sec"].to_numpy(dtype=np.float32), gs_data["alt"].to_numpy(dtype=np.float32)]),
                    face_color=gs_data["colors"].to_list(),
                    edge_color=gs_data["colors"].to_list(),
                    size=5,
                    symbol=gs_data["symbol"],
                )
            if not cc_data.empty:
                self.ui.cc0.set_data(
                    pos=np.column_stack([cc_data["utc_sec"].to_numpy(dtype=np.float32), cc_data["alt"].to_numpy(dtype=np.float32)]),
                    face_color=cc_data["colors"].to_list(),
                    edge_color=cc_data["colors"].to_list(),
                    size=5,
                    symbol=cc_data["symbol"],
                )

            coords = [["lon", "alt"], ["lon", "lat"], ["alt", "lat"]]
            targets = [self.ui.gs1, self.ui.gs3, self.ui.gs4]
            cc_targets = [self.ui.cc1, self.ui.cc3, self.ui.cc4]

            # Same logic as before, just condensed to a loop
            for (x, y), gs_plot, cc_plot in zip(coords, targets, cc_targets, strict=True):
                gs_data = temp[(temp["type"] == 0) | (temp["type"] == 40)]
                cc_data = temp[temp["type"] == 1]

                if not gs_data.empty:
                    gs_plot.set_data(
                        pos=gs_data[[x, y]].to_numpy(dtype=np.float32),
                        face_color=gs_data["colors"].to_list(),
                        edge_color=gs_data["colors"].to_list(),
                        size=5,
                        symbol=gs_data["symbol"],
                    )

                if not cc_data.empty:
                    cc_plot.set_data(
                        pos=cc_data[[x, y]].to_numpy(dtype=np.float32),
                        face_color=cc_data["colors"].to_list(),
                        edge_color=cc_data["colors"].to_list(),
                        size=5,
                        symbol=cc_data["symbol"],
                    )


        logger.info("Finished vis.py plotting.")

    def undo(self) -> None:
        """Revert the last user action or selection.

        If polygon selections exist in `self.polyfilter`, removes the last
        polygon. Otherwise, restores the previous application state from
        `self.state.history` and updates `self.state.future` to allow redo.
        Refreshes visualizations via `self.state.replot()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        logger.info("Undo called")
        if len(self.polyfilter.clicks) > 0:
            self.polyfilter.clicks.pop()
            self.polyfilter.handle_poly_plot()
        elif self.state.history:
            self.state.future.append(self.state)
            self.state = self.state.history.pop()
            self.state.replot()

    def redo(self) -> None:
        """Reapply an action that was previously undone.

        Restores the most recent future state from `self.state.future`, moves
        the current state to `self.state.history`, and updates visualizations
        by calling `self.state.replot()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        logger.info("Redo called")
        if self.state.future:
            self.state.history.append(self.state)
            self.state = self.state.future.pop()
            self.state.replot()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = HLMA()
    window.setWindowTitle("Aggie XLMA")
    window.setWindowIcon(QIcon("assets/icons/hlma.svg"))
    window.show()
    sys.exit(app.exec())
