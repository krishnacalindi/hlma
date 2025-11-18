"""Interactive polygon-based filtering utilities for linked scientific plots.

This module provides the :class:`PolygonFilter` class, which enables
click-driven polygon selection across multiple coordinated views such
as time-altitude, longitude-altitude, longitude-latitude, and
altitude-latitude plots. Users can draw polygons or ranges directly
on the plots, and the module computes corresponding boolean masks for
both the main dataset and a supplementary GSD dataset.

The module also handles updating plot visuals, hiding GS or CC layers
when they become empty, and writing updated masks back into the
application state.
"""

import logging

import numpy as np
from shapely.geometry import Polygon
from shapely.vectorized import contains

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("polygon.py")
logger.setLevel(logging.DEBUG)
TIME_ALT_PLOT = 0
LON_ALT_PLOT = 1
LON_LAT_PLOT = 3
ALT_LAT_PLOT = 4
LEFT_CLICK = 1
RIGHT_CLICK = 2

class PolygonFilter:
    """Interactive polygon-based filter for coordinated scientific visualizations.

    The ``PolygonFilter`` class listens for mouse events across several linked
    views, records user clicks, renders temporary polygon shapes, and computes
    selection masks based on either range constraints or true 2D polygon
    inclusion tests. The resulting masks are used to filter both the main
    trajectory dataset and a secondary GSD dataset (GS/CC types), with full
    visual updates applied when requested.

    The class supports both inclusion and exclusion modes and integrates with
    an external application that provides ``ui`` and ``state`` objects. It
    assumes the UI exposes view widgets (``v0``, ``v1``, ``v3``, ``v4``),
    point-drawing objects (``pd*`` and ``pl*``), and GS/CC visual objects
    (``gs*`` and ``cc*``). Filtering results are written back to
    ``obj.state.update`` along with visualization changes.

    Parameters
    ----------
    obj : Any, optional
        Parent object that must define ``ui`` and ``state`` attributes, where
        ``ui`` provides the plotting widgets and ``state`` holds boolean masks and
        pandas data. If ``None``, event registration will not occur.
    remove : bool, default False
        If True, polygon selection excludes (removes) points from the mask;
        otherwise it includes points.

    Attributes
    ----------
    obj : Any or None
        Parent application object containing ``ui`` and ``state``.
    ui : Any
        UI container providing plot canvases, views, and visual elements.
    remove : bool
        Whether polygon selection removes points instead of including them.
    prev_ax : int or None
        The last active view index where clicking occurred.
    clicks : list of tuple of float
        Accumulated polygon vertices in the active view.
    view_index : int or None
        Currently active view for receiving click events.

    """

    def __init__(self, obj: object, remove: bool = False) -> None:
        """Initialize the PolygonFilter, register mouse callbacks, and prepare internal state for polygon-based selection.

        Parameters
        ----------
        obj : Any, optional
            Parent object that must contain ``ui`` and ``state`` attributes.
        remove : bool, default False
            If True, polygon selection removes points instead of including them.

        """
        self.obj = obj
        self.ui = obj.ui
        self.remove = remove
        self.prev_ax = None
        self.clicks = []
        self.view_index = None

        self.ui.c0.events.mouse_press.connect(lambda ev: self.on_click(ev, view_index=0))
        self.ui.c1.events.mouse_press.connect(lambda ev: self.on_click(ev, view_index=1))
        self.ui.c3.events.mouse_press.connect(lambda ev: self.on_click(ev, view_index=3))
        self.ui.c4.events.mouse_press.connect(lambda ev: self.on_click(ev, view_index=4))

    def on_click(self, event: object, view_index: int) -> None:
        """Handle a mouse click event on one of the linked views.

        This records the click position, updates temporary polygon drawings,
        and when a right-click completes a polygon, triggers filtering.

        This is intended to be connected to a VisPy canvas

        Parameters
        ----------
        event : Any
            Mouse event containing ``pos`` and ``button`` attributes.
        view_index : int
            Index of the plot/view where the click occurred.

        """
        self.view_index = view_index
        pos = event.pos
        lines = self.ui.__getattribute__(f"pl{view_index}")
        transform = lines.transforms.get_transform(map_from="canvas", map_to="visual")
        x, y = transform.map(pos)[:2]

        if self.prev_ax is None or self.prev_ax == view_index:
            if event.button == LEFT_CLICK:
                self.prev_ax = view_index
                self.clicks.append((x, y))
                self.handle_poly_plot()

            elif event.button == RIGHT_CLICK:
                if len(self.clicks) > 1:
                    # Store the current axis index for use in polygon logic
                    self.prev_ax = view_index

                    self.polygon(self.view_index, update=True)

                    self.clicks.clear()
                    self.handle_poly_plot()

        event.handled = True

    def handle_poly_plot(self) -> None:
        """Update the temporary polygon drawing in the active view.

        This draws the clicked vertices as red dots and, once at least
        two points exist, draws the connecting red lines.
        Clears visuals if no points remain.
        """
        view = self.ui.__getattribute__(f"v{self.view_index}")
        dots = self.ui.__getattribute__(f"pd{self.view_index}")
        lines = self.ui.__getattribute__(f"pl{self.view_index}")

        if len(self.clicks) == 0:
            view.scene.children.remove(dots)
            dots.parent = None
            lines.set_data(np.empty((0, 2)))
            self.prev_ax = None
            self.clicks.clear()
        elif len(self.clicks) == 1:
            if dots.parent is None:
                view.add(dots)

            dots.set_data(np.array(self.clicks), face_color="red", size=5)
            lines.set_data(np.empty((0, 2)))
        else:
            dots.set_data(np.array(self.clicks), face_color="red", size=5)
            lines.set_data(self.clicks, color="red")

    def update_filter(self, new: bool) -> None:
        """Set whether polygon selection includes or excludes points.

        Parameters
        ----------
        new : bool
            If True, selected points are removed instead of included.

        """
        logger.info("remove is now %s", new)
        self.remove = new

    def _handle_empty_data(self, new_mask: np.ndarray) -> None:
        """Hide GS and/or CC visuals when their filtered subsets become empty.

        This inspects the GSD table after polygon filtering and removes the
        corresponding visuals (GS, CC, or both) from all linked views.

        Note: This is a helper for polygon.py be aware that calling it outside of intended circumstances
        may cause issues.

        Parameters
        ----------
        new_mask : ndarray
            Boolean mask applied to ``obj.state.gsd`` after polygon filtering.

        """
        if self.obj.state.gsd.empty:
            gs_empty = True
            cc_empty = True
        else:
            gs_mask = self.obj.state.gsd["type"].isin([0, 40])
            cc_mask = self.obj.state.gsd["type"] == 1

            gs_empty = len(self.obj.state.gsd[gs_mask & new_mask]) == 0
            cc_empty = len(self.obj.state.gsd[cc_mask & new_mask]) == 0

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

    def polygon(self, num: int, update: bool = False) -> np.ndarray | None:
        """Compute a polygon-based mask for the active plot, and optionally update all visuals and the application state.

        Depending on the active view, the selection is interpreted as a
        range filter (time-altitude, lon-altitude, alt-latitude) or a
        2D polygon (lon-latitude). When ``update=True``, the method updates
        plot masks, GSD masks, and redraws GS/CC visuals.

        Parameters
        ----------
        num : int
            Identifier of the active plot type (e.g. TIME_ALT_PLOT).
        update : bool, default False
            If True, apply the masks to ``obj.state`` and update visuals.

        Returns
        -------
        ndarray or None
            Returns the boolean mask for the plot when ``update=False``.
            Returns None when applying updates.

        """
        logger.info("Started filtering.")
        lyl_mask = self.obj.state.plot

        lyl_temp = self.obj.state.all[self.obj.state.plot]
        entln_temp = self.obj.state.gsd[self.obj.state.gsd_mask]

        has_entln = not entln_temp.empty
        if num == TIME_ALT_PLOT:
            x_values = [pt[0] for pt in self.clicks]  # extract x (time in seconds)
            min_x = min(x_values)
            max_x = max(x_values)

            lyl_mask = (lyl_temp["seconds"] > min_x) & (lyl_temp["seconds"] < max_x)
            if has_entln:
                entln_mask = (entln_temp["utc_sec"] > min_x) & (entln_temp["utc_sec"] < max_x)
        elif num == LON_ALT_PLOT:
            x_values = [pt[0] for pt in self.clicks]
            min_x = min(x_values)
            max_x = max(x_values)

            lyl_mask = (lyl_temp["lon"] > min_x) & (lyl_temp["lon"] < max_x)
            if has_entln:
                entln_mask = (entln_temp["lon"] > min_x) & (entln_temp["lon"] < max_x)
        elif num == LON_LAT_PLOT:
            polygon = Polygon(self.clicks)
            lon = lyl_temp["lon"].to_numpy()
            lat = lyl_temp["lat"].to_numpy()

            lyl_mask = contains(polygon, lon, lat)
            if has_entln:
                entln_lon = entln_temp["lon"].to_numpy()
                entln_lat = entln_temp["lat"].to_numpy()
                entln_mask = contains(polygon, entln_lon, entln_lat)
        elif num == ALT_LAT_PLOT:
            y_values = [pt[1] for pt in self.clicks]
            min_y = min(y_values)
            max_y = max(y_values)

            lyl_mask = (lyl_temp["lat"] > min_y) & (lyl_temp["lat"] < max_y)
            if has_entln:
                entln_mask = (entln_temp["lat"] > min_y) & (entln_temp["lat"] < max_y)

        logger.info("Mask created.")

        temp_mask = np.zeros(len(self.obj.state.all), dtype=bool)
        temp_mask[lyl_temp.index] = (lyl_mask ^ self.remove)
        new_mask = temp_mask
        update_gsd_mask = False

        if update:
            if has_entln:
                temp_mask = np.zeros(len(self.obj.state.gsd), dtype=bool)
                temp_mask[entln_temp.index] = (entln_mask ^ self.remove)

                # Separate GS and CC masks
                gsd_filtered = self.obj.state.gsd[temp_mask]
                gs_mask = gsd_filtered["type"].isin([0, 40])
                cc_mask = gsd_filtered["type"] == 1

                # Update GS visuals
                if gs_mask.any():
                    gs_data = gsd_filtered[gs_mask]
                    symbols = gs_data["symbol"].to_numpy()
                    colors = np.stack(gs_data["colors"].to_numpy())

                    positions = np.column_stack([gs_data["utc_sec"].to_numpy(dtype=np.float32), gs_data["alt"].to_numpy(dtype=np.float32)])
                    self.ui.gs0.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=symbols)

                    positions = gs_data[["lon", "alt"]].to_numpy().astype(np.float32)
                    self.ui.gs1.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=symbols)

                    positions = gs_data[["lon", "lat"]].to_numpy().astype(np.float32)
                    self.ui.gs3.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=symbols)

                    positions = gs_data[["alt", "lat"]].to_numpy().astype(np.float32)
                    self.ui.gs4.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=symbols)
                else:
                    self.ui.gs0.set_data(np.empty((0, 2)))
                    self.ui.gs1.set_data(np.empty((0, 2)))
                    self.ui.gs3.set_data(np.empty((0, 2)))
                    self.ui.gs4.set_data(np.empty((0, 2)))

                # Update CC visuals
                if cc_mask.any():
                    cc_data = gsd_filtered[cc_mask]
                    symbols = cc_data["symbol"].to_numpy()
                    colors = np.stack(cc_data["colors"].to_numpy())

                    positions = np.column_stack([cc_data["utc_sec"].to_numpy(dtype=np.float32), cc_data["alt"].to_numpy(dtype=np.float32)])
                    self.ui.cc0.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=symbols)

                    positions = cc_data[["lon", "alt"]].to_numpy().astype(np.float32)
                    self.ui.cc1.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=symbols)

                    positions = cc_data[["lon", "lat"]].to_numpy().astype(np.float32)
                    self.ui.cc3.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=symbols)

                    positions = cc_data[["alt", "lat"]].to_numpy().astype(np.float32)
                    self.ui.cc4.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=symbols)
                else:
                    self.ui.cc0.set_data(np.empty((0, 2)))
                    self.ui.cc1.set_data(np.empty((0, 2)))
                    self.ui.cc3.set_data(np.empty((0, 2)))
                    self.ui.cc4.set_data(np.empty((0, 2)))

                # Update gsd_mask in state
                update_gsd_mask = True

            # Always update main plot mask
            if update_gsd_mask:
                self._handle_empty_data(temp_mask)
                self.obj.state.update(plot=new_mask, gsd_mask=temp_mask)
            else:
                self.obj.state.update(plot=new_mask)
            return None
        return lyl_mask
