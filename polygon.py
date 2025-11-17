"""Interactive polygon and range-based filtering tools for the XLMA-style Python/VisPy/PyQt visualization environment.

This module provides the logic required to reproduce XLMA's graphical
selection and filtering capabilities in Python. It supports multi-view
click interactions across linked VisPy canvases, allowing users to draw
polygons or axis-aligned bounding regions in different projections of the
dataset (e.g., time-altitude, lon-lat, lon-alt, alt-lat). The resulting
selections are converted into boolean masks that update the application's
shared data state, enabling iterative and exploratory scientific analysis.

Core Features
-------------
- Mouse-driven point collection across multiple VisPy views
- Live visualization of selection dots and connecting line segments
- Polygon creation using user-clicked vertices (:meth:`PolygonFilter.polygon`)
- Spatial filtering (e.g., lon/lat polygons)
- Axis-aligned range filtering (e.g., time, altitude, longitude, latitude)
- Inclusion and removal modes matching XLMA's filtering behavior
- Integration with a central application state model

Intended Use
------------
This module is part of a broader system replicating XLMA's interactive
data exploration tools. It is designed to operate alongside a VisPy + PyQt
frontend, with dynamically named UI components (e.g., ``v{index}``,
``pd{index}``, ``pl{index}``) and a state object containing the full dataset
and the currently active mask.

The main entry point is the :class:`PolygonFilter`, which encapsulates all
interaction, selection, and mask-building behavior. Other parts of the
application—renderers, controllers, and UI panels—use this class to
coordinate filtering operations during user interaction.

Example Usage
-------------
.. code-block:: python

    # In the application controller or main window:

    # Create the polygon filtering tool and attach it to the UI/state container.
    self.polyfilter = PolygonFilter(self)

    # Connect PyQt menu actions for switching between keep/remove modes.
    ui.filter_menu_keep.triggered.connect(
        lambda val: self.polyfilter.update_filter(False)
    )
    ui.filter_menu_remove.triggered.connect(
        lambda val: self.polyfilter.update_filter(True)
    )

    # Once initialized, PolygonFilter automatically binds mouse-click handlers
    # to the configured VisPy canvases. During interaction:
    # - Left-click adds points to a temporary polygon or range selection
    # - Right-click finalizes the selection and applies the mask
    # - Visual dots and lines update in real time
    # - The internal state object receives updated masks when required

References
----------
- :class:`PolygonFilter` — main class for handling polygon and range selection
- :meth:`PolygonFilter.on_click` — mouse click handler
- :meth:`PolygonFilter.handle_poly_plot` — updates visuals for active selections
- :meth:`PolygonFilter.update_filter` — toggles keep/remove mode
- :meth:`PolygonFilter.polygon` — builds masks from user-defined polygons/ranges

Notes
-----
This module does not define rendering logic or dataset semantics; it only
defines the interactive tools for creating masks. The specifics of the data
(e.g., meaning of lon/lat/time/altitude fields) are determined by the
application's state object.

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
    """Interactive polygon and range-selection filter for XLMA-style scientific data.

    Within a VisPy + PyQt visualization environment. This class provides the user-interaction logic needed to replicate XLMA's
    graphical filtering tools in Python. Users can click on multiple linked plot
    views (e.g., time-altitude, lon-lat, lon-alt, alt-lat) to draw polygons or
    axis-aligned boxes that filter the underlying dataset. The resulting mask can
    be applied to the application's state, enabling exploratory data analysis
    consistent with XLMA workflows.

    The class manages:
      • click handling across multiple VisPy canvases
      • live display of selection points and line segments
      • polygon construction for spatial filters
      • bounding-range selection for time/altitude/longitude/latitude views
      • inclusion or removal filtering modes
      • updates to a shared application data state

    Integration
    -----------
    This class operates within a VisPy + PyQt UI layer, relying on:
      • dynamically named view and plot objects (e.g., ``v{index}``, ``pd{index}``, ``pl{index}``)
      • VisPy event connections for mouse interaction
      • an application `state` object storing full and filtered datasets
      • XLMA-style multi-panel plots showing different projections of the data

    Parameters
    ----------
    obj : object
        The application controller or model providing:
          - ``ui`` : VisPy/PyQt UI components and plot objects
          - ``state`` : data container with ``all`` and ``plot`` attributes
    remove : bool, optional
        If True, selections will exclude data rather than include it.
        Implemented via XOR masking to mirror XLMA's “remove mode.”

    Notes
    -----
    This class does not perform rendering directly; instead it updates the
    VisPy scene graph and application state, allowing the UI layer to reflect
    filtering changes in real time. It is intended to be lightweight and
    UI-agnostic beyond the established XLMA-compatible naming conventions.

    """

    def __init__(self, obj: object | None = None, remove: bool = False) -> None:
        """Initialize the instance with interaction state, UI bindings, and click tracking.

        Parameters
        ----------
        obj : object | None, optional
            An object expected to contain a `ui` attribute and a `state` attribute
            with an iterable `all`. If None, attributes depending on `obj` may fail.
        remove : bool, optional
            Flag indicating whether removal behavior is enabled for click actions.

        Attributes
        ----------
        obj : object | None
            Stored reference to the provided object.
        ui : Any
            User-interface object extracted from `obj.ui`.
        remove : bool
            Whether removal mode is active.
        prev_ax : Any
            Tracks the previously active axis (initially None).
        clicks : list
            Stores click event information.
        view_index : Any
            Index of the current view (initially None).

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
        """Handle mouse click events on a specific view.

        This method records click positions, updates the active view,
        and triggers polygon plotting logic depending on the mouse button used.

        Parameters
        ----------
        event : object
            The mouse event object. Expected to have:
            - `pos`: the click position in canvas coordinates.
            - `button`: mouse button integer (1 = left, 2 = right).
            - `handled`: boolean flag to mark the event as processed.
        view_index : int
            Index of the view (axis) on which the click occurred.

        Notes
        -----
        - Left click (`button == 1`) records a point and triggers polygon-preview plotting.
        - Right click (`button == 2`) finalizes a polygon if multiple points exist,
        updates internal structures, and clears stored points.

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
        """Update the visual representation of the polygon-in-progress.

        This method updates point markers and line segments for the current view
        based on how many points have been clicked so far. It handles three states:

        - **0 points**: Clears all markers and lines, resets masks and state.
        - **1 point**: Displays a single red dot but no connecting lines.
        - **2+ points**: Displays all points as red dots and draws lines connecting them.

        Notes
        -----
        - Uses dynamic view and plot elements such as `v{index}`, `pd{index}`, and `pl{index}`.
        - Modifies the internal click list and inclusion mask when resetting.
        - Intended to be called after each click or polygon update.

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
        """Update the removal-mode flag.

        Parameters
        ----------
        new : bool
            The new value indicating whether removal mode should be enabled.

        Notes
        -----
        Logs the change and updates the internal `remove` attribute.

        """
        logger.info("remove is now %b", new)
        self.remove = new

    def polygon(self, num: int, update: bool = False) -> np.ndarray | None:
        """Apply polygon-based or axis-aligned filtering based on the current click geometry.

        Parameters
        ----------
        num : int
            Identifier for which plot/view is being filtered. Expected to match one of:
            - `TIME_ALT_PLOT` - 0
            - `LON_ALT_PLOT`  - 1
            - `LON_LAT_PLOT`  - 3
            - `ALT_LAT_PLOT`  - 4
        update : bool, optional
            If True, the generated mask is written back to `self.obj.state.update(...)`
            and the function returns None. If False, the raw mask for `temp` is returned.

        Returns
        -------
        np.ndarray | None
            - Boolean mask aligned with the filtered subset (`temp`) if `update=False`
            - None if `update=True`

        Notes
        -----
        Filtering behavior:
        - **TIME_ALT_PLOT**: Filters by an x-range (seconds).
        - **LON_ALT_PLOT**: Filters by an x-range (longitude).
        - **LON_LAT_PLOT**: Uses a polygon to filter points based on (lon, lat).
        - **ALT_LAT_PLOT**: Filters by a y-range (latitude).

        The final mask is XOR-combined with `self.remove` to support removal mode.

        """
        logger.info("Started filtering.")
        lyl_mask = new_mask = self.obj.state.plot
        entln_mask = self.obj.state.gsd_mask

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
                self.obj.state.__dict__["gsd_mask"] = temp_mask

            # Always update main plot mask
            self.obj.state.update(plot=new_mask)
            return None
        return lyl_mask
