from shapely.geometry import Polygon
from shapely.vectorized import contains
import numpy as np
import pandas as pd

import logging
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("polygon.py")
logger.setLevel(logging.DEBUG)

class PolygonFilter(): 
    def __init__(self, obj=None, remove=False):
        self.obj = obj
        self.ui = obj.ui
        self.remove = remove
        self.prev_ax = None
        self.clicks = []
        self.view_index = None
        # self.inc_mask = np.zeros(len(obj.state.all), dtype=bool)

        self.ui.c0.events.mouse_press.connect(lambda ev: self.on_click(ev, view_index=0))
        self.ui.c1.events.mouse_press.connect(lambda ev: self.on_click(ev, view_index=1))
        self.ui.c3.events.mouse_press.connect(lambda ev: self.on_click(ev, view_index=3))
        self.ui.c4.events.mouse_press.connect(lambda ev: self.on_click(ev, view_index=4))

    def on_click(self, event, view_index):
        self.view_index = view_index
        pos = event.pos
        view = self.ui.__getattribute__(f'v{view_index}')
        dots = self.ui.__getattribute__(f'pd{view_index}')
        lines = self.ui.__getattribute__(f'pl{view_index}')     
        # print(f"{pos}")
        transform = lines.transforms.get_transform(map_from="canvas", map_to="visual")
        x, y = transform.map(pos)[:2]

        # print(f"Clicked on view {view_index}: x={x}, y={y}")
        if self.prev_ax == None or self.prev_ax == view_index:
            if event.button == 1: # Left click
                self.prev_ax = view_index
                self.clicks.append((x, y))
                self.handle_poly_plot()

                # if (((self.view_index in [0, 1, 4]) and len(self.clicks) == 2) or (len(self.clicks) >= 3 and self.view_index == 3)):                    
                #     self.inc_mask = self.polygon(view_index, False)

                #     # self.obj.state.replot()
                #     lyl_temp = self.obj.state.all[self.obj.state.plot]
                #     lyl_temp.alt /= 1000
                #     cvar = self.obj.state.plot_options.cvar
                #     cmap = self.obj.state.plot_options.cmap
                #     arr = lyl_temp[cvar].to_numpy()
                #     norm = (arr - arr.min()) / (arr.max() - arr.min())
                #     colors = cmap(norm)
                #     colors[~self.inc_mask, 3] = 0.5

                #     self.ui.s3.set_data(self.ui.s3._data, face_color=colors)
                #     self.ui.s3.update()
            elif event.button == 2: # Right click
                # print("Right click detected")
                if len(self.clicks) > 1:
                    # Store the current axis index for use in polygon logic
                    self.prev_ax = view_index

                    self.polygon(self.view_index, True)

                    self.clicks.clear()
                    self.handle_poly_plot()

        event.handled = True

    def handle_poly_plot(self):
        view = self.ui.__getattribute__(f'v{self.view_index}')
        dots = self.ui.__getattribute__(f'pd{self.view_index}')
        lines = self.ui.__getattribute__(f'pl{self.view_index}')    

        if len(self.clicks) == 0:
            view.scene.children.remove(dots)
            dots.parent = None
            lines.set_data(np.empty((0, 2)))
            self.prev_ax = None
            # self.inc_mask = np.zeros(len(self.obj.state.all), dtype=bool)
            self.clicks.clear()
        elif len(self.clicks) == 1:
            if dots.parent == None:
                view.add(dots)
            
            dots.set_data(np.array(self.clicks), face_color='red', size=5)
            lines.set_data(np.empty((0, 2)))
        else:
            dots.set_data(np.array(self.clicks), face_color='red', size=5)
            lines.set_data(self.clicks, color='red')

    def update_filter(self, new):
        logger.info(f"remove is now {new}")
        self.remove = new

    def polygon(self, num, update = False):
        logger.info(f"Started filtering.")
        lyl_mask = new_mask = self.obj.state.plot
        entln_mask = self.obj.state.gsd_mask

        lyl_temp = self.obj.state.all[self.obj.state.plot]
        entln_temp = self.obj.state.gsd[self.obj.state.gsd_mask]

        has_entln = not entln_temp[entln_mask].empty
        logger.info(f"{lyl_temp.head()}")
        logger.info(f"{entln_temp.head()}")
        if num == 0:
            x_values = [pt[0] for pt in self.clicks]  # extract x (time in seconds)
            min_x = min(x_values)
            max_x = max(x_values)
            logger.info(f"min_x: {min_x}, max_x: {max_x}")

            lyl_mask = (lyl_temp['seconds'] > min_x) & (lyl_temp['seconds'] < max_x)
            if has_entln:
                entln_mask = (entln_temp['utc_sec'] > min_x) & (entln_temp['utc_sec'] < max_x)
        elif num == 1:
            x_values = [pt[0] for pt in self.clicks]  
            min_x = min(x_values)
            max_x = max(x_values)

            lyl_mask = (lyl_temp['lon'] > min_x) & (lyl_temp["lon"] < max_x)
            if has_entln:
                entln_mask = (entln_temp['lon'] > min_x) & (entln_temp['lon'] < max_x)
        elif num == 3:
            polygon = Polygon(self.clicks)
            lon = lyl_temp['lon'].to_numpy()
            lat = lyl_temp['lat'].to_numpy()

            lyl_mask = contains(polygon, lon, lat) 
            if has_entln:
                entln_lon = entln_temp['lon'].to_numpy()
                entln_lat = entln_temp['lat'].to_numpy()
                entln_mask = contains(polygon, entln_lon, entln_lat)
        elif num == 4:
            y_values = [pt[1] for pt in self.clicks]
            min_y = min(y_values)
            max_y = max(y_values)

            lyl_mask = (lyl_temp['lat'] > min_y) & (lyl_temp['lat'] < max_y)
            if has_entln:
                entln_mask = (entln_temp['lat'] > min_y) & (entln_temp['lat'] < max_y)

        logger.info("Mask created.")

        temp_mask = np.zeros(len(self.obj.state.all), dtype=bool)
        temp_mask[lyl_temp.index] = (lyl_mask ^ self.remove)
        new_mask = temp_mask 

        temp_mask = np.zeros(len(self.obj.state.gsd), dtype=bool)
        temp_mask[entln_temp.index] = (entln_mask ^ self.remove)

        if update:
            # self.inc_mask = np.zeros(new_mask.sum(), dtype=bool)
            self.obj.state.update(plot=new_mask)

            # Needs to be done before update so we get a state change with an update gsd_mask
            if len(self.obj.state.gsd[temp_mask]) > 0:
                sym = 'triangle_up'
                colors = np.stack(self.obj.state.gsd[temp_mask]['colors'].to_numpy())
                positions = np.column_stack([self.obj.state.gsd[temp_mask]['utc_sec'].to_numpy(dtype=np.float32),self.obj.state.gsd[temp_mask]['alt'].to_numpy(dtype=np.float32)])
                self.ui.gs0.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

                positions = self.obj.state.gsd[temp_mask][['lon', 'alt']].to_numpy().astype(np.float32)
                self.ui.gs1.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

                positions = self.obj.state.gsd[temp_mask][['lon', 'lat']].to_numpy().astype(np.float32)
                self.ui.gs3.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)

                positions = self.obj.state.gsd[temp_mask][['alt', 'lat']].to_numpy().astype(np.float32)
                self.ui.gs4.set_data(pos=positions, face_color=colors, edge_color=colors, size=2, symbol=sym)
            else:
                self.ui.gs0.set_data(np.empty((0, 2)))
                self.ui.gs1.set_data(np.empty((0, 2)))
                self.ui.gs3.set_data(np.empty((0, 2)))
                self.ui.gs4.set_data(np.empty((0, 2)))
            self.obj.state.__dict__['gsd_mask'] = temp_mask
        else:
            return lyl_mask