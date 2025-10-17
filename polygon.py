from shapely.geometry import Polygon
from shapely import contains_xy
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
    def __init__(self, state=None, ui=None, remove=False, status_callback=None):
        self.state = state
        self.ui = ui
        self.remove = remove
        self.prev_ax = None
        self.clicks = []
        self.view_index = None
        self.inc_mask = np.zeros(len(state.all), dtype=bool)

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
        print(f"{pos}")
        transform = lines.transforms.get_transform(map_from="canvas", map_to="visual")
        x, y = transform.map(pos)[:2]

        print(f"Clicked on view {view_index}: x={x}, y={y}")
        if self.prev_ax == None or self.prev_ax == view_index:
            if event.button == 1: # Left click
                self.prev_ax = view_index
                self.clicks.append((x, y))
                if dots.parent is None:
                    view.add(dots)
                dots.set_data(np.array(self.clicks), face_color='red', size=5)
                lines.set_data(self.clicks)

                if len(self.clicks) >= 3 and self.view_index == 3:
                    # Get the last triangle formed from (first, prev, current)
                    # triangle = Polygon([self.clicks[0], self.clicks[-2], self.clicks[-1]])
                    # lon = temp[self.state.plot]['lon'].to_numpy()
                    # lat = temp[self.state.plot]['lat'].to_numpy()
                    # triangle_mask = contains_xy(triangle, lon, lat)

                    # if not hasattr(self, 'inc_mask') or len(self.inc_mask) != len(triangle_mask):
                    #     self.inc_mask = np.zeros(len(triangle_mask), dtype=bool)

                    # print(f"inc mask size: {len(self.inc_mask)}, triangle_mask size: {len(triangle_mask)}")
                    
                    self.inc_mask = self.polygon(view_index, False)

                    self.state.replot()

            elif event.button == 2: # Right click
                print("Right click detected")
                if len(self.clicks) > 1:
                    dots.set_data(np.array(self.clicks), face_color='green', size=5)
                    lines.set_data(pos=self.clicks + [self.clicks[0]], color=[[0, 1, 0, 1]] * (len(self.clicks) + 1))

                    # Store the current axis index for use in polygon logic
                    self.prev_ax = view_index

                    self.polygon(self.view_index, True)

                    self.clear_polygon_visuals(view, dots, lines)

        event.handled = True


    def clear_polygon_visuals(self, view, dots, lines):
        view.scene.children.remove(dots)
        dots.parent = None
        lines.set_data(np.empty((0, 2)), color='red')
        self.prev_ax = None
        self.inc_mask = np.zeros(len(self.state.all), dtype=bool)
        self.clicks.clear()

    def prompt_polygon_action(ui):
        # This only uses terminal, but it will suffice until we get a UI for this
        print("\nPolygon completed. Choose action:")
        print("1: Keep")
        print("2: Remove")
        print("3: Zoom")
        print("4: Cancel")
        while True:
            try:
                choice = int(input("Enter choice [1-4]: "))
                if choice in (1, 2, 3, 4):
                    return choice
            except ValueError:
                pass
            print("Invalid input, try again.")


    def polygon(self, num, update = False):
        new_mask = self.state.plot
        temp = self.state.all[self.state.plot]
        if num == 0:
            x_values = [pt[0] for pt in self.clicks]  # extract x (time in seconds)
            min_x = min(x_values)
            max_x = max(x_values)

            start_of_day = temp['datetime'].iloc[0].normalize()
            min_x = start_of_day + pd.to_timedelta(min_x, unit='s')
            max_x = start_of_day + pd.to_timedelta(max_x, unit='s')

            print(min_x)
            print(max_x)

            new_mask = self.state.plot & (((temp['datetime'] > min_x) & (temp['datetime'] < max_x)) ^ self.remove)
        if num == 1:
            x_values = [pt[0] for pt in self.clicks]  
            min_x = min(x_values)
            max_x = max(x_values)

            new_mask = self.state.plot & (((temp['lon'] > min_x) & (temp["lon"] < max_x)) ^ self.remove)    
        elif num == 3:
            polygon = Polygon(self.clicks)
            lon = temp['lon'].to_numpy()
            lat = temp['lat'].to_numpy()

            mask = contains_xy(polygon, lon, lat) 
            
            temp_mask = np.zeros(len(self.state.all), dtype=bool)
            temp_mask[temp.index] = (mask ^ self.remove)
            new_mask = temp_mask 

            logger.info(f"len(all): {len(self.state.all)}")
            logger.info(f"len(plot): {len(self.state.plot)}")
        elif num == 4:
            y_values = [pt[1] for pt in self.clicks]
            min_y = min(y_values)
            max_y = max(y_values)

            new_mask = self.state.plot & (((temp['lat'] > min_y) & (temp['lat'] < max_y)) ^ self.remove)

        if update:
            logger.info(f"update len(all): {len(self.state.all)}")
            logger.info(f"update len(plot): {len(self.state.plot)}")
            logger.info(f"update len(new_mask): {len(new_mask)}")
            self.inc_mask = np.zeros(new_mask.sum(), dtype=bool)
            self.state.update(plot=new_mask)
            logger.info(f"post update len(all): {len(self.state.all)}")
            logger.info(f"post update len(plot): {len(self.state.plot)}")
        else:
            return mask