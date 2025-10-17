from shapely import Polygon, contains_xy
import numpy as np
import pandas as pd

class PolygonFilter(): 
    def __init__(self, state=None, ui=None, remove=False, status_callback=None):
        self.state = state
        self.ui = ui
        self.remove = remove
        self.prev_ax = None
        self.clicks = []
        self.view_index = None

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

                return False
            elif event.button == 2: # Right click
                print("Right click detected")
                if len(self.clicks) > 1:
                    dots.set_data(np.array(self.clicks), face_color='green', size=5)
                    lines.set_data(pos=self.clicks + [self.clicks[0]], color=[[0, 1, 0, 1]] * (len(self.clicks) + 1))

                    # Store the current axis index for use in polygon logic
                    self.prev_ax = view_index

                    self.prev_ax = None

                    print(f"self.remove is currently {self.remove}")
                    self.polygon(self.view_index)

                    self.clear_polygon_visuals(view, dots, lines)

        event.handled = True


    def clear_polygon_visuals(self, view, dots, lines):
        view.scene.children.remove(dots)
        dots.parent = None
        lines.set_data(np.empty((0, 2)), color='red')
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


    def polygon(self, num):
        new_mask = self.state.plot
        if num == 0:
            x_values = [pt[0] for pt in self.clicks]  # extract x (time in seconds)
            min_x = min(x_values)
            max_x = max(x_values)

            start_of_day = self.state.all['datetime'].iloc[0].normalize()
            min_x = start_of_day + pd.to_timedelta(min_x, unit='s')
            max_x = start_of_day + pd.to_timedelta(max_x, unit='s')

            print(min_x)
            print(max_x)

            new_mask = self.state.plot & (((self.state.all['datetime'] > min_x) & (self.state.all['datetime'] < max_x)) ^ self.remove)
        if num == 1:
            x_values = [pt[0] for pt in self.clicks]  
            min_x = min(x_values)
            max_x = max(x_values)

            new_mask = self.state.plot & (((self.state.all['lon'] > min_x) & (self.state.all["lon"] < max_x)) ^ self.remove)    
        elif num == 3:
            polygon = Polygon(self.clicks)
            lon = self.state.all['lon'].to_numpy()
            lat = self.state.all['lat'].to_numpy()

            mask = contains_xy(polygon, lon, lat)

            new_mask = self.state.plot & (mask ^ self.remove)
        elif num == 4:
            y_values = [pt[1] for pt in self.clicks]
            min_y = min(y_values)
            max_y = max(y_values)

            new_mask = self.state.plot & (((self.state.all['lat'] > min_y) & (self.state.all['lat'] < max_y)) ^ self.remove)

        
        self.state.update(plot=new_mask)