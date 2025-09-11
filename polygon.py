from shapely import Polygon, vectorized
import numpy as np

def polygon(self, num):
    if not hasattr(self, "masks"):
        self.masks = []

    if num == 0:
        print(self.clicks)
        min_x = min(self.clicks).replace(tzinfo=None)
        max_x = max(self.clicks).replace(tzinfo=None)

        mask = (self.lyl['datetime'] > min_x) & (self.lyl['datetime'] < max_x)
    if num == 1:
        x_values = [pt[0] for pt in self.clicks]  
        min_x = min(x_values)
        max_x = max(x_values)

        mask = (self.lyl['lon'] > min_x) & (self.lyl["lon"] < max_x)
    elif num == 3:
        polygon = Polygon(self.clicks)
        lon = self.lyl['lon'].to_numpy()
        lat = self.lyl['lat'].to_numpy()

        mask = vectorized.contains(polygon, lon, lat)
    elif num == 4:
        y_values = [pt[1] for pt in self.clicks]
        min_y = min(y_values)
        max_y = max(y_values)

        mask = (self.lyl['lat'] > min_y) & (self.lyl['lat'] < max_y)

    if self.remove:
        mask = ~mask

    self.lyl = self.lyl[mask]
    # and then when we call plots/etc we can check to see if the lyl is not none else we can send it in or sum ting else.

    if len(mask) < len(self.og):
        # Pad mask with false for later undo operations
        mask = np.pad(mask, (0, len(self.og) - len(mask)), constant_values=False)
    if not self.lyl.empty:
        self.masks.append(mask)
        print(self.masks)
        self.do_plot()
    else:
        self.update_status("Polygon failed")

def undo_filter(self):
    if self.masks:
        mask = self.masks.pop()
        self.popped_masks.append(mask)
        self.apply_filters()
    else:
        self.update_status("No filter to undo")
    
def redo_filter(self):
    if self.popped_masks:
        redo_mask = self.popped_masks.pop()
        self.masks.append(redo_mask)
        self.apply_filters()
    else:
        self.update_status("No filter to redo")

def apply_filters(self):
    if not self.masks:
        self.lyl = self.og.copy()
    else:
        stacked_masks = np.stack(self.masks)
        # Make a single mask to apply all at once
        combined_masks = np.logical_and.reduce(stacked_masks, axis=0)
        self.lyl = self.og[combined_masks]
    
    self.do_update(self.lyl)