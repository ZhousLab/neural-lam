import cartopy
import numpy as np

wandb_project = "neural-lam"

seconds_in_year = 365*24*60*60 # Assuming no leap years in dataset (2024 is next)

# Log prediction error for these lead times
val_step_log_errors = np.array([1, 3, 5])

# Variable names
param_names = [
    'total_column_water_vapor'
]

param_names_short = [
    'tcwv'
]
param_units = [
    'kg/m2'
]

# Projection and grid
# TODO Do not hard code this, make part of static dataset files
grid_shape = (81, 381) # (y, x)

lambert_proj_params = {
     'a': 6367470,
     'b': 6367470,
     'lat_0': 63.3,
     'lat_1': 63.3,
     'lat_2': 63.3,
     'lon_0': 15.0,
     'proj': 'lcc'
 }

grid_limits = [ # In projection
    -1059506.5523409774, # min x
    1310493.4476590226, # max x
    -1331732.4471934352, # min y
    1338267.5528065648, # max y
]

# Create projection
lambert_proj = cartopy.crs.LambertConformal(
        central_longitude=lambert_proj_params['lon_0'],
        central_latitude=lambert_proj_params['lat_0'],
        standard_parallels=(lambert_proj_params['lat_1'],
        lambert_proj_params['lat_2']))
