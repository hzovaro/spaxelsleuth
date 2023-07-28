from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import numpy as np

# This example from https://docs.astropy.org/en/stable/visualization/wcsaxes/generic_transforms.html 

plt.close("all")

as_per_px = 0.5
x0_px = 50
y0_px = 50
pa_deg = 30

# Set up an affine transformation
transform = Affine2D()
transform.scale(as_per_px)
transform.translate(- x0_px * as_per_px, - y0_px * as_per_px)
transform.rotate(-np.deg2rad(pa_deg))  # radians

# Set up metadata dictionary
coord_meta = {}
coord_meta['name'] = 'x', 'y'
coord_meta['type'] = 'scalar', 'scalar'
coord_meta['wrap'] = None, None
coord_meta['unit'] = u.arcsec, u.arcsec
coord_meta['format_unit'] = u.arcsec, u.arcsec

fig = plt.figure()
ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], aspect='equal',
             transform=transform, coord_meta=coord_meta)
fig.add_axes(ax)
# ax.set_xlim(-0.5, 499.5)
# ax.set_ylim(-0.5, 399.5)
ax.grid()
# ax.coords['y'].set_axislabel('Longitude')
# ax.coords['x'].set_axislabel('Latitude')

data = np.random.uniform(size=(100, 100))
ax.imshow(data)