"""Fetch the terrain for the mountain keycaps from the Copernicus GLO-30 DEM (public, on AWS) and resample it onto a
square grid in metres centred on the summit. Writes tools/models/terrain/<name>.npy (N x N heights in metres, north up).
Run:  <python with rasterio + scipy> tools/models/fetch_terrain.py
Terrain: Copernicus GLO-30 DEM, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under
COPERNICUS by the European Union and ESA; all rights reserved."""
import os, math, numpy as np, rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates

URL = 'https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_COG_10_{t}_DEM.tif'
N = 96                                    # grid points per side (about 0.13 mm apart on a 12.7 mm keycap top)
MOUNTAINS = {                             # name: (summit lat, lon, DEM tile, half-width of the square in km)
    'fuji': (35.3606, 138.7274, 'N35_00_E138_00', 7.0),
    'matterhorn': (45.9763, 7.6586, 'N45_00_E007_00', 2.6),
}

def fetch(lat, lon, tile, half_km):
    m_lat = 111_320.0; m_lon = 111_320.0 * math.cos(math.radians(lat))
    pad = 1.1 * half_km * 1000
    with rasterio.open('/vsicurl/' + URL.format(t=tile)) as src:
        win = from_bounds(lon - pad / m_lon, lat - pad / m_lat, lon + pad / m_lon, lat + pad / m_lat, src.transform)
        z = src.read(1, window=win).astype('float64'); tr = src.window_transform(win)
    s = np.linspace(-half_km * 1000, half_km * 1000, N)
    E, Nn = np.meshgrid(s, s[::-1])                                   # row 0 = north
    col = (lon + E / m_lon - tr.c) / tr.a - 0.5; row = (lat + Nn / m_lat - tr.f) / tr.e - 0.5
    return map_coordinates(z, [row, col], order=1, mode='nearest').astype('float32')

if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), 'terrain'); os.makedirs(out, exist_ok=True)
    for name, args in MOUNTAINS.items():
        z = fetch(*args); np.save(f'{out}/{name}.npy', z)
        print(name, z.shape, 'heights', round(float(z.min())), 'to', round(float(z.max())), 'm')
