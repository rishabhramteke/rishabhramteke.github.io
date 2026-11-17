"""Printable models for the 'Printable things' project page. All sizes in millimetres.
Run:  <python with trimesh + manifold3d + shapely + scikit-image> tools/models/make_models.py   -> assets/models/*.stl
None of these has been test-printed yet; adjust the clearances below for your printer."""
import numpy as np, trimesh
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from skimage.measure import find_contours

OUT = 'assets/models'

def rounded_square(w, r=1.2, n=10):
    """Square of side w with corner radius r, as a 2D vertex ring."""
    pts = []
    h = w / 2 - r
    for cx, cy, a0 in [(h, h, 0), (-h, h, 90), (-h, -h, 180), (h, -h, 270)]:
        for k in range(n + 1):
            a = np.radians(a0 + 90 * k / n); pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)

def frustum(w0, w1, z0, z1, r=1.2):
    """Convex hull of two rounded squares: the body of a keycap."""
    a = rounded_square(w0, r); b = rounded_square(w1, r)
    v = np.vstack([np.c_[a, np.full(len(a), z0)], np.c_[b, np.full(len(b), z1)]])
    return trimesh.convex.convex_hull(v)

def extrude(poly, h, z0=0.0):
    parts = [g for g in getattr(poly, 'geoms', [poly]) if g.area > 1e-4]
    m = trimesh.boolean.union([trimesh.creation.extrude_polygon(g, h) for g in parts]) if len(parts) > 1 else trimesh.creation.extrude_polygon(parts[0], h)
    m.apply_translation([0, 0, z0]); return m

def keycap(top_art=None):
    """1u keycap with a Cherry MX cross stem, uniform (DSA-like) height, flat top."""
    H = 7.6
    outer = frustum(18.2, 12.7, 0.0, H)
    inner = frustum(15.6, 10.3, -0.01, H - 1.4, r=0.8)
    shell = outer.difference(inner)
    stem = trimesh.creation.cylinder(radius=2.75, height=H - 1.4, sections=48); stem.apply_translation([0, 0, (H - 1.4) / 2])
    cross = unary_union([Polygon([(-2.07, -0.65), (2.07, -0.65), (2.07, 0.65), (-2.07, 0.65)]),
                         Polygon([(-0.65, -2.07), (0.65, -2.07), (0.65, 2.07), (-0.65, 2.07)])])
    stem = stem.difference(extrude(cross, 4.2, -0.01))          # 4 mm deep cross for the switch
    cap = shell.union(stem)
    if top_art is not None:                                       # engrave 0.6 mm into the flat top
        cap = cap.difference(extrude(top_art, 0.61, H - 0.6))
    return cap

def hexagon_art():
    """An H3-style hexagon with its ring of six neighbours, as engraved lines."""
    def hexring(cx, cy, r):
        return [(cx + r * np.cos(np.radians(60 * k + 30)), cy + r * np.sin(np.radians(60 * k + 30))) for k in range(6)]
    lines = [LineString(hexring(0, 0, 2.4) + [hexring(0, 0, 2.4)[0]])]
    for k in range(6):
        a = np.radians(60 * k); cx, cy = 4.16 * np.cos(a), 4.16 * np.sin(a)
        ring = hexring(cx, cy, 2.4); lines.append(LineString(ring + [ring[0]]))
    clip = Polygon(rounded_square(11.7, 1.0))
    return unary_union([l.buffer(0.28, join_style=2) for l in lines]).intersection(clip)

def contour_art():
    """Contour lines of a small made-up summit, like the homepage map, as engraved lines."""
    x = np.linspace(-6, 6, 241); X, Y = np.meshgrid(x, x)
    Z = np.exp(-((X - 1.2) ** 2 + (Y + 0.6) ** 2) / 9) + 0.55 * np.exp(-((X + 2.6) ** 2 + (Y - 2.2) ** 2) / 4) + 0.04 * np.sin(1.7 * X + 0.8 * Y)
    lines = []
    for lev in np.linspace(0.12, 1.0, 8):
        for c in find_contours(Z, lev):
            pts = [(x[0] + (p[1] / 240) * 12, x[0] + (p[0] / 240) * 12) for p in c]
            if len(pts) > 8: lines.append(LineString(pts).simplify(0.05))
    clip = Polygon(rounded_square(11.7, 1.0))
    return unary_union([l.buffer(0.22, resolution=4) for l in lines]).intersection(clip)

def towel_hook():
    """Swing-out towel hook, printed in place, lying flat. In use the wall plate is vertical and the hinge axis is
    vertical too: the arm lies flat along the wall when folded and swings out 90 degrees. A hanging towel pulls along
    the hinge axis, so it cannot fold the arm shut and no stop is needed. Print coordinates: bed = wall plane,
    +y = up in use. The arm's knuckle turns on a pin with 0.4 mm clearance all round."""
    T, CL = 4.0, 0.4                       # plate and arm thickness, hinge clearance
    plate = trimesh.creation.box([22, 56, T]); plate.apply_translation([0, 0, T / 2])
    for y in (-18, 18):                    # two 4 mm screw holes with countersinks, one above the other
        hole = trimesh.creation.cylinder(radius=2.1, height=T * 3, sections=40); hole.apply_translation([0, y, T / 2])
        sink = trimesh.creation.cone(radius=4.2, height=4.2, sections=40)
        sink.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])); sink.apply_translation([0, y, T + 2.1])
        plate = plate.difference(hole).difference(sink)
    ax = 11 + 4.0                          # hinge axis beside the plate, parallel to y (vertical in use)
    def knuckle(y0, y1, r):
        c = trimesh.creation.cylinder(radius=r, height=y1 - y0, sections=56)
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])); c.apply_translation([ax, (y0 + y1) / 2, 4.0]); return c
    lugs = []
    for yc in (-13.0, 13.0):               # plate-to-knuckle bridges
        lug = trimesh.creation.box([6, 8, T]); lug.apply_translation([11 + 1.5, yc, T / 2]); lugs.append(lug)
    base = plate
    for part in lugs + [knuckle(-17, -9, 4.0), knuckle(9, 17, 4.0), knuckle(-17, 17, 1.6)]:   # pin runs through all three knuckles
        base = base.union(part)
    arm_knuckle = knuckle(-9 + CL, 9 - CL, 4.0).difference(knuckle(-9.5, 9.5, 1.6 + CL))
    bar = trimesh.creation.box([64, 14, T]); bar.apply_translation([ax + 4.5 + 32, 0, T / 2])          # the arm, 64 mm
    neck = trimesh.creation.box([6, 14, T]); neck.apply_translation([ax + 3.0, 0, T / 2])               # knuckle to arm
    lip = trimesh.creation.box([T + 1, 12, T]); lip.apply_translation([ax + 4.5 + 64 - (T + 1) / 2, 7 + 6, T / 2])  # tip rises 12 mm up in use
    arm = arm_knuckle.union(neck.difference(knuckle(-9.5, 9.5, 1.6 + CL))).union(bar).union(lip)
    return base, arm

if __name__ == '__main__':
    import os; os.makedirs(OUT, exist_ok=True)
    for name, art in [('keycap-h3-hexagon', hexagon_art()), ('keycap-contours', contour_art())]:
        m = keycap(art); m.export(f'{OUT}/{name}.stl'); print(name, 'watertight', m.is_watertight, 'volume mm3', round(m.volume, 1), 'bounds', np.round(m.bounds, 2).tolist())
    base, arm = towel_hook()
    hook = trimesh.util.concatenate([base, arm]); hook.export(f'{OUT}/folding-towel-hook.stl')
    gap = base.intersection(arm).volume if base.intersection(arm).volume else 0.0
    print('folding-towel-hook', 'parts watertight', base.is_watertight, arm.is_watertight, 'parts overlap mm3', round(gap, 4), 'bounds', np.round(hook.bounds, 1).tolist())
