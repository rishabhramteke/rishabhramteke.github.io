"""Printable models for the 'Printable things' project page. All sizes in millimetres.
Run:  <python with trimesh + manifold3d + shapely> tools/models/make_models.py   -> assets/models/*.stl
The mountain keycaps read tools/models/terrain/*.npy, made by fetch_terrain.py from the Copernicus GLO-30 DEM.
None of these has been test-printed yet; adjust the clearances below for your printer."""
import os, numpy as np, trimesh, manifold3d as mf
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

OUT = 'assets/models'
TERRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'terrain')

# ---------- small helpers ----------
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

def to_mf(t):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))

def to_tm(m):
    x = m.to_mesh(); t = trimesh.Trimesh(np.asarray(x.vert_properties)[:, :3], np.asarray(x.tri_verts))
    t.merge_vertices(digits_vertex=5); t.update_faces(t.nondegenerate_faces()); t.remove_unreferenced_vertices()   # weld float32 slivers
    return t

def rounded_slab(poly, T, r=1.2):
    """Extrude a 2D outline to height T with the top edges rounded over (radius r) and a flat bottom for the print
    bed: the Minkowski sum of the outline shrunk by r with a half-sphere of radius r. The outline is split into
    triangles, each triangle's sum is an exact convex hull, and the union of those is the sum for the whole outline."""
    ball = trimesh.creation.icosphere(subdivisions=2, radius=r)
    half = ball.intersection(trimesh.creation.box([3 * r, 3 * r, 1.5 * r], transform=trimesh.transformations.translation_matrix([0, 0, 0.75 * r])))
    v2, f2 = trimesh.creation.triangulate_polygon(poly.buffer(-r, resolution=16))
    pieces = []
    for tri in f2:
        p = v2[tri]; prism = np.vstack([np.c_[p, np.zeros(3)], np.c_[p, np.full(3, T - r)]])
        pieces.append(mf.Manifold.hull_points((prism[:, None, :] + half.vertices[None, :, :]).reshape(-1, 3)))
    return to_tm(mf.Manifold.batch_boolean(pieces, mf.OpType.Add))

# ---------- keycaps ----------
def keycap(top_art=None, H=7.6):
    """1u keycap with a Cherry MX cross stem, uniform (DSA-like) height, flat top."""
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

def heightfield(Z, half_w, zb):
    """Closed solid under a height grid Z (row 0 = north = +y), spanning +-half_w in x and y, floor at zb."""
    n = Z.shape[0]; s = np.linspace(-half_w, half_w, n)
    X, Y = np.meshgrid(s, s[::-1])
    top = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = [np.c_[a, c, b], np.c_[b, c, d]]
    ring = np.r_[idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]]       # perimeter, clockwise from above
    bot = top[ring].copy(); bot[:, 2] = zb
    nb = len(ring); bi = n * n + np.arange(nb); centre = n * n + nb
    verts = np.vstack([top, bot, [[0, 0, zb]]])
    nxt = np.roll(np.arange(nb), -1)
    faces += [np.c_[ring, bi[nxt], ring[nxt]], np.c_[ring, bi, bi[nxt]],            # walls
              np.c_[bi, np.full(nb, centre), bi[nxt]]]                              # floor
    m = trimesh.Trimesh(verts, np.vstack(faces), process=True)
    trimesh.repair.fix_normals(m)
    return m

MOUNTAINS = {   # name: (half-width of the terrain square in km, vertical exaggeration); must match fetch_terrain.py
    'fuji': (7.0, 2.0),
    'matterhorn': (2.6, 1.0),
}

def mountain_keycap(name, H=6.6, plinth=0.3, fade=1.8):
    """Keycap whose top is the real terrain around a summit, cut to the keycap's rounded-square top.
    Horizontal scale: the terrain square fills the 12.7 mm top. Vertical: true scale times the exaggeration, eased down
    to a thin plinth over the last `fade` mm so the mountain rises out of the key instead of ending in cliffs."""
    half_km, exag = MOUNTAINS[name]
    Z = np.load(f'{TERRAIN}/{name}.npy').astype(float)
    half_w = 12.7 / 2; mm_per_m = half_w / (half_km * 1000) * exag
    n = Z.shape[0]; s = np.linspace(-half_w, half_w, n); X, Y = np.meshgrid(s, s[::-1])
    top = Polygon(rounded_square(12.7, 1.2))
    edge = np.array([top.exterior.distance(Point(x, y)) if top.contains(Point(x, y)) else 0.0 for x, y in zip(X.ravel(), Y.ravel())]).reshape(n, n)
    base = Z[edge > 0].min()
    u = np.clip(edge / fade, 0, 1); weight = u * u * (3 - 2 * u)                  # smoothstep: the relief fades to the rim
    Zmm = H + plinth + np.clip(Z - base, 0, None) * mm_per_m * weight
    relief = heightfield(Zmm, half_w, H - 0.4).intersection(extrude(Polygon(rounded_square(12.7, 1.2)), 30, H - 0.5))
    return keycap(None, H).union(relief), (Z.max() - base, Zmm.max())

# ---------- towel hook ----------
def towel_hook():
    """Swing-out towel hook, printed in place, lying flat. In use the wall plate is vertical and the hinge axis is
    vertical too: the arm lies flat along the wall when folded and swings out 90 degrees. A hanging towel pulls along
    the hinge axis, so it cannot fold the arm shut and no stop is needed. Print coordinates: bed = wall plane,
    +y = up in use. The arm's knuckle turns on a pin with 0.4 mm clearance all round. All top edges are rounded."""
    T, CL, R_K, AX = 4.0, 0.4, 4.0, 15.0     # thickness, hinge clearance, knuckle radius, hinge axis x (axis at z = 4)
    def knuckle(y0, y1, r, c=0.6):
        """Cylinder along y around the hinge axis, with chamfered ends."""
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False); ring = lambda y, rr: np.c_[AX + rr * np.cos(t), np.full(64, y), 4.0 + rr * np.sin(t)]
        return trimesh.convex.convex_hull(np.vstack([ring(y0, r - c), ring(y0 + c, r), ring(y1 - c, r), ring(y1, r - c)]))
    def rod(y0, y1, r):
        c = trimesh.creation.cylinder(radius=r, height=y1 - y0, sections=48)
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])); c.apply_translation([AX, (y0 + y1) / 2, 4.0]); return c

    # wall plate: a slim pill with two lobes that carry the fixed knuckles, filleted where they meet
    pill = LineString([(0, -19), (0, 19)]).buffer(10, resolution=24)
    lobes = [Polygon([(5, y0), (AX, y0), (AX, y1), (5, y1)]) for y0, y1 in ((9.4, 16.6), (-16.6, -9.4))]
    outline = unary_union([pill] + lobes).buffer(3, resolution=16).buffer(-3, resolution=16)
    plate = rounded_slab(outline, T, 1.2)
    for y in (-21, 21):                    # two 4 mm screw holes with countersinks, one above the other
        hole = trimesh.creation.cylinder(radius=2.1, height=T * 3, sections=40); hole.apply_translation([0, y, T / 2])
        sink = trimesh.creation.cone(radius=4.2, height=4.2, sections=40)
        sink.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])); sink.apply_translation([0, y, T + 2.1])
        plate = plate.difference(hole).difference(sink)
    base = plate
    for part in (knuckle(-17, -9, R_K), knuckle(9, 17, R_K), rod(-17, 17, 1.6)):   # the pin runs through all three knuckles
        base = base.union(part)

    # arm: a J that tapers from 12 mm at the hinge to 7 mm and ends in a round knob; the curl rises up in use
    Ls, R, sweep = 46.0, 10.0, np.radians(105)
    s_line = np.linspace(0, Ls, 40); s_arc = np.linspace(0, sweep, 40)[1:]
    P = np.vstack([np.c_[AX + s_line, np.zeros_like(s_line)],
                   np.c_[AX + Ls + R * np.sin(s_arc), R - R * np.cos(s_arc)]])
    tang = np.gradient(P, axis=0); tang /= np.linalg.norm(tang, axis=1)[:, None]; nrm = np.c_[-tang[:, 1], tang[:, 0]]
    w = np.linspace(6.0, 3.5, len(P))[:, None]                                    # half-width
    body = Polygon(np.vstack([P + nrm * w, (P - nrm * w)[::-1]])).buffer(0)
    knob = Point(P[-1]).buffer(4.2, resolution=24)
    arm_outline = unary_union([body, knob]).buffer(1.5, resolution=16).buffer(-1.5, resolution=16)
    arm = rounded_slab(arm_outline, T, 1.2).union(knuckle(-9 + CL, 9 - CL, R_K)).difference(rod(-9.5, 9.5, 1.6 + CL))
    return base, arm

def swing(arm, deg):
    """The arm turned about the hinge axis; negative angles swing it out of the wall."""
    return arm.copy().apply_transform(trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0], point=[15.0, 0, 4.0]))

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name in MOUNTAINS:
        m, (relief_m, top_mm) = mountain_keycap(name); m.export(f'{OUT}/keycap-{name}.stl')
        print(f'keycap-{name}', 'watertight', m.is_watertight, 'relief m', round(relief_m), 'height mm', round(top_mm, 2), 'bounds', np.round(m.bounds, 2).tolist())
    base, arm = towel_hook()
    hook = trimesh.util.concatenate([base, arm]); hook.export(f'{OUT}/folding-towel-hook.stl')
    print('folding-towel-hook', 'parts watertight', base.is_watertight, arm.is_watertight, 'bounds', np.round(hook.bounds, 1).tolist())
    for deg in (0, -30, -60, -90):         # the arm must clear the base all the way out
        x = base.intersection(swing(arm, deg)); print(f'  overlap with the base at {deg} deg: {round(x.volume if len(x.faces) else 0.0, 4)} mm3')
