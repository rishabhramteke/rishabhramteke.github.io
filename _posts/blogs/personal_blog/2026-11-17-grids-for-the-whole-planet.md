---
layout: post
title: "Gridding the planet: hexagons, squares and curves"
description: "An honest comparison of the main ways to cut the Earth into indexed cells: lat/lon boxes, web map tiles, geohash, S2, H3, HEALPix and a few others."
image_show_on_page: false
category: blog
permalink: geo-grids
---
Sooner or later a lot of map data ends up as a column of cell ids. At work I use H3, so hexagons are the grid I know best. But H3 is one answer among several, and the others made different trade-offs for good reasons. This post lays them side by side: what each one is, where it bends, and when I would reach for it.

## Why grid the planet at all

Points and polygons are awkward to compute with. Matching ten million GPS points to fifty thousand polygons needs a spatial index and many geometry tests. Turn each point into a cell id and it becomes an ordinary key, and the usual tools work again. A grid gives you five things:

- **Joins.** Two datasets with nothing in common but location can be joined on a cell id.
- **Aggregation.** Count or average per cell, and the map no longer depends on postcode or city boundaries.
- **Neighbours.** "What is around this cell?" becomes a lookup, for smoothing, flow or model features.
- **Multiple resolutions.** Most systems nest cells, so fine results roll up to coarse ones.
- **Stable ids.** The same place gets the same id next year, on another machine, in another team.

There is an ML reason too. Random train/test splits leak, because nearby samples end up on both sides. Grouping samples by a coarse cell gives a cheap spatial block split.

## Latitude and longitude boxes

The obvious grid is a box of fixed size in degrees, say 1° by 1°. It is easy to compute, and a lot of climate and weather data ships this way. The problem is that a degree of longitude gets shorter as you go north. On a sphere with the Earth's mean radius:

| Latitude band | Area of a 1° × 1° box |
|---|---|
| 0° to 1° | 12,364 km² |
| 52° to 53° (Amsterdam) | 7,527 km² |
| 80° to 81° | 2,041 km² |
| 89° to 90° | 108 km² |

So a count per box in Amsterdam is not comparable with one on the equator until you correct for area. At the poles the top row of boxes turns into 360 slivers that all meet at one point, which is not a useful idea of "neighbour".

## Squares and quadtrees

The next family keeps square cells but splits each one into four children, again and again. That gives an exact hierarchy: every child sits wholly inside one parent, and four children cover their parent exactly.

### Web map tiles and quadkeys

Almost every web map is drawn from square tiles in Web Mercator. At zoom *z* the world is 2<sup>z</sup> by 2<sup>z</sup> tiles, and the projection is cut off at about 85.05° north and south so that the world fits in a square. The tile over Dam Square at zoom 12 is x = 2103, y = 1346.

Bing Maps folded those two numbers into one string, the quadkey: one digit per zoom level, each digit from 0 to 3 saying which quarter you went into. The Amsterdam tile is `120202110131`. Drop the last digit and you have its parent.

The price is Mercator itself. Tiles are equal in the projection, not on the ground. A tile's ground width shrinks with the cosine of latitude, so in Amsterdam a tile covers about 37% of the ground area of a tile at the same zoom on the equator (cos² 52.4° ≈ 0.37).

### Morton codes, or the Z-order curve

For a long time I half-remembered this one as "merone" codes. The name is **Morton**, after G. M. Morton, who described it in a 1966 IBM report on a geodetic database. It is also called the **Z-order** curve. The idea is simple: write x and y in binary and interleave their bits.

With x = 5 (`101`) and y = 3 (`011`), taking a y bit then an x bit at each position from the top, you get `01 10 11`, which is 27. Read those pairs as base-4 digits and you have the quadkey idea exactly: a quadkey is a Morton code written in base 4.

```python
def morton(x, y, bits=16):
    z = 0
    for i in range(bits):
        z |= ((x >> i) & 1) << (2 * i)      # x: even bits
        z |= ((y >> i) & 1) << (2 * i + 1)  # y: odd bits
    return z

def quadkey(x, y, zoom):
    digits = ""
    for i in range(zoom, 0, -1):
        mask = 1 << (i - 1)
        d = (1 if x & mask else 0) + (2 if y & mask else 0)
        digits += str(d)
    return digits

print(morton(5, 3))              # 27
print(quadkey(3, 5, 3))          # 213, as in the Bing docs
print(quadkey(2103, 1346, 12))   # 120202110131, Dam Square
qk = quadkey(2103, 1346, 12)
print(int(qk, 4) == morton(2103, 1346))   # True
```

Visiting cells in Morton order traces a Z in each block of four, then a Z of Zs, and so on up.

<figure>
<svg viewBox="0 0 640 330" width="100%" style="max-width:100%;height:auto;display:block" role="img" aria-label="Two four by four grids. On the left a Z-order or Morton curve visits the sixteen cells in repeated Z shapes and makes long diagonal jumps between blocks. On the right a Hilbert curve visits the same sixteen cells and every step moves to an edge neighbour." font-family="'Barlow Condensed', sans-serif" font-weight="500">
<text x="160" y="44" text-anchor="middle" font-size="20" font-weight="600" fill="#5B1426" letter-spacing="1">Z-ORDER (MORTON)</text>
<line x1="40" y1="60" x2="280" y2="60" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="40" y1="60" x2="40" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="40" y1="120" x2="280" y2="120" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="100" y1="60" x2="100" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="40" y1="180" x2="280" y2="180" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="160" y1="60" x2="160" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="40" y1="240" x2="280" y2="240" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="220" y1="60" x2="220" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="40" y1="300" x2="280" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="280" y1="60" x2="280" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<polyline points="70,90 130,90 70,150 130,150 190,90 250,90 190,150 250,150 70,210 130,210 70,270 130,270 190,210 250,210 190,270 250,270" fill="none" stroke="#8A2A3D" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="70" cy="90" r="5" fill="#5B1426"/>
<text x="45" y="77" font-size="15" fill="#5A4549">0</text>
<circle cx="130" cy="90" r="3" fill="#5B1426"/>
<text x="105" y="77" font-size="15" fill="#5A4549">1</text>
<circle cx="70" cy="150" r="3" fill="#5B1426"/>
<text x="45" y="137" font-size="15" fill="#5A4549">2</text>
<circle cx="130" cy="150" r="3" fill="#5B1426"/>
<text x="105" y="137" font-size="15" fill="#5A4549">3</text>
<circle cx="190" cy="90" r="3" fill="#5B1426"/>
<text x="165" y="77" font-size="15" fill="#5A4549">4</text>
<circle cx="250" cy="90" r="3" fill="#5B1426"/>
<text x="225" y="77" font-size="15" fill="#5A4549">5</text>
<circle cx="190" cy="150" r="3" fill="#5B1426"/>
<text x="165" y="137" font-size="15" fill="#5A4549">6</text>
<circle cx="250" cy="150" r="3" fill="#5B1426"/>
<text x="225" y="137" font-size="15" fill="#5A4549">7</text>
<circle cx="70" cy="210" r="3" fill="#5B1426"/>
<text x="45" y="197" font-size="15" fill="#5A4549">8</text>
<circle cx="130" cy="210" r="3" fill="#5B1426"/>
<text x="105" y="197" font-size="15" fill="#5A4549">9</text>
<circle cx="70" cy="270" r="3" fill="#5B1426"/>
<text x="45" y="257" font-size="15" fill="#5A4549">10</text>
<circle cx="130" cy="270" r="3" fill="#5B1426"/>
<text x="105" y="257" font-size="15" fill="#5A4549">11</text>
<circle cx="190" cy="210" r="3" fill="#5B1426"/>
<text x="165" y="197" font-size="15" fill="#5A4549">12</text>
<circle cx="250" cy="210" r="3" fill="#5B1426"/>
<text x="225" y="197" font-size="15" fill="#5A4549">13</text>
<circle cx="190" cy="270" r="3" fill="#5B1426"/>
<text x="165" y="257" font-size="15" fill="#5A4549">14</text>
<circle cx="250" cy="270" r="3" fill="#5B1426"/>
<text x="225" y="257" font-size="15" fill="#5A4549">15</text>
<text x="480" y="44" text-anchor="middle" font-size="20" font-weight="600" fill="#5B1426" letter-spacing="1">HILBERT</text>
<line x1="360" y1="60" x2="600" y2="60" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="360" y1="60" x2="360" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="360" y1="120" x2="600" y2="120" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="420" y1="60" x2="420" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="360" y1="180" x2="600" y2="180" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="480" y1="60" x2="480" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="360" y1="240" x2="600" y2="240" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="540" y1="60" x2="540" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="360" y1="300" x2="600" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<line x1="600" y1="60" x2="600" y2="300" stroke="rgba(58,29,36,.28)" stroke-width="1.5"/>
<polyline points="390,90 450,90 450,150 390,150 390,210 390,270 450,270 450,210 510,210 510,270 570,270 570,210 570,150 510,150 510,90 570,90" fill="none" stroke="#8A2A3D" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="390" cy="90" r="5" fill="#5B1426"/>
<text x="365" y="77" font-size="15" fill="#5A4549">0</text>
<circle cx="450" cy="90" r="3" fill="#5B1426"/>
<text x="425" y="77" font-size="15" fill="#5A4549">1</text>
<circle cx="450" cy="150" r="3" fill="#5B1426"/>
<text x="425" y="137" font-size="15" fill="#5A4549">2</text>
<circle cx="390" cy="150" r="3" fill="#5B1426"/>
<text x="365" y="137" font-size="15" fill="#5A4549">3</text>
<circle cx="390" cy="210" r="3" fill="#5B1426"/>
<text x="365" y="197" font-size="15" fill="#5A4549">4</text>
<circle cx="390" cy="270" r="3" fill="#5B1426"/>
<text x="365" y="257" font-size="15" fill="#5A4549">5</text>
<circle cx="450" cy="270" r="3" fill="#5B1426"/>
<text x="425" y="257" font-size="15" fill="#5A4549">6</text>
<circle cx="450" cy="210" r="3" fill="#5B1426"/>
<text x="425" y="197" font-size="15" fill="#5A4549">7</text>
<circle cx="510" cy="210" r="3" fill="#5B1426"/>
<text x="485" y="197" font-size="15" fill="#5A4549">8</text>
<circle cx="510" cy="270" r="3" fill="#5B1426"/>
<text x="485" y="257" font-size="15" fill="#5A4549">9</text>
<circle cx="570" cy="270" r="3" fill="#5B1426"/>
<text x="545" y="257" font-size="15" fill="#5A4549">10</text>
<circle cx="570" cy="210" r="3" fill="#5B1426"/>
<text x="545" y="197" font-size="15" fill="#5A4549">11</text>
<circle cx="570" cy="150" r="3" fill="#5B1426"/>
<text x="545" y="137" font-size="15" fill="#5A4549">12</text>
<circle cx="510" cy="150" r="3" fill="#5B1426"/>
<text x="485" y="137" font-size="15" fill="#5A4549">13</text>
<circle cx="510" cy="90" r="3" fill="#5B1426"/>
<text x="485" y="77" font-size="15" fill="#5A4549">14</text>
<circle cx="570" cy="90" r="3" fill="#5B1426"/>
<text x="545" y="77" font-size="15" fill="#5A4549">15</text>
</svg>
<figcaption>Left: Morton (Z-order) visits a 4 × 4 grid in nested Z shapes and jumps between blocks, for example from cell 7 back to the far left for cell 8. Right: the Hilbert curve visits the same cells and every step goes to an edge neighbour.</figcaption>
</figure>

Morton order is cheap, just bit shifts, and nearby cells usually get nearby numbers, which is why it makes a good database key. But it jumps. The Hilbert curve, which David Hilbert described in 1891, covers the same cells with no jumps at all. It costs a little more to compute, and in return a run of consecutive ids is more likely to be one compact patch.

### Geohash

Geohash, published by Gustavo Niemeyer in 2008 and put in the public domain, is a Morton code on plain longitude and latitude. Halve the longitude range and write 1 if you are in the upper half, then do the same for latitude, and repeat. Every 5 bits become one character from a 32-character alphabet that skips a, i, l and o.

```python
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"  # no a, i, l, o

def geohash(lat, lng, length=7):
    lat_rng, lng_rng = [-90.0, 90.0], [-180.0, 180.0]
    code, bits, use_lng = "", 0, True
    for i in range(length * 5):
        rng, v = (lng_rng, lng) if use_lng else (lat_rng, lat)
        mid = (rng[0] + rng[1]) / 2
        bit = v >= mid
        rng[0 if bit else 1] = mid      # keep our half
        bits = (bits << 1) | bit
        if i % 5 == 4:                  # 5 bits per character
            code, bits = code + BASE32[bits], 0
        use_lng = not use_lng
    return code

print(geohash(57.64911, 10.40744, 11))  # u4pruydqqvj
print(geohash(52.3731, 4.8926))         # u173zq4, Dam Square
# two points 14 m apart, either side of Greenwich:
print(geohash(51.4779, -0.0001, 6))     # gcpuzg
print(geohash(51.4779, 0.0001, 6))      # u10hb5
```

The first line reproduces the usual worked example, a point near the tip of Jutland. Geohashes are popular because a prefix is a containing cell, and prefix search is something every database can do. Two things to know. First, the cells inherit the lat/lon problem: they get narrower towards the poles, and their shape alternates with length. A 5-character hash is about 4.9 km by 4.9 km at the equator, a 6-character one about 1.2 km by 0.6 km. Second, a shared prefix means two points are close, but two close points need not share a prefix. The two Greenwich points at the end are 14 m apart and share no characters at all, because the very first split puts them on opposite sides.

### S2

Google's S2 library avoids the pole problem by putting the Earth inside a cube. Each of the six faces is projected onto the sphere and cut as a quadtree, with 31 levels from 0 to 30; leaf cells are about a centimetre across. A non-linear transform evens out the sizes, so at a given level the largest cell is roughly twice the smallest: at level 13 they range from 0.76 to 1.59 km². Cell edges are geodesics.

On each face the cells are ordered along a Hilbert curve, and the six face curves join into one continuous loop around the sphere. The 64-bit id holds three bits for the face, two bits per level for the child, and a trailing 1 bit that marks where the level ends. All descendants of a cell fall in one contiguous range of ids, so a polygon can be described by a small "covering" of cells, and a point-in-polygon test turns into an integer range check. The one oddity is at the eight corners of the cube, where three cells meet instead of four.

## Hexagons: H3

H3 was built at Uber and open-sourced in 2018. It starts from an icosahedron, the 20-faced solid, oriented so that all twelve of its vertices land in the ocean, and lays a hexagonal grid on its faces. Resolution 0 has 122 cells. Each finer resolution has about seven times as many cells (aperture 7), through resolution 15, so there are 16 resolutions: from about 4.36 million km² per cell on average at resolution 0 to under 1 m² at resolution 15. Resolution 7, about 5.16 km² on average, is a handy size for city-scale work.

You cannot tile a sphere with hexagons alone, so every resolution has exactly 12 pentagons, centred on the icosahedron's vertices. A pentagon has five neighbours instead of six, and six children instead of seven. Because they sit in the ocean, most work on land never meets one, but any code that walks the grid has to handle them.

### Why hexagons

The appeal is the neighbours. A square has four neighbours across an edge and four more across a corner, at two different distances. A hexagon has six neighbours, all across an edge, all at the same distance from the centre. Rings of neighbours look like a disc rather than a square, and smoothing over a hex grid does not favour the diagonals.

<figure>
<svg viewBox="0 0 640 350" width="100%" style="max-width:100%;height:auto;display:block" role="img" aria-label="On the left a square cell surrounded by eight squares: four share an edge and sit at distance 1, four touch only at a corner and sit at distance 1.41. On the right a hexagon surrounded by six hexagons, all sharing an edge and all at the same distance." font-family="'Barlow Condensed', sans-serif" font-weight="500">
<text x="172.0" y="38" text-anchor="middle" font-size="20" font-weight="600" fill="#5B1426" letter-spacing="1">SQUARE</text>
<rect x="64" y="56" width="72" height="72" fill="none" stroke="#231518" stroke-width="1.5"/>
<rect x="64" y="128" width="72" height="72" fill="none" stroke="#231518" stroke-width="1.5"/>
<rect x="64" y="200" width="72" height="72" fill="none" stroke="#231518" stroke-width="1.5"/>
<rect x="136" y="56" width="72" height="72" fill="none" stroke="#231518" stroke-width="1.5"/>
<rect x="136" y="128" width="72" height="72" fill="#8A2A3D" stroke="#231518" stroke-width="1.5"/>
<rect x="136" y="200" width="72" height="72" fill="none" stroke="#231518" stroke-width="1.5"/>
<rect x="208" y="56" width="72" height="72" fill="none" stroke="#231518" stroke-width="1.5"/>
<rect x="208" y="128" width="72" height="72" fill="none" stroke="#231518" stroke-width="1.5"/>
<rect x="208" y="200" width="72" height="72" fill="none" stroke="#231518" stroke-width="1.5"/>
<line x1="172" y1="164" x2="100" y2="92" stroke="#5A4549" stroke-width="2" stroke-dasharray="5 4"/>
<circle cx="100" cy="92" r="4" fill="#5A4549"/>
<line x1="172" y1="164" x2="100" y2="164" stroke="#5B1426" stroke-width="2"/>
<circle cx="100" cy="164" r="4" fill="#5B1426"/>
<line x1="172" y1="164" x2="100" y2="236" stroke="#5A4549" stroke-width="2" stroke-dasharray="5 4"/>
<circle cx="100" cy="236" r="4" fill="#5A4549"/>
<line x1="172" y1="164" x2="172" y2="92" stroke="#5B1426" stroke-width="2"/>
<circle cx="172" cy="92" r="4" fill="#5B1426"/>
<line x1="172" y1="164" x2="172" y2="236" stroke="#5B1426" stroke-width="2"/>
<circle cx="172" cy="236" r="4" fill="#5B1426"/>
<line x1="172" y1="164" x2="244" y2="92" stroke="#5A4549" stroke-width="2" stroke-dasharray="5 4"/>
<circle cx="244" cy="92" r="4" fill="#5A4549"/>
<line x1="172" y1="164" x2="244" y2="164" stroke="#5B1426" stroke-width="2"/>
<circle cx="244" cy="164" r="4" fill="#5B1426"/>
<line x1="172" y1="164" x2="244" y2="236" stroke="#5A4549" stroke-width="2" stroke-dasharray="5 4"/>
<circle cx="244" cy="236" r="4" fill="#5A4549"/>
<circle cx="172" cy="164" r="4" fill="#EDEFEC"/>
<text x="172" y="302" text-anchor="middle" font-size="18" fill="#231518">4 edge neighbours at 1</text>
<text x="172" y="324" text-anchor="middle" font-size="18" fill="#5A4549">4 corner neighbours at 1.41</text>
<text x="470" y="38" text-anchor="middle" font-size="20" font-weight="600" fill="#5B1426" letter-spacing="1">HEXAGON</text>
<polygon points="512,164 491,200.4 449,200.4 428,164 449,127.6 491,127.6" fill="#8A2A3D" stroke="#231518" stroke-width="1.5"/>
<polygon points="575,200.4 554,236.7 512,236.7 491,200.4 512,164 554,164" fill="none" stroke="#231518" stroke-width="1.5"/>
<polygon points="512,236.7 491,273.1 449,273.1 428,236.7 449,200.4 491,200.4" fill="none" stroke="#231518" stroke-width="1.5"/>
<polygon points="449,200.4 428,236.7 386,236.7 365,200.4 386,164 428,164" fill="none" stroke="#231518" stroke-width="1.5"/>
<polygon points="449,127.6 428,164 386,164 365,127.6 386,91.3 428,91.3" fill="none" stroke="#231518" stroke-width="1.5"/>
<polygon points="512,91.3 491,127.6 449,127.6 428,91.3 449,54.9 491,54.9" fill="none" stroke="#231518" stroke-width="1.5"/>
<polygon points="575,127.6 554,164 512,164 491,127.6 512,91.3 554,91.3" fill="none" stroke="#231518" stroke-width="1.5"/>
<line x1="470" y1="164" x2="533" y2="200.4" stroke="#5B1426" stroke-width="2"/>
<circle cx="533" cy="200.4" r="4" fill="#5B1426"/>
<line x1="470" y1="164" x2="470" y2="236.7" stroke="#5B1426" stroke-width="2"/>
<circle cx="470" cy="236.7" r="4" fill="#5B1426"/>
<line x1="470" y1="164" x2="407" y2="200.4" stroke="#5B1426" stroke-width="2"/>
<circle cx="407" cy="200.4" r="4" fill="#5B1426"/>
<line x1="470" y1="164" x2="407" y2="127.6" stroke="#5B1426" stroke-width="2"/>
<circle cx="407" cy="127.6" r="4" fill="#5B1426"/>
<line x1="470" y1="164" x2="470" y2="91.3" stroke="#5B1426" stroke-width="2"/>
<circle cx="470" cy="91.3" r="4" fill="#5B1426"/>
<line x1="470" y1="164" x2="533" y2="127.6" stroke="#5B1426" stroke-width="2"/>
<circle cx="533" cy="127.6" r="4" fill="#5B1426"/>
<circle cx="470" cy="164" r="4" fill="#EDEFEC"/>
<text x="470" y="302" text-anchor="middle" font-size="18" fill="#231518">6 edge neighbours, all at 1</text>
<text x="470" y="324" text-anchor="middle" font-size="18" fill="#5A4549">no corner-only neighbours</text>
</svg>
<figcaption>A square cell has neighbours at two distances: 1 across an edge, about 1.41 across a corner. A hexagon has six neighbours, all across an edge and all at the same distance.</figcaption>
</figure>

On the sphere this holds approximately rather than exactly. For the Amsterdam cell in the example below, the six neighbour centres are between 2.10 and 2.32 km away.

### The catch: children do not tile the parent

Seven hexagons cannot be put together into one bigger hexagon. H3 places the seven children, one in the centre and six around it, rotated by about 19° relative to the parent. They cover the same area but not the same shape. When I measured it, 1/14 of the children's combined area sticks out past the parent's edge, and an equal area of the parent is covered by children of neighbouring cells.

<figure>
<svg viewBox="0 0 640 340" width="100%" style="max-width:100%;height:auto;display:block" role="img" aria-label="A large parent hexagon with seven smaller child hexagons drawn over it: one in the centre and six around it. The children are rotated about 19 degrees relative to the parent, so small wedges of the outer children stick out past the parent edge and matching wedges of the parent are left uncovered." font-family="'Barlow Condensed', sans-serif" font-weight="500">
<path d="M274,218.5 L254,225.4 L246,267 Z M134,267 L190,267 L174,253.1 L134,267 L134,267 Z M78,170 L106,218.5 L110,197.7 L78,170 L78,170 Z M134,73 L106,121.5 L126,114.6 L134,73 L134,73 Z M246,73 L190,73 L206,86.9 L198,128.4 L206,86.9 L246,73 L246,73 Z M274,121.5 L270,142.3 L302,170 Z" fill="rgba(58,29,36,.28)" stroke="none"/>
<path d="M134,73 L190,73 L174,59.1 Z M86,128.4 L78,170 L106,121.5 Z M102,239.3 L134,267 L106,218.5 Z M206,280.9 L246,267 L190,267 Z M294,211.6 L302,170 L274,218.5 Z M278,100.7 L246,73 L274,121.5 Z" fill="#8A2A3D" fill-opacity=".35" stroke="none"/>
<polygon points="230,156.1 222,197.7 182,211.6 150,183.9 158,142.3 198,128.4" fill="none" stroke="#8A2A3D" stroke-width="1.5"/>
<polygon points="302,170 294,211.6 254,225.4 222,197.7 230,156.1 270,142.3" fill="none" stroke="#8A2A3D" stroke-width="1.5"/>
<polygon points="254,225.4 246,267 206,280.9 174,253.1 182,211.6 222,197.7" fill="none" stroke="#8A2A3D" stroke-width="1.5"/>
<polygon points="182,211.6 174,253.1 134,267 102,239.3 110,197.7 150,183.9" fill="none" stroke="#8A2A3D" stroke-width="1.5"/>
<polygon points="158,142.3 150,183.9 110,197.7 78,170 86,128.4 126,114.6" fill="none" stroke="#8A2A3D" stroke-width="1.5"/>
<polygon points="206,86.9 198,128.4 158,142.3 126,114.6 134,73 174,59.1" fill="none" stroke="#8A2A3D" stroke-width="1.5"/>
<polygon points="278,100.7 270,142.3 230,156.1 198,128.4 206,86.9 246,73" fill="none" stroke="#8A2A3D" stroke-width="1.5"/>
<polygon points="302,170 246,267 134,267 78,170 134,73 246,73" fill="none" stroke="#5B1426" stroke-width="2" stroke-dasharray="7 5"/>
<circle cx="190" cy="170" r="3.5" fill="#5B1426"/>
<line x1="360" y1="96" x2="394" y2="96" stroke="#5B1426" stroke-width="2" stroke-dasharray="7 5"/>
<text x="406" y="102" font-size="18" fill="#231518">parent cell</text>
<rect x="360" y="128" width="34" height="18" fill="none" stroke="#8A2A3D" stroke-width="1.5"/>
<text x="406" y="143" font-size="18" fill="#231518">its 7 children</text>
<rect x="360" y="168" width="34" height="18" fill="#8A2A3D" fill-opacity=".35"/>
<text x="406" y="183" font-size="18" fill="#231518">child area outside parent</text>
<rect x="360" y="208" width="34" height="18" fill="rgba(58,29,36,.28)"/>
<text x="406" y="223" font-size="18" fill="#231518">parent area left uncovered</text>
<text x="360" y="262" font-size="16" fill="#5A4549">children rotated about 19°;</text>
<text x="360" y="282" font-size="16" fill="#5A4549">each shaded total is 1/14 of the parent</text>
</svg>
<figcaption>An H3-style parent (dashed) and its seven children. Wedges of the outer children stick out past the parent’s edge, and matching wedges of the parent belong to its neighbours’ children.</figcaption>
</figure>

So `cell_to_parent` is exact as an operation on ids, since every child has one parent, but the shapes differ. Roll fine cells up by parent id and some points land in a coarse cell whose polygon does not contain them. Of 20,000 random points around Amsterdam, about 6% had a resolution 9 cell whose resolution 7 parent was not the resolution 7 cell they fall in. For counts that is usually fine; for anything that must be exact, it is not.

H3 is also not equal area. The Amsterdam cell at resolution 7 is 4.32 km², below the 5.16 km² average, and in a random sample of 20,000 resolution 7 cells across the globe, the largest was about twice the area of the smallest.

### The id

An H3 cell id is a 64-bit integer, usually written as 15 hex digits. In `871969c9bffffff` the leading `8` comes from the mode field, which marks it as a cell, and the `7` is the resolution. Then come seven bits for the base cell (12 here) and one 3-bit digit per resolution. Digits beyond the cell's resolution are set to 7, binary `111`, which is why ids end in a run of `f`.

```python
import h3

cell = h3.latlng_to_cell(52.3731, 4.8926, 7)  # Dam Square

print(cell)                              # 871969c9bffffff
print(h3.get_resolution(cell))           # 7
print(h3.cell_to_parent(cell, 5))        # 851969cbfffffff
print(len(h3.cell_to_children(cell, 8))) # 7
print(len(h3.grid_disk(cell, 1)))        # 7: itself + 6
print(round(h3.cell_area(cell, unit="km^2"), 2))  # 4.32
a7 = h3.average_hexagon_area(7, unit="km^2")
print(round(a7, 2))                      # 5.16
```

## Other grids worth knowing

**HEALPix** comes from astronomy (Górski and colleagues, 2005), where it is the usual way to store all-sky maps such as the cosmic microwave background. The sphere is cut into 12 base pixels, each split into four again and again, so a map has 12 × N<sub>side</sub>² pixels. Every pixel has the same area, and pixel centres sit on rings of constant latitude, which makes spherical harmonic transforms fast. The hierarchy is exact.

**rHEALPix** rearranges HEALPix into six square base cells, a cube-like layout, and usually splits each cell 3 by 3 into nine children. It keeps equal area and an exact hierarchy, and its cells are squares in the projected plane.

**ISEA grids**, described by Kevin Sahr and colleagues and generated by the DGGRID software, use Snyder's equal-area projection of an icosahedron and lay hexagons, triangles or diamonds on it. ISEA3H, with aperture-3 hexagons, is the best known. Like H3 it has 12 pentagons at every resolution and no exact nesting. Unlike H3, its cells are equal area. The OGC publishes an abstract specification, Topic 21, for this whole family of discrete global grid systems.

**MGRS and UTM** are older and different in spirit: a reference you can read over a radio. UTM cuts the world into 60 zones, each 6° of longitude wide, between 80° S and 84° N. MGRS adds latitude bands lettered C to X, then 100 km squares named by two letters, then digits. Dam Square is `31U FU 28839 04222` to the metre. Cut the digits to `31U FU 28 04` and you have the 1 km square that contains it. Inside a zone the squares nest exactly and are in metres, but there is a seam at every zone edge, and the poles need a separate system.

## Side by side

<div class="table-wrapper" markdown="1">

| System | Cell shape | Exact hierarchy | Equal area | Neighbours | Typical uses |
|---|---|---|---|---|---|
| Lat/lon boxes | Rectangles in degrees | Yes, if you halve | No, shrinks with latitude | 8 at two distances; all meet at the poles | Climate and weather rasters |
| XYZ tiles, quadkeys | Squares in Web Mercator | Yes, 4 children | No; stops at 85° | 8 at two distances | Web maps, imagery tiles, image chips |
| Geohash | Rectangles in degrees | Yes, 32 children | No | 8; close points may share no prefix | Prefix keys in plain databases |
| S2 | Quadrilaterals on a cube | Yes, 4 children | No, within about 2× | 8; 3 cells meet at cube corners | Spatial indexing, region coverings |
| H3 | Hexagons, 12 pentagons | No, about 7 children | No, within about 2× | 6 at nearly equal distance | Aggregation, neighbourhoods, ML features |
| HEALPix | Curved quadrilaterals | Yes, 4 children | Yes | Mostly 8 | Astronomy, all-sky maps |
| rHEALPix | Squares in the projection | Yes, usually 9 children | Yes | Mostly 8 | Equal-area statistics |
| ISEA3H | Hexagons, 12 pentagons | No, aperture 3 | Yes | 6 | Equal-area global science |
| MGRS / UTM | Squares within each zone | Yes, within a zone | Nearly, within a zone | 8; seams at zone edges | Military, rescue, field work |

</div>

## Which one would I pick for what

- **Counting things and looking at neighbourhoods**, such as points, trips, detections or model features: H3. Equal neighbours and `grid_disk` rings make smoothing and context features easy. I accept that rolling up by parent is approximate.
- **Indexing shapes and testing containment exactly**: S2. Exact nesting and Hilbert-ordered ids turn polygon queries into range scans.
- **Lining up with a web map or imagery**: XYZ tiles. If training chips are cut along tiles, I do not want a second grid in between.
- **A quick key in a system with no geo support**: geohash, with explicit neighbour lookups for "nearby", because of the edge cases.
- **Equal-area statistics**, such as global densities: HEALPix near astronomy tooling, ISEA3H or rHEALPix otherwise.
- **Telling a person where to go**: MGRS. It is meant to be read aloud.
- **Spatial train/test splits**: almost any of them. Keep whole coarse cells on one side of the split.

None of these is the correct grid. Each is a set of compromises about shape, area and nesting, and the useful thing is to know which compromise you are making.

## Sources

- H3 documentation: [overview](https://h3geo.org/docs/core-library/overview), [indexing and approximate containment](https://h3geo.org/docs/highlights/indexing), [neighbours and aggregation](https://h3geo.org/docs/highlights/aggregation), [the index bit layout](https://h3geo.org/docs/core-library/h3Indexing)
- S2 Geometry: [the S2 cell hierarchy](https://s2geometry.io/devguide/s2cell_hierarchy) and [S2 cell statistics](https://s2geometry.io/resources/s2cell_statistics)
- Microsoft: [Bing Maps Tile System](https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system), for tiles, the 85.05° limit and quadkeys
- Gustavo Niemeyer, [geohash.org is public!](https://web.archive.org/web/20080305223755/http://blog.labix.org/#post-85), 26 February 2008 (archived), and the [Geohash description on Wikipedia](https://en.wikipedia.org/wiki/Geohash)
- G. M. Morton, [A computer oriented geodetic data base and a new technique in file sequencing](https://search.worldcat.org/title/A-computer-oriented-geodetic-data-base-and-a-new-technique-in-file-sequencing/oclc/52181262), IBM, Ottawa, 1966
- K. M. Górski et al., [HEALPix: a framework for high-resolution discretization and fast analysis of data distributed on the sphere](https://doi.org/10.1086/427976), The Astrophysical Journal 622, 759 to 771, 2005 ([arXiv](https://arxiv.org/abs/astro-ph/0409513))
- K. Sahr, D. White and A. J. Kimerling, [Geodesic discrete global grid systems](https://doi.org/10.1559/152304003100011090), Cartography and Geographic Information Science 30(2), 2003
