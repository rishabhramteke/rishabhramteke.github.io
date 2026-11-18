---
layout: post
title: "Below the canopy: looking for lost sites in lidar"
description: "A week of searching airborne laser scans for ancient sites nobody has recorded, in forests, closed military ranges and island valleys: how the search worked, what it found and what it could not see."
image: assets/images/blog/below-the-canopy-relief.jpg
image_show_on_page: false
category: blog
permalink: below-the-canopy
---
An aircraft flying a laser scanner over a forest sends out hundreds of thousands of pulses a second. Most bounce off leaves, but a few slip through the gaps and come back from the ground. Keep only those, and you get a model of the forest floor with the trees taken away. Banks, ditches and stone platforms that nobody can see from the air, and that are hard to notice on foot, show up as clear shapes.

This is how archaeologists found the Maya cities under the Guatemalan jungle and the Gallo-Roman farms under French forests. Over the past week I tried it myself, aiming for places that few people have searched: mountain forests, closed military ranges, and island valleys you can reach only from the sea. This post is about how the search worked, what it found, and the more useful list of things it could not see.

## How one search worked

I searched in blocks of about 100 km². For each block I used bare-earth terrain models built from the national lidar surveys, such as IGN's LiDAR HD in France and the National Land Survey's laser scanning in Finland. For Hawaiʻi I built the ground models myself from NOAA's raw point clouds, which are public domain.

<figure>
<svg viewBox="0 0 640 440" width="100%" style="max-width:100%;height:auto;display:block" role="img" aria-label="Six steps. One: choose a block of forest where the ground model is really bare earth. Two: render the relief so that half-metre banks and hollows show. Three: an AI reviewer reads every image and zooms into at most twelve shapes. Four: drop anything already in a heritage register. Five: I give each lead a verdict. Six: promising sites go to the regional heritage service." font-family="'Barlow Condensed', sans-serif" font-weight="500">
<g fill="none" stroke="rgba(58,29,36,.28)" stroke-width="1.5"><rect x="10" y="10" width="300" height="118" rx="6"/><rect x="330" y="10" width="300" height="118" rx="6"/><rect x="10" y="160" width="300" height="118" rx="6"/><rect x="330" y="160" width="300" height="118" rx="6"/><rect x="10" y="310" width="300" height="118" rx="6"/><rect x="330" y="310" width="300" height="118" rx="6"/></g>
<g stroke="#8A2A3D" stroke-width="2" fill="none"><path d="M312 69.0 h8"/><path d="M312 219.0 h8"/><path d="M312 369.0 h8"/><path d="M480.0 130 V144.0 H160.0 V150"/><path d="M480.0 280 V294.0 H160.0 V300"/></g>
<g fill="#8A2A3D"><path d="M320 64.0 l8 5 -8 5z"/><path d="M320 214.0 l8 5 -8 5z"/><path d="M320 364.0 l8 5 -8 5z"/><path d="M155.0 150 l5 8 5 -8z"/><path d="M155.0 300 l5 8 5 -8z"/></g>
<g font-size="23" font-weight="600" fill="#5B1426" letter-spacing="1"><text x="28" y="46">1 · CHOOSE A BLOCK</text><text x="348" y="46">2 · RENDER THE RELIEF</text><text x="28" y="196">3 · AI REVIEW</text><text x="348" y="196">4 · REGISTER CHECK</text><text x="28" y="346">5 · MY VERDICT</text><text x="348" y="346">6 · REPORT</text></g>
<g font-size="20" fill="#5A4549"><text x="28" y="78">About 100 km² of forest where</text><text x="28" y="104">the ground model is really bare</text><text x="348" y="78">Subtract the smooth hillside so</text><text x="348" y="104">half-metre banks and hollows show</text><text x="28" y="228">Read every 2 km image, zoom into</text><text x="28" y="254">at most 12 shapes, keep built ones</text><text x="348" y="228">Drop anything already in a</text><text x="348" y="254">public heritage register</text><text x="28" y="378">Promising, unsure, natural,</text><text x="28" y="404">modern or already known</text><text x="348" y="378">Send the promising ones to the</text><text x="348" y="404">regional heritage service</text></g>
</svg>
<figcaption>One search, from choosing the block to reporting what is left.</figcaption>
</figure>

The rendering step matters more than it sounds. A plain shaded relief is dominated by hills and valleys, and a 50 cm bank on a slope disappears. So I subtracted a smoothed copy of the terrain from itself, which leaves only the local bumps and hollows, and tinted them red for raised and blue for lowered. Then I cut each block into 2 km images with a labelled grid.

The first reading was done by an AI reviewer, Claude, with a brief for each region: what ancient sites look like there, and what usually fools you, such as forestry drains, tank tracks, landslides or lava. It read every image, zoomed into at most twelve spots with the relief next to an aerial photo, and kept only shapes that were clearly built and had several parts: an enclosure with a bank, a group of platforms, a set of terraces. A lone mound or ring did not count. Everything it kept was checked against the public heritage registers where they exist, and then I looked at each survivor myself and marked it.

## Did it find anything real?

The first test was a block in Sweden, where the national register is public and very complete. The reviewer flagged three sites, and all three were already registered: a stone setting, a grave field and an ancient field. That was the result I wanted. The method finds real things, and in a well-studied country they are already known.

In Finland I tuned a ring detector on cells full of registered tar pits. Two of its ten strongest rings were registered tar pits, against a 1% chance by luck. Then I ran it on the neighbouring cells, where nothing is registered.

Across 39 blocks, 50 leads reached my review. This is where they went:

<div class="table-wrapper" markdown="1">

| Where the leads went | Leads |
|---|---:|
| Reached my review | 50 |
| Already in a heritage register | 12 |
| Dropped by me as low confidence | 10 |
| Marked unsure | 19 |
| Promising, reported to the heritage services | 9 |

</div>

The nine that survived:

- **The Forêt de Châtillon in Burgundy:** four enclosures and a field system of low banks, the kind of Gallo-Roman farms already known from this forest. There is also an oval bank round a walled rectangle on the plateau above Voulaines-les-Templiers, a village named after its Templar commandery. It may be medieval. France does not publish its archaeological map, so I could not check whether any of these are known.
- **The forest south of Raahe in Finland:** four tar pits, rings 9 to 16 m across with a hollow up to 1.35 m deep in the middle. Ostrobothnia made tar for wooden ships for centuries, and many old pits are listed as heritage sites.

None of this is a discovery yet. A lead becomes a site when someone who knows the area looks at it, in the archive or on the ground. That is why the next step is the heritage services and not this blog, and why I give only the general areas here.

## What it could not see

The places that came back empty taught me more than the leads did.

**Small things vanish.** On La Gomera in the Canary Islands, the reviewer first checked a known site: the Fortaleza de Chipude, a flat-topped rock that the island's first people used as a sanctuary. The altars on top are rings and piles of stone a metre or two across, with burnt bones from offerings. None of it shows in a 1 m ground model. Ritual sites of that size cannot be found this way. Large ones, like Hawaiian heiau, which are temple platforms 20 to 60 m long, can be.

**Dense canopy starves the ground.** In the deep windward valleys of Hawaiʻi Island, such as Waimanu and Honopue, only 5 to 30% of the 1 m ground cells in the gorges got a laser return. The software fills the gaps from the nearest ground, and the fill draws its own patterns.

<span class="image fit"><img src="{{ '/assets/images/blog/below-the-canopy-relief.jpg' | relative_url }}" alt="Lidar relief of a forested gulch between open pastures. In the gulch the surface is a web of straight-edged polygons, while the pasture on either side shows fine natural bumps and a road." /></span>
*A forested gulch on the Kaʻū slope of Hawaiʻi Island, about 750 m across. Where too few pulses reached the ground, the gap filling makes a web of straight-edged polygons. The open pasture on either side is real ground. Lidar: NOAA, 2017, public domain.*

**Some ground models still have trees in them.** In a few forests the ground filter had left trees behind as sharp-sided discs, and the real forest floor was unreadable. I learned to compare the bare-earth model with the surface model before searching a block.

**Farming wipes the slate.** The Kaʻū slope of Mauna Loa was one of the big rain-fed farming areas of old Hawaiʻi. After a century of sugar cane the lower slope is graded smooth, and no field walls showed.

**Nature imitates buildings.** On lava, smooth domes called tumuli look like platforms, and pressure ridges look like walls. In limestone country, folded rock round a knoll can look like a hillfort. Most of what the reviewer zoomed into turned out to be geology or forestry.

**Closed is not the same as untouched.** I searched two former Soviet training grounds in Brandenburg, still closed because of unexploded munitions. Neither showed anything ancient that I could trust.

## The rules I kept

- No digging and no metal detecting at any of these places. Both can damage a site, and disturbing a protected site is against the law.
- Report to the regional heritage service first, and publish only general areas, never coordinates.
- Hawaiian heiau, house sites and burials are sacred and protected by state law. Nothing from Hawaiʻi will be published with a location.

Next I want to try the rainforest valleys of Molokaʻi, where a newer survey has many more pulses reaching the ground. The question is whether the platforms and terraces in those valleys finally show.
