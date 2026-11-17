---
layout: post
title: "Printable things: keycaps and fixes for the house"
description: "3D models I design for printing, from keycaps for my keyboards to small fixes around the house. Every model can be turned in the browser and downloaded as an STL."
image_show_on_page: false
image: assets/images/project/printable-things.png
category: project
permalink: printable-things
models:
  - title: Mount Fuji keycap
    file: assets/models/keycap-fuji.stl
    size: 943 KB
    about: "A 1u keycap for Cherry MX-style switches whose top is the real ground around Mount Fuji: a 14 km square of the Copernicus 30 m elevation model shrunk onto the 12.7 mm top, with heights doubled so the cone reads at this size. It stands 12.4 mm tall, taller than a normal key, so it suits Esc or a corner."
    print: "Upright, resin if you have it; on FDM use 0.08–0.12 mm layers and a 0.2 mm nozzle for the slopes. The inside bridges without supports."
  - title: Matterhorn keycap
    file: assets/models/keycap-matterhorn.stl
    size: 943 KB
    about: "The same keycap with the 5.2 km square around the Matterhorn on top, at true scale: the pyramid needs no exaggeration and rises 5 mm off the key, 11.9 mm tall in all. The 30 m elevation model rounds the summit to 4,330 m instead of 4,478 m."
    print: "Upright, resin if you have it; on FDM use 0.08–0.12 mm layers. The ridges are fine, so print slowly. The inside bridges without supports."
  - title: Swing-out towel hook
    file: assets/models/folding-towel-hook.stl
    size: 1.3 MB
    about: "A wall hook that lies flat along the wall and swings out 90 degrees when you need it: a slim pill-shaped plate, rounded edges all round, and an arm that tapers into a J and ends in a round knob that keeps towels from sliding off. The hinge axis is vertical, so a hanging towel pulls along the pin and cannot fold it shut."
    print: "Prints in place, lying flat: the arm turns on a pin with 0.4 mm clearance. PETG, 0.2 mm layers, four walls. Free the hinge gently after printing, then fix it with two 4 mm countersunk screws, hinge side vertical."
---
I am setting up a 3D printer, and this page collects the models I design for it: keycaps for my keyboards first, then small things the house needs. They are generated from a parameter script, so sizes and clearances are easy to change; the [script is on GitHub](https://github.com/rishabhramteke/rishabhramteke.github.io/blob/master/tools/models/make_models.py).

The mountain keycaps are cut from the Copernicus GLO-30 elevation model: © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA; all rights reserved. A [second script](https://github.com/rishabhramteke/rishabhramteke.github.io/blob/master/tools/models/fetch_terrain.py) fetches the terrain, so any other mountain is one line away.

None of these has been test-printed yet, so treat the clearances as a starting point. Drag a model to turn it.
