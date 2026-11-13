---
name: Rishabh Ramteke
description: A personal site drawn as surveyed map sheets, with work plotted on real terrain in wine ink on cool survey paper.
colors:
  wine: "#8A2A3D"
  wine-deep: "#5B1426"
  survey-paper: "#EDEFEC"
  survey-paper-shade: "#E2E5E1"
  wine-black-ink: "#231518"
  ink-faded: "#5A4549"
  neatline-rule: "#3A1D24"
  rule-soft: "rgba(58, 29, 36, .28)"
  grid-overprint: "rgba(138, 42, 61, .08)"
  reticle-hairline: "rgba(138, 42, 61, .32)"
  signal-invalid: "#A3122E"
typography:
  display:
    fontFamily: "Barlow Condensed, Barlow, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(2.6rem, 4.4vw, 4.4rem)"
    fontWeight: 600
    lineHeight: 1.02
    letterSpacing: "0.005em"
  headline:
    fontFamily: "Barlow Condensed, Barlow, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(2.2rem, 4vw, 3.4rem)"
    fontWeight: 600
    lineHeight: 1.02
    letterSpacing: "0.005em"
  title:
    fontFamily: "Barlow Condensed, Barlow, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.45rem"
    fontWeight: 600
    lineHeight: 1.1
    letterSpacing: "0.02em"
  lede:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "1.18rem"
    fontWeight: 400
    lineHeight: 1.5
  body:
    fontFamily: "Barlow, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 400
    lineHeight: 1.55
  note:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "0.97rem"
    fontWeight: 400
    lineHeight: 1.45
  annotation:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "0.86rem"
    fontWeight: 400
    lineHeight: 1.35
  point-label:
    fontFamily: "Barlow, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.05rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.01em"
  readout:
    fontFamily: "Barlow Condensed, Barlow, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.8rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.07em"
    fontFeature: "tnum"
  label-caps:
    fontFamily: "Barlow Condensed, Barlow, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.74rem"
    fontWeight: 600
    lineHeight: 1.55
    letterSpacing: "0.16em"
  tick:
    fontFamily: "Barlow Condensed, Barlow, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.78rem"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "0.3em"
  map-lettering:
    fontFamily: "Barlow, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 500
    letterSpacing: "0.02em"
    fontFeature: "tnum"
rounded:
  none: "0px"
spacing:
  gutter: "clamp(16px, 2.2vw, 32px)"
  furniture-inset: "3.5%"
  note-pad: ".75rem .9rem"
  titleblock-pad: "1rem 1.3rem 1.1rem"
  legend-pad: "1.6rem 1.8rem 1.5rem"
  band-y: "clamp(2.5rem, 6vw, 4.5rem)"
components:
  button-primary:
    backgroundColor: "{colors.wine}"
    textColor: "{colors.survey-paper}"
    rounded: "{rounded.none}"
    padding: ".5rem 1.05rem"
  button-primary-hover:
    backgroundColor: "{colors.wine-deep}"
    textColor: "{colors.survey-paper}"
  button-submit:
    backgroundColor: "{colors.wine}"
    textColor: "{colors.survey-paper}"
    rounded: "{rounded.none}"
    padding: ".85rem 1.4rem"
  legend-box:
    backgroundColor: "{colors.survey-paper}"
    textColor: "{colors.wine-black-ink}"
    rounded: "{rounded.none}"
    padding: "{spacing.legend-pad}"
    width: "min(30rem, 34%)"
  title-block:
    backgroundColor: "{colors.survey-paper}"
    textColor: "{colors.ink-faded}"
    rounded: "{rounded.none}"
    padding: "{spacing.titleblock-pad}"
    width: "min(24rem, 30%)"
  survey-note:
    backgroundColor: "{colors.survey-paper}"
    textColor: "{colors.wine-black-ink}"
    typography: "{typography.note}"
    rounded: "{rounded.none}"
    padding: "{spacing.note-pad}"
    width: "19rem"
  input-underline:
    backgroundColor: "transparent"
    textColor: "{colors.wine-black-ink}"
    rounded: "{rounded.none}"
    padding: ".45rem 0"
---

# Design System: Rishabh Ramteke

## Overview

**Creative North Star: "The Survey Sheet"**

Every surface is a printed map sheet: a ruled neatline on cool survey paper, real contours drawn from a public elevation model in wine overprint, and the content laid on top as map furniture. Who he is sits in a legend box, what he has made is plotted as survey points with notes, and the sheet's facts sit in a title block with a working scale bar. The sheet is numbered (Sheet 1 Work, Sheet 2 Journal, Sheet 3 Travel), and moving between surfaces is moving between sheets of one series.

The sheet is alive but quiet. It is printed in separate overprint layers that drift apart with the pointer and rise at different rates on scroll; plotted points float on their own slow paths and lean a little toward the cursor; the cursor itself is a survey reticle that reads the grid square and the ground height under it. Nothing bounces, flashes or announces itself. The palette is one ink, wine, in three strengths, on two tones of paper.

Density is that of a good topographic map: a lot of fine information (ticks, contour values, spot heights, tabular coordinates), held in check by generous paper and a small number of bordered boxes. The world refuses the dark developer portfolio and the headline-plus-card-grid.

**Key Characteristics:**
- Cool survey paper ground with a faint wine grid of eight columns (A to H) and numbered rows.
- One ink family: wine for lines and symbols, deep wine for titles, wine-black for reading text.
- Map furniture (legend box, title block, survey note) as flat paper panels with a 1.5px rule; no radius, no drop shadow.
- Real terrain: contours from Copernicus DEM GLO-30, values and spot heights only, never names.
- Grotesque condensed caps and tabular figures for map data; a serif for anything read as prose.
- Motion as a printed sheet coming apart in layers, never as UI animation.

## Colors

One wine ink printed at three strengths on cool, slightly green-grey survey paper.

### Primary
- **Survey Wine** (wine): the overprint ink. Contour lines, point symbols, range rings, the journal-stop rule, the globe graticule and route arcs, the primary action fill, links on hover, focus outlines, selection, and the small data lettering (coordinates, note meta, captions, form labels).
- **Deep Wine** (wine-deep): the title ink. Display and headline type, title-block titles, journal stop titles, the trig point label, the scale bar's filled segments, coastlines on the globe, and the hover state of the primary action.

### Neutral
- **Survey Paper** (survey-paper): the page ground, every furniture panel's fill, and the knockout halo behind lettering. Also the globe sphere, so the far side reads as hidden.
- **Shaded Paper** (survey-paper-shade): Sheet 2's ground, so the journal reads as a different sheet of the same series. Also the scrollbar track.
- **Wine-Black Ink** (wine-black-ink): body text, point labels, note text, form input text.
- **Faded Ink** (ink-faded): secondary prose: ledes, title-block values, journal stop notes, gazetteer figures, colophon, scale labels.
- **Neatline Rule** (neatline-rule): the 1.5px border of the neatline and every furniture box, the scale bar outline, the grid ticks, input underlines.
- **Soft Rule** (rule-soft): 1px dividers inside furniture (gazetteer rows, the title block's "next sheet" link, the colophon).
- **Grid Overprint** (grid-overprint): the eight-by-six sheet grid printed as 1px lines on the neatline.
- **Reticle Hairline** (reticle-hairline): the cursor crosshair lines.

### Signal
- **Invalid Red** (signal-invalid): the input underline after a failed submit (`:user-invalid`). Its only use.

### Named Rules
**The One Ink Rule.** Every mark is wine at some strength (wine, deep wine, wine-black, or wine at reduced opacity). Contours print at 26% (ordinary) and 50% (index); the grid at 8%; the reticle at 32%. A second hue is never introduced for decoration.

**The Paper Panel Rule.** Furniture is paper on paper: the same survey-paper fill as the ground, separated only by the neatline rule. Never tint a panel to make it stand out.

## Typography

**Display Font:** Barlow Condensed (with Barlow, Helvetica Neue, Arial)
**Body Font:** Barlow (with Helvetica Neue, Arial)
**Prose Font:** Source Serif 4 (with Georgia)

All three are self-hosted (`assets/fonts/fonts.css`), Latin and Latin Extended subsets. Barlow 400/500/600, Barlow Condensed 500/600, Source Serif 4 400 roman and italic.

**Character:** a surveyor's grotesque for names, data and labels, set against a quiet book serif for anything meant to be read as a sentence. The condensed cut does the cartographic work: titles, ticks, coordinates, captions.

### Hierarchy
- **Display** (Barlow Condensed 600, clamp(2.6rem, 4.4vw, 4.4rem), 1.02): the person's name in the legend box; on Travel, the sheet heading (clamp(2.2rem, 3.6vw, 3.6rem)). Deep wine, balanced wrap.
- **Headline** (Barlow Condensed 600, clamp(2.2rem, 4vw, 3.4rem), 1.02): section headings (Journal, Write to me). Deep wine.
- **Title** (Barlow Condensed 600, 1.45rem, 1.1, 0.02em): title-block titles ("Sheet 1 · Work"); journal stop titles at 1.35rem.
- **Lede** (Source Serif 4 400, 1.16–1.18rem, 1.42–1.5): the role line under the name and the lede under section headings. Max 36–46ch.
- **Body** (Barlow 400, 1.0625rem, 1.55): base text and links; links and actions at 500–600.
- **Point label** (Barlow 600, 1.05rem, 1.2): plotted point names on the map, with the paper halo; the trig point at 1.15rem in deep wine.
- **Note** (Source Serif 4 400, 0.97rem, 1.45): the survey note that lifts from a point.
- **Annotation** (Source Serif 4 italic 400, 0.86–1.02rem): title-block values, journal stop notes, projection notes, the colophon, the quote attribution.
- **Readout** (Barlow Condensed 600, 0.8–0.84rem, 0.07em, tabular figures): coordinates under the name, note meta (grid square and height), the cursor readout (0.14em).
- **Label caps** (Barlow Condensed 600, 0.74rem, 0.16em, uppercase): title-block keys, the gazetteer caption; form labels at 0.8rem, 0.2em.
- **Tick** (Barlow Condensed 500, 0.78rem, 0.3em): the A–H and 1–n grid ticks inside the neatline, in rule colour.
- **Map lettering** (Barlow 500, 15px in sheet units, tabular): contour values in wine and spot heights in deep wine (14px), rotated along the line, knocked out with a 4px paper stroke.

### Named Rules
**The Tabular Figures Rule.** Every number that is data (coordinates, heights, grid squares, years, the place count) is set with `font-variant-numeric: tabular-nums`.

**The Prose Is Serif Rule.** If it reads as a sentence, it is Source Serif 4; if it names, labels or measures, it is Barlow or Barlow Condensed.

## Layout

**The sheet.** Each sheet is a section padded by the gutter (clamp(16px, 2.2vw, 32px)) holding a neatline: a 1.5px ruled frame, `overflow: hidden`, with an eight-column by n-row grid printed inside it (12.5% columns; 6 rows on Sheet 1 and Travel, 4 on Sheet 2). Sheet 1 and Travel fill the viewport (100svh, min 660px, max 1000px on Sheet 1); Sheet 2 is content height. Grid ticks sit 6px inside the top and left edges.

**Furniture placement.** On a full sheet the legend box is anchored top left (3.5% in, 7% down) and the title block bottom right (3.5% in, 6% up). Content is placed on the sheet by percentage coordinates (`--x`, `--y`), and a collision guard nudges any point that lands under a furniture box out by the shortest way, with a 14px margin. Sheet 2 is a two-column grid (text and title block left, globe right, up to 560px).

**Bands between sheets.** A sheet can be followed by a plain band on the page ground, outside any neatline: the quote band (centred on one line where it fits, vertical padding clamp(2.5rem, 6vw, 4.5rem)) between Sheet 1 and Sheet 2, and the imprint (max 72rem, two columns, form on the right) after the last sheet.

**Mid widths (961–1240px).** The furniture tightens so it does not swallow the map: legend min(25rem, 38%) with 1.2rem 1.3rem padding and display at clamp(2.3rem, 3.9vw, 3.2rem); title block min(22rem, 34%) with .8rem 1rem .9rem padding.

**Narrow (≤960px).** The sheet stops being a single plate and becomes a folded sheet read top to bottom. The neatline turns into a column: a map face 86vw tall (contours plus the same point symbols, drifting gently), then the legend, then the points as a list with their notes always shown, then the title block. Furniture loses its box and keeps only a 1.5px top rule. Grid ticks and all map lettering are hidden. Sheet 2 and the imprint collapse to one column; on Travel the heading and title block become full-width bands pinned to the top and bottom of the globe.

## Elevation & Depth

The system is flat. Depth comes from print, not light: separate overprint layers that drift apart, and paper panels lying on the paper ground separated by a rule. There are no drop shadows anywhere.

### Shadow Vocabulary
- **Paper halo** (`text-shadow: 0 0 2px paper, 0 0 2px paper, 0 0 3px paper, 1px 1px 0 paper, -1px -1px 0 paper`): a cartographic knockout behind lettering that sits over contours (point labels, the cursor readout). It reads as cleared paper, not as a shadow. In SVG the same job is done by `paint-order: stroke` with a 4px paper stroke.
- **Focus underline** (`box-shadow: 0 1.5px 0 wine`): thickens a form field's underline to 3px on focus. Its only use.

### Named Rules
**The Overprint Rule.** Depth is the sheet's layers parting: ordinary contours (depth 0.2), index contours and lettering (depth 0.45), plotted points (depth 1.15). Never fake depth with a shadow, blur or gradient.

## Shapes

Square corners everywhere (0px). Boxes are drawn with a 1.5px solid neatline rule; internal dividers are 1px soft rule. Circles appear only as map symbols: the work point (a 5-unit circle in a 16-unit box), the dashed range ring (46px, 1.25px dashed wine), globe route dots. The other symbols are the trig point (a triangle with a centre dot, 20px) and the journal stop (a diamond, 16px), all outlined in wine at 2–2.2px and filled with paper at rest. The journal stops on Sheet 2 are linked by a 2px dashed left rule; Sheet 1 carries no route line. The scale bar is a 7px bar of five alternating deep wine and paper segments.

## Components

### Buttons
Restrained and printed: a solid block of ink, no radius, no shadow.
- **Shape:** square (0px).
- **Primary:** wine fill, paper text, Barlow 500, .5rem 1.05rem padding (the Résumé action in the legend). The form submit uses the same fill at Barlow 600 1rem with .85rem 1.4rem padding.
- **Hover / Focus:** fill shifts to deep wine over .25s on the house ease; focus is a 2px wine outline offset 3px (global).
- **Secondary:** underlined text links in the same row (LinkedIn, Scholar, GitHub, Contact), 1px underline at .22em offset, wine on hover.

### Inputs / Fields
- **Style:** underline only. Transparent ground, a 1.5px rule along the bottom, square, Source Serif 4 1.05rem text, wine caret. Labels above in label caps (0.8rem, 0.2em, uppercase, wine).
- **Focus:** the underline turns wine and thickens with the focus underline; no outline box.
- **Error:** after a failed submit the underline turns Invalid Red.

### Navigation
There is no navigation bar. Primary links live in the legend box's action row; the title block carries the link to the next sheet (a 1px soft rule above, underlined, 600); a sheet's own heading carries the link back ("Back to Sheet 2 · Journal"). The plotted points are the work navigation on Sheet 1.

### Neatline Sheet (signature)
The frame every surface lives in. A 1.5px rule, the wine grid at 8%, grid ticks lettered A–H across and numbered down, a DEM contour plate behind (`preserveAspectRatio: slice`) on Sheet 1, and a survey-reticle cursor on fine pointers. The contour plate on Sheet 1 is the Dolomites (Seceda to Sella) at 30 m intervals with 150 m index contours; it carries contour values and spot heights only.

### Legend Box
Top left, paper panel with a 1.5px rule, min(30rem, 34%) wide. Holds the name at display size, the role line in serif lede, the coordinates and H3 cell in readout type, then the action row. No symbol key.

### Plotted Point
A symbol and a label, placed by percentage. At rest: paper-filled wine outline, label in wine-black with the paper halo. On hover or focus: the symbol fills wine, the label turns deep wine, a dashed range ring scales in from 0.4 (.5s), and the survey note rises 6px into place (.25s opacity, .35s transform). The note is a paper panel (19rem, 1.5px rule) with a readout line (grid square and the ground height read from the DEM) above a serif sentence. Points flip their note left (`side: left`) or upward (when below 55% of the sheet) to stay on the sheet. The trig point is the start point, larger and in deep wine.

### Title Block with Scale Bar
Bottom right paper panel, min(24rem, 30%). A title ("Sheet 1 · Work"), a live scale bar sized to the drawn map scale (the largest of 10, 5, 2 or 1 km that fits in 220px), then a two-column definition list (label caps keys in wine, italic serif values in faded ink), and the link to the next sheet. On Sheet 2 it sits in the flow and carries the projection note and the route gazetteer.

### Route Gazetteer
A compact table inside the title block: label-caps caption, rows divided by 1px soft rule, place in condensed deep wine, years in ink, latitude and longitude right-aligned in faded ink, all tabular.

### Journal Stops
An ordered list joined by a 2px dashed wine rule down the left; each stop is a diamond symbol (paper fill, wine on hover), a condensed deep-wine title (underlined on hover), and an italic serif note.

### Globe
A WebGL orthographic-feeling globe: a paper sphere, a wine graticule every 20° at about 20% opacity, Natural Earth 110m coastlines in deep wine at 85%, the lived-in route as raised wine arcs with wine dots, and a small deep-wine plane flying the legs in turn. It turns slowly on its own and can be dragged; it loads only when near the viewport. On Travel it fills the sheet, adds country borders at 28%, wheel and pinch zoom, and paper discs with a wine rim for each place, with a paper hover card (1.5px rule) naming the city.

### Quote Band
A centred band between sheets on the page ground: the quote in condensed 600 uppercase at 0.22em in deep wine, attribution on the same line in italic serif after a dash.

### Imprint
The closing band: a headline, a row of underlined links, the underline-field contact form, and a colophon in italic serif above a soft rule.

### Motion Grammar
- **House ease:** cubic-bezier(.16, 1, .3, 1) for every transition.
- **Contour draw-in:** on load, ordinary contours draw in over 1.6s, index contours start .25s later, lettering fades in over .8s at 1.1s. Content is visible from the start.
- **Layer drift:** layers translate opposite the pointer (up to 22px across, 15px down, times layer depth) and rise at 0.16 of scroll, eased toward target at 0.07 per frame.
- **Point float and lean:** each point drifts on its own sine path (up to 10px by 8px) and, within 170px of the cursor, leans toward it (factor 0.2, capped at 18px, eased at 0.12).
- **Reticle cursor:** on fine pointers only, a 1px crosshair at 32% wine with a readout of grid square and ground height; it fades in over .3s and the native cursor becomes a crosshair (a pointer over links).
- **Mobile face:** the map-face symbols bob gently (±3–4px, 6s plus 1s per point, alternating).
- **Reduced motion:** no draw-in, no ambient float or bob, no globe spin or plane flight (the plane rests at the last leg), no note, ring or reticle transitions, no smooth scroll.

## Do's and Don'ts

### Do:
- **Do** put every new surface inside a neatline, with its sheet number in a title block ("Sheet n · Name").
- **Do** draw with one ink: wine (#8A2A3D) for lines and symbols, deep wine (#5B1426) for titles, wine-black (#231518) for text, on survey paper (#EDEFEC).
- **Do** make furniture paper on paper: survey-paper fill, 1.5px neatline rule, square corners.
- **Do** set data in condensed type with tabular figures, and prose in Source Serif 4.
- **Do** knock lettering out of the contours with the paper halo rather than a backing box.
- **Do** use real terrain from a public DEM and say which in the title block.
- **Do** keep motion layered and slow on the house ease, and stop all ambient motion under `prefers-reduced-motion`.
- **Do** at ≤960px turn the sheet into a column: map face, then legend, then a list, then title block, each separated by a 1.5px top rule.

### Don't:
- **Don't** letter place or peak names on a map; contour values and spot heights only.
- **Don't** add a map key or symbol legend to the legend box.
- **Don't** name an employer anywhere on the sheets.
- **Don't** use drop shadows, blurs, gradients or rounded rectangles; depth is overprint layers, and circles are map symbols only.
- **Don't** introduce a second accent hue; Invalid Red is for a failed field and nothing else.
- **Don't** build a dark developer portfolio or a headline-plus-card-grid.
