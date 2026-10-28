---
layout: post
title: When direct flights sell out, what happens to connecting fares?
image_show_on_page: false
image: assets/images/project/flight-price-substitution.png
category: project
description: Panel regressions, an event study and six robustness checks on how indirect flight prices track direct prices on European short-haul routes, 2023 vs 2026
permalink: flight-price-substitution
---
An empirical study of how the price of connecting flights moves with the price of non-stop flights on 120+ European short-haul routes, and what happens to connecting fares when the non-stop flight disappears. Code, results and a full question-by-question write-up are on <a href="https://github.com/rishabhramteke/flight-price-substitution">GitHub</a>.
<br>

<h4>The question</h4>
Airlines reprice seats continuously as departure approaches. A non-stop flight is the quality benchmark, so it carries a premium. When non-stop seats run out, the travellers who wanted one have to go somewhere, and connections are the obvious fallback. Two things follow, and I tested both. <b>Substitution</b>: do connecting fares track non-stop fares? <b>Scarcity</b>: do connecting fares jump when non-stops sell out?
<br>

<h4>Data and model</h4>
Fares for departures on 15-21 April, observed daily in the weeks before departure, in two years: a 2023 reference panel (172,591 connecting-flight observations, 128 routes) and my own scrape in 2026 (687,256 observations, 121 routes). The model is a panel regression of the log connecting fare on the average non-stop fare for the same route and date, its interaction with days before departure, travel time (and its square) and a sell-out indicator, with route and departure-date fixed effects and standard errors clustered by route. On top of that: an event study around the sell-out day, interaction models, and six robustness checks.
<br>

<h4>Result 1: connecting fares follow non-stop fares, more closely near departure</h4>
A 10% rise in the non-stop fare goes with a 2.6% to 3.4% rise in the connecting fare on the same route and date, and the link is clearly stronger as departure approaches.
<figure style="margin:1.2em 0"><img src="{{ site.url }}/assets/images/project/flights/01_main_coefficients.png" alt="All five regression coefficients, 2023, 2026 and pooled. Bars are 95% confidence intervals." style="width:100%;height:auto"><figcaption style="font-size:0.85em;opacity:0.8">All five regression coefficients, 2023, 2026 and pooled. Bars are 95% confidence intervals.</figcaption></figure>
<figure style="margin:1.2em 0"><img src="{{ site.url }}/assets/images/project/flights/02_substitution_over_time.png" alt="How closely connecting fares follow non-stop fares, by days before departure." style="width:100%;height:auto"><figcaption style="font-size:0.85em;opacity:0.8">How closely connecting fares follow non-stop fares, by days before departure.</figcaption></figure>

<h4>Result 2: scarcity is a dose, not a switch</h4>
Fares are higher when a non-stop flight has disappeared, and the more of a route's non-stops are gone, the higher they go: losing all of them implies roughly +22% (2023) and +27% (2026). A simple yes/no sell-out indicator would report only about 5%.
<figure style="margin:1.2em 0"><img src="{{ site.url }}/assets/images/project/flights/06_dose_response.png" alt="Implied connecting-fare increase against the share of non-stop flights that have disappeared." style="width:100%;height:auto"><figcaption style="font-size:0.85em;opacity:0.8">Implied connecting-fare increase against the share of non-stop flights that have disappeared.</figcaption></figure>

<h4>Result 3: the sell-out jump is clear in 2023, much less so in 2026</h4>
In the 2023 event study, connecting fares are flat until the day the first non-stop disappears, then step up by about 11% and stay up, while routes where non-stops never sell out stay flat. In 2026 fares were already climbing before the sell-out day and the comparison group climbs just as much, so that dataset cannot cleanly separate a sell-out effect from the ordinary rise near departure. I report it that way rather than as a replication.
<figure style="margin:1.2em 0"><img src="{{ site.url }}/assets/images/project/flights/03_event_study.png" alt="Connecting fares around the day a non-stop first disappears (0), against routes where non-stops never sell out." style="width:100%;height:auto"><figcaption style="font-size:0.85em;opacity:0.8">Connecting fares around the day a non-stop first disappears (0), against routes where non-stops never sell out.</figcaption></figure>

<h4>Result 4: fares follow non-stops up, but not down</h4>
In 2023, connecting fares responded to non-stop fares rising above their route-date average (0.55) and not at all to them falling below it (0.04). In 2026 they follow downward moves too, at roughly half the strength.
<figure style="margin:1.2em 0"><img src="{{ site.url }}/assets/images/project/flights/08_asymmetric_response.png" alt="Response of connecting fares to non-stop fares above and below their route-date average." style="width:100%;height:auto"><figcaption style="font-size:0.85em;opacity:0.8">Response of connecting fares to non-stop fares above and below their route-date average.</figcaption></figure>

<h4>How much to trust it</h4>
Six checks (first differences, a sell-out interaction, fixed vs random effects, residual diagnostics, Granger tests, and a placebo test that shuffles the sell-out indicator) all support the main findings. The placebo is the clearest: the real sell-out effect sits 21 (2023) and 29 (2026) standard deviations above shuffled runs.
<figure style="margin:1.2em 0"><img src="{{ site.url }}/assets/images/project/flights/14_placebo.png" alt="The true sell-out effect against 100 placebo runs with a shuffled sell-out indicator." style="width:100%;height:auto"><figcaption style="font-size:0.85em;opacity:0.8">The true sell-out effect against 100 placebo runs with a shuffled sell-out indicator.</figcaption></figure>
Two things did not hold up, and the write-up says so. The quantile regression came out identical at every quantile, which points to a computation problem, so I do not report it as a finding. And the data are observational: a demand shock that empties non-stop seats and lifts connecting fares at the same time cannot be fully ruled out.
<br>

<h4>2023 vs 2026</h4>
The mechanism is stable: substitution, a positive sell-out premium, no reliable difference on low-cost routes. What changed is the level and some details. With everything else controlled, 2026 connecting fares are 27% to 38% higher than in 2023, the travel-time discount that was clear in 2023 disappears, and sell-outs are rarer and come later in the booking window.
<figure style="margin:1.2em 0"><img src="{{ site.url }}/assets/images/project/flights/15_year_effect.png" alt="Higher 2026 connecting fares, with route, date, direct fare, travel time and sell-out controlled." style="width:100%;height:auto"><figcaption style="font-size:0.85em;opacity:0.8">Higher 2026 connecting fares, with route, date, direct fare, travel time and sell-out controlled.</figcaption></figure>

<h4>Links</h4>
<a href="https://github.com/rishabhramteke/flight-price-substitution">Code and results</a> · <a href="https://github.com/rishabhramteke/flight-price-substitution/blob/main/docs/FINDINGS.md">Full findings, every question with hypothesis and caveats</a>
