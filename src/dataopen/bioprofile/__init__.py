"""BioProfile Engine (docs/BIOPROFILE.md): an online, per-player profile of how a person moves a mouse and reacts to targets.

MEASUREMENT ONLY. The engine consumes (a) the player's own relative mouse motion and (b) target observations (the detector's
`KeypointArray`, converted to angles from the crosshair), classifies each episode (wide flick / micro-tracking / surprise exposure,
with low-visibility as a modifier), keeps rolling median + sigma of the key metrics, tracks session fatigue, and publishes a compact
(< 100 bytes) versioned, CRC-protected profile that other modules can read. It never produces mouse or keyboard output.
"""
