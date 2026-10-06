"""Video path of the assistive device: HDMI/DisplayPort passthrough with a parallel capture tap, and the SoC-side frame preparation
(colour, crop, downscale, DMA buffer) that hands 640x640 RGB frames to the existing runtime. See docs/VIDEO.md.

What is here: timing and link-budget math, EDID parsing and the passthrough rules, the input autodetect state machine, a C99 streaming
frame-preparation core with a numpy reference, geometry back to screen pixels, and a simulator of capture timing. What is NOT here:
the passthrough/capture hardware itself (no board)."""
