"""Assistive HID bridge: a transparent USB proxy for one mouse that edits only the X/Y fields of the mouse's own reports, and only
ever makes them smaller. C99 core (csrc/), ctypes wrapper, and a simulator of the whole rig (mouse, PC, analog switch, watchdog, module).
See docs/BRIDGE.md."""
