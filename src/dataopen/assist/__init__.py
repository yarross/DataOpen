"""Adaptive Sensitivity Correction (ASC, docs/ASSIST.md): an accessibility aid that lowers the EFFECTIVE sensitivity of a pointing device
near an object of interest, using the person's BioProfile. Output is a scalar K in [0.1, 1] and the input deltas scaled by it.

Hard guarantees (property-tested): K never exceeds 1.0 (no amplification), K is exactly 1.0 until the person has started moving,
the scaled delta never has a larger magnitude or another sign than the raw one, and zero input gives zero output: the module can
not move the pointer by itself. Where the objects come from is outside the module (an adapter supplies the nearest one per tick).
"""
