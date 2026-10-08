"""Factory provisioning, device identity records, recovery and the four levels of reset (docs/PROVISIONING.md).

Everything here is a MODEL of the rules and the flows, run on the simulator: no bootloader, no OTP or eFuse, no secure element, no jig
hardware and no real HSM exist yet. What is fixed here is WHAT happens, in what order, who may do it, and what must stay working.
"""
