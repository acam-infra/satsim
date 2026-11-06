"""Co-simulation with orbitsim: a time-varying constellation drives the inference network.

The closed scenarios (``satsim.method``) test C1/C2 on a fixed topology with one scripted power
switch. This package replaces the script with an orbitsim contact trace (``orbitsim.contact_trace``):
links come and go with the orbits, compute satellites slow down in the umbra, and remote-sensing
(RS) satellites submit requests whose results must get back to them. See docs/adr/0005.
"""
