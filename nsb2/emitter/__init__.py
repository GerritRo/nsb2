"""Night sky background emission models.

Each module builds a :class:`~nsb2.core.sources.Source` for one contribution
to the night sky background: starlight, moonlight, zodiacal light and
airglow.  They are kept as separate modules rather than imported here,
because several of them download reference data on first use.
"""
