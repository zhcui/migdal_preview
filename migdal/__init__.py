"""Production radial one-band Migdal/GW drivers."""

__all__ = ["G0W0", "GW0", "ScGW", "TakadaProjection"]


def __getattr__(name):
    if name == "G0W0":
        from migdal.g0w0 import G0W0

        return G0W0
    if name == "GW0":
        from migdal.gw0 import GW0

        return GW0
    if name == "ScGW":
        from migdal.scgw import ScGW

        return ScGW
    if name == "TakadaProjection":
        from migdal.takada import TakadaProjection

        return TakadaProjection
    raise AttributeError(name)
