"""Full-Profile Analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("full_profile")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Roger A.R. Moens"
__email__ = "r.a.r.moens@tudelft.nl"
