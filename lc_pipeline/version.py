"""DeLPHI software version and metadata."""

__version__ = "1.1.1"


def get_version() -> str:
    return __version__


def get_version_info() -> dict[str, str | bool]:
    return {
        "version": __version__,
        "description": "DeLPHI K3 asteroid pole-axis candidate generator",
        "status": "stable_research_release",
        "scientific_release_approved": True,
    }
