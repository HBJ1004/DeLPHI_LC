"""DeLPHI software version.

Scientific result metadata never lives in this source file. Accepted metrics
are read from immutable, schema-validated release manifests so documentation
cannot drift from per-object evidence.
"""

__version__ = "2.0.0.dev0"


def get_version() -> str:
    return __version__


def get_version_info() -> dict[str, str | bool]:
    return {
        "version": __version__,
        "description": "DeLPHI validity-first asteroid lightcurve pipeline",
        "status": "research_only",
        "scientific_release_approved": False,
    }
