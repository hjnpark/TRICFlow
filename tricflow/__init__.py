"""TRICFlow: reaction-pathway workflow using geomeTRIC."""

from ._version import get_versions

__version__ = get_versions()["version"]

# Mirrors tricflow.tricflow.__all__
_PUBLIC_API = (
    "TRICWorkflow",
    "optimize_frames",
    "interpolate",
    "run_neb",
    "optimize_ts",
    "run_irc",
    "get_energies",
    "assemble_pathways",
)

_LAZY_SUBMODULES = frozenset({"tricflow", "refine", "errors"})


def __getattr__(name: str):
    if name in _PUBLIC_API:
        import importlib

        mod = importlib.import_module(".tricflow", __name__)
        value = getattr(mod, name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_PUBLIC_API) | _LAZY_SUBMODULES)


__all__ = ["__version__", *_PUBLIC_API, *_LAZY_SUBMODULES]