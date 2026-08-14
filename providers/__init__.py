"""Provider selection.

Each of the three integration points is chosen independently by an env var, so
you can run a real poser against mock rigging and mock training:

    PROVIDER_POSING=real flask --app app run

"real" resolves to providers/real/<name>.py. Those files are not in the repo
yet — each team adds their own. If you select "real" before the file exists,
you get a clear error naming the file to create rather than an ImportError.
"""

from __future__ import annotations

import importlib

import config
from providers.base import Poser, Rigger, Trainer

_SPECS = {
    "rigging": ("Rigger", "rigging"),
    "posing": ("Poser", "posing"),
    "training": ("Trainer", "training"),
}


def _load(kind: str, flavour: str):
    class_suffix, module_name = _SPECS[kind]
    module_path = f"providers.{flavour}.{module_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if module_path not in str(exc):
            raise
        raise RuntimeError(
            f"PROVIDER_{kind.upper()}={flavour} but {module_path.replace('.', '/')}.py "
            f"does not exist. Copy providers/mock/{module_name}.py to "
            f"providers/{flavour}/{module_name}.py and replace the body — see "
            f"docs/adding-a-provider.md."
        ) from exc

    # Convention: the class is named <Flavour><Suffix>, e.g. MockPoser, RealPoser.
    class_name = f"{flavour.capitalize()}{class_suffix}"
    if not hasattr(module, class_name):
        raise RuntimeError(
            f"{module_path} must define a class called {class_name} "
            f"subclassing providers.base.{class_suffix}."
        )
    return getattr(module, class_name)()


_cache: dict[str, object] = {}


def _get(kind: str, flavour: str):
    key = f"{kind}:{flavour}"
    if key not in _cache:
        _cache[key] = _load(kind, flavour)
    return _cache[key]


def get_rigger() -> Rigger:
    return _get("rigging", config.PROVIDER_RIGGING)


def get_poser() -> Poser:
    return _get("posing", config.PROVIDER_POSING)


def get_trainer() -> Trainer:
    return _get("training", config.PROVIDER_TRAINING)


def active() -> dict[str, str]:
    """Shown in the UI footer so it is always obvious what is real and what is not."""
    return {
        "rigging": config.PROVIDER_RIGGING,
        "posing": config.PROVIDER_POSING,
        "training": config.PROVIDER_TRAINING,
    }
