"""Unified cpu/gpu backend resolution.

Both the vendored ``pdex`` dispatcher and the metric dispatchers consult
``resolve_backend`` so a single switch (``CELL_EVAL_BACKEND`` env var or an
explicit ``backend=`` kwarg) controls all subsystems.

Accepted values: ``"cpu"`` / ``"illico"`` and ``"gpu"`` / ``"rsc"``. The latter
aliases exist so backend-specific code that knows which implementation it wants
can be explicit; ``cpu``/``gpu`` is the user-facing terminology.
"""

from __future__ import annotations

import os
from typing import Literal

Backend = Literal["cpu", "gpu", "illico", "rsc"]
ResolvedBackend = Literal["cpu", "gpu"]

ENV_VAR = "CELL_EVAL_BACKEND"
DEFAULT_BACKEND: ResolvedBackend = "cpu"


def resolve_backend(backend: Backend | None = None) -> ResolvedBackend:
    """Resolve a user-supplied backend hint to ``"cpu"`` or ``"gpu"``.

    Order of resolution:
      1. ``backend=`` argument if not ``None``.
      2. ``CELL_EVAL_BACKEND`` environment variable.
      3. Default ``"cpu"``.
    """
    raw = backend if backend is not None else os.environ.get(ENV_VAR, DEFAULT_BACKEND)
    match raw.lower():
        case "cpu" | "illico":
            return "cpu"
        case "gpu" | "rsc":
            return "gpu"
        case other:
            raise ValueError(
                f"Unknown backend {other!r}. Expected one of: cpu, gpu, illico, rsc."
            )
