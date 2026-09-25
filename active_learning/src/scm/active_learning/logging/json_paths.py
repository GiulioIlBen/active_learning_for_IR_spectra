"""Deprecated compatibility imports for the former JSON-only module name."""

# TODO: Remove this compatibility module after downstream imports use state_paths.
from scm.active_learning.logging.state_paths import (
    PATH_POLICY_REGISTRY,
    collect_paths,
    collect_runtime_paths,
    fix_path,
    rewrite_payload_paths,
)

__all__ = [
    "PATH_POLICY_REGISTRY",
    "collect_paths",
    "collect_runtime_paths",
    "fix_path",
    "rewrite_payload_paths",
]
