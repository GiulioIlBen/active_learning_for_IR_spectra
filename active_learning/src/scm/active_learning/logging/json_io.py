"""Deprecated compatibility imports for the former JSON-only module name."""

# TODO: Remove this compatibility module after downstream imports use state_io.
from scm.active_learning.logging.state_io import files_copy

__all__ = ["files_copy"]
