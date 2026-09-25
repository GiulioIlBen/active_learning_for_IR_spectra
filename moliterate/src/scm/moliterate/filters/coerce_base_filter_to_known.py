from __future__ import annotations

from functools import lru_cache

from pydantic import TypeAdapter

# These functions should be useful to make filters to be loaded dynamically, so that allows to user implementations
# but difficult in the current set up, you should use registry pattern instead...


@lru_cache(maxsize=1)
def _union_filters_adapter() -> TypeAdapter:
    from scm.moliterate.filters import UnionFilters

    return TypeAdapter(UnionFilters)


def _try_parse_union_filter(value):
    if not isinstance(value, dict):
        return value
    if "type" not in value:
        return value
    try:
        return _union_filters_adapter().validate_python(value)
    except Exception:
        return value


def coerce_field_with_base_filter(value):
    if isinstance(value, dict):
        return [_try_parse_union_filter(value)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_try_parse_union_filter(v) for v in value]
    return _try_parse_union_filter(value)
