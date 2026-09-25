def conversion_ratio(in_unit: str, out_unit: str) -> float:
    try:
        from scm.plams import Units
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "conversion_ratio: plams is required but not installed. To install it use `pip install plams`"
        )

    return Units.conversion_ratio(in_unit, out_unit)
