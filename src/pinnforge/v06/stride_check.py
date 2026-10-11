"""The v0.6 identifier reuses the v0.5.2 center-stride fraction."""


def listed_stride_is_v052(protocol: dict) -> bool:
    frac = float(protocol["methods"]["wsindy"]["test_function"]["center_stride_fraction"])
    return abs(frac - 1.0 / 6.0) < 1e-15
