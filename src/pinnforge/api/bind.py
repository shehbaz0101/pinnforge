"""Loopback bind address for ``pinnforge serve``.

The default listen address is ``127.0.0.1``. ``0.0.0.0`` and ``::``
accept connections on every interface. Those hosts are refused unless
the caller passes ``allow_remote``. This module does not import
FastAPI, uvicorn, or torch.
"""

from __future__ import annotations

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Hosts that listen on every interface. ``[::]`` is the bracketed form
# some tools print for the IPv6 wildcard.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})


def resolve_bind_host(host: str | None, *, allow_remote: bool = False) -> str:
    """Return the address ``pinnforge serve`` should bind.

    ``None`` selects :data:`DEFAULT_HOST`. A wildcard host is returned
    only when ``allow_remote`` is true. Any other explicit host is
    returned unchanged, aside from surrounding whitespace.

    Raises:
        ValueError: ``host`` is empty, or it is a wildcard and
            ``allow_remote`` is false.
    """

    if host is None:
        return DEFAULT_HOST
    if not isinstance(host, str):
        raise ValueError("host must be a string")
    cleaned = host.strip()
    if cleaned == "":
        raise ValueError("host must not be empty")
    if cleaned.lower() in _WILDCARD_HOSTS:
        if not allow_remote:
            raise ValueError(
                f"refusing to bind all interfaces ({cleaned}). "
                f"The default bind is {DEFAULT_HOST}. "
                "Pass --allow-remote to force this host."
            )
    return cleaned


def resolve_port(port: int) -> int:
    """Return ``port`` when it is a TCP port number.

    Raises:
        ValueError: ``port`` is not an integer from 1 to 65535.
    """

    if isinstance(port, bool) or not isinstance(port, int) or port < 1 or port > 65535:
        raise ValueError("port must be an integer from 1 to 65535")
    return port
