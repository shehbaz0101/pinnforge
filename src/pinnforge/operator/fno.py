"""1D Fourier neural operator for Burgers windows.

The architecture follows Li et al. (2021): a pointwise lift, spectral
convolution on the retained low modes plus a local 1x1 convolution, GELU
between Fourier layers, and a pointwise projection to the target frames.
Fourier layers act on the spatial axis. Time enters only as channels: the
input frames, then one extra channel that is the normalized viscosity,
constant in space.

Spectral weights are stored as real and imaginary float parameters so a
checkpoint loads with ``weights_only=True`` without a complex tensor.
"""

from __future__ import annotations

from pinnforge.ml_import import require_torch

torch = require_torch()
nn = torch.nn


class SpectralConv1d(nn.Module):
    """Multiply the lowest ``modes`` rFFT coefficients by a learned matrix.

    Higher modes are left at zero. If the grid has fewer stored frequencies
    than ``modes``, the extra weights are unused for that forward pass.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        _require_positive_int(in_channels, "in_channels")
        _require_positive_int(out_channels, "out_channels")
        _require_positive_int(modes, "modes")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("spectral conv expects (batch, channels, n_space)")
        batch, channels, n_space = values.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {channels}")
        spectrum = torch.fft.rfft(values, dim=-1)
        used = min(self.modes, int(spectrum.shape[-1]))
        weight = torch.complex(self.weight_real[:, :, :used], self.weight_imag[:, :, :used])
        out_spectrum = torch.zeros(
            batch,
            self.out_channels,
            spectrum.shape[-1],
            dtype=torch.complex64,
            device=values.device,
        )
        out_spectrum[:, :, :used] = torch.einsum("bcm,com->bom", spectrum[:, :, :used], weight)
        return torch.fft.irfft(out_spectrum, n=n_space, dim=-1)


class FourierLayer1d(nn.Module):
    """Spectral convolution plus a pointwise local branch."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.spectral(values) + self.local(values)


class FNO1d(nn.Module):
    """Map a channel stack on a periodic line to an output channel stack.

    ``forward`` takes ``(batch, in_channels, n_space)`` and returns
    ``(batch, out_channels, n_space)``. For the Burgers window task,
    ``in_channels`` is the input-frame count plus one viscosity channel,
    and ``out_channels`` is the target-frame count.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        width: int = 32,
        modes: int = 16,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        _require_positive_int(in_channels, "in_channels")
        _require_positive_int(out_channels, "out_channels")
        _require_positive_int(width, "width")
        _require_positive_int(modes, "modes")
        _require_positive_int(n_layers, "n_layers")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.n_layers = n_layers
        hidden = width * 2
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.layers = nn.ModuleList([FourierLayer1d(width, modes) for _ in range(n_layers)])
        self.project_hidden = nn.Conv1d(width, hidden, kernel_size=1)
        self.project_out = nn.Conv1d(hidden, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("FNO1d expects (batch, channels, n_space)")
        hidden = self.lift(values)
        last = self.n_layers - 1
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < last:
                hidden = self.activation(hidden)
        hidden = self.activation(self.project_hidden(hidden))
        return self.project_out(hidden)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, int]:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "width": self.width,
            "modes": self.modes,
            "n_layers": self.n_layers,
        }


def fno_from_config(config: dict[str, object]) -> FNO1d:
    """Rebuild a network from the integers stored in a checkpoint."""

    return FNO1d(
        _require_positive_int(config.get("in_channels"), "in_channels"),
        _require_positive_int(config.get("out_channels"), "out_channels"),
        width=_require_positive_int(config.get("width"), "width"),
        modes=_require_positive_int(config.get("modes"), "modes"),
        n_layers=_require_positive_int(config.get("n_layers"), "n_layers"),
    )


def pack_inputs(inputs: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Stack normalized frames and a constant viscosity channel.

    ``inputs`` is ``(batch, frames, n_space)``. ``nu`` is ``(batch,)``,
    already normalized. The result is ``(batch, frames + 1, n_space)``.
    """

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (batch, frames, n_space)")
    if nu.ndim != 1 or nu.shape[0] != inputs.shape[0]:
        raise ValueError("nu must have shape (batch,)")
    channel = nu.to(dtype=inputs.dtype, device=inputs.device).view(-1, 1, 1).expand(-1, 1, inputs.shape[-1])
    return torch.cat((inputs, channel), dim=1)


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be an integer >= 1")
    return value
