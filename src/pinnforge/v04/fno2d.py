"""2D Fourier neural operator for v0.4 windows."""

from __future__ import annotations

from pinnforge.ml_import import require_torch

torch = require_torch()
nn = torch.nn


class SpectralConv2d(nn.Module):
    """Multiply the lowest ``modes x modes`` rFFT2 coefficients."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        _positive(in_channels, "in_channels")
        _positive(out_channels, "out_channels")
        _positive(modes, "modes")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight1_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))
        self.weight1_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))
        self.weight2_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))
        self.weight2_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("spectral conv expects (batch, channels, n, n)")
        batch, channels, height, width = values.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {channels}")
        spectrum = torch.fft.rfft2(values, dim=(-2, -1))
        used_y = min(self.modes, height)
        used_x = min(self.modes, spectrum.shape[-1])
        out = torch.zeros(
            batch,
            self.out_channels,
            height,
            spectrum.shape[-1],
            dtype=torch.complex64,
            device=values.device,
        )
        w1 = torch.complex(self.weight1_real[:, :, :used_y, :used_x], self.weight1_imag[:, :, :used_y, :used_x])
        w2 = torch.complex(self.weight2_real[:, :, :used_y, :used_x], self.weight2_imag[:, :, :used_y, :used_x])
        out[:, :, :used_y, :used_x] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, :used_y, :used_x], w1)
        out[:, :, -used_y:, :used_x] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, -used_y:, :used_x], w2)
        return torch.fft.irfft2(out, s=(height, width), dim=(-2, -1))


class FourierLayer2d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.local = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.spectral(values) + self.local(values)


class FNO2d(nn.Module):
    """Map ``(batch, in_channels, n, n)`` to ``(batch, out_channels, n, n)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        width: int = 12,
        modes: int = 6,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = _positive(in_channels, "in_channels")
        self.out_channels = _positive(out_channels, "out_channels")
        self.width = _positive(width, "width")
        self.modes = _positive(modes, "modes")
        self.n_layers = _positive(n_layers, "n_layers")
        hidden = width * 2
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.layers = nn.ModuleList([FourierLayer2d(width, modes) for _ in range(n_layers)])
        self.project_hidden = nn.Conv2d(width, hidden, kernel_size=1)
        self.project_out = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("FNO2d expects (batch, channels, n, n)")
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


def pack_inputs(frames: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """``frames`` is ``(batch, frames, components, n, n)``; ``nu`` is ``(batch,)``."""

    if frames.ndim != 5:
        raise ValueError("frames must have shape (batch, frames, components, n, n)")
    if nu.ndim != 1 or nu.shape[0] != frames.shape[0]:
        raise ValueError("nu must have shape (batch,)")
    batch, n_frames, n_comp, height, width = frames.shape
    stacked = frames.reshape(batch, n_frames * n_comp, height, width)
    channel = nu.to(dtype=frames.dtype, device=frames.device).view(batch, 1, 1, 1).expand(batch, 1, height, width)
    return torch.cat((stacked, channel), dim=1)


def unpack_outputs(values: torch.Tensor, n_frames: int, n_comp: int) -> torch.Tensor:
    batch, channels, height, width = values.shape
    if channels != n_frames * n_comp:
        raise ValueError("output channels do not match frames * components")
    return values.reshape(batch, n_frames, n_comp, height, width)


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be an integer >= 1")
    return value
