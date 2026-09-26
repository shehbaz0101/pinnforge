"""Fully connected network used as a PINN field.

The input width is the number of independent variables: ``t`` for the
harmonic oscillator, ``(x, t)`` for Burgers, and ``x`` or ``(x, y)`` for
the Poisson toy. The output is the scalar field ``u``. ``tanh`` is the
default activation because its second derivative stays smooth, which is
what the strong-form residual operators differentiate. ``relu`` is
rejected: its second derivative is zero almost everywhere, so a
strong-form second-derivative residual does not see the activation.
"""

from __future__ import annotations

from collections.abc import Sequence

from pinnforge.equations.base import EquationSpec
from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonToySpec
from pinnforge.ml_import import require_torch

torch = require_torch()
nn = torch.nn

_ACTIVATION_TYPES: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}

_RELU_ERROR = (
    "activation 'relu' is not valid for strong-form second-derivative residuals; "
    "its second derivative is zero almost everywhere. Use tanh or silu"
)

ACTIVATIONS: tuple[str, ...] = tuple(sorted(_ACTIVATION_TYPES))


def input_features(spec: EquationSpec) -> int:
    """Number of collocation columns for ``spec``.

    Raises:
        TypeError: ``spec`` is not one of the three Day 1 equation specs.
    """

    _require_supported_spec(spec)
    return len(spec.collocation_domain().axes)


def mlp_from_spec(
    spec: EquationSpec,
    hidden_widths: Sequence[int] = (32, 32),
    *,
    activation: str = "tanh",
    out_features: int = 1,
) -> MLP:
    """Build an :class:`MLP` whose input width matches ``spec``."""

    return MLP(
        input_features(spec),
        hidden_widths,
        activation=activation,
        out_features=out_features,
    )


class MLP(nn.Module):
    """Configurable multilayer perceptron with a scalar field output.

    Hidden layers use ``activation``. The output layer is linear. Widths
    are positive integers. An empty ``hidden_widths`` is a single linear
    map, which is valid but a weak PINN (its second derivatives vanish).
    """

    def __init__(
        self,
        in_features: int,
        hidden_widths: Sequence[int] = (32, 32),
        *,
        activation: str = "tanh",
        out_features: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = _positive_int(in_features, label="in_features")
        self.out_features = _positive_int(out_features, label="out_features")
        self.hidden_widths = _hidden_widths(hidden_widths)
        self.activation = _activation_name(activation)
        widths = (self.in_features, *self.hidden_widths, self.out_features)
        blocks: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths, widths[1:])):
            blocks.append(nn.Linear(left, right))
            if index < len(widths) - 2:
                blocks.append(_ACTIVATION_TYPES[self.activation]())
        self.network = nn.Sequential(*blocks)
        self.apply(_init_linear)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate ``u`` at ``coords`` of shape ``(n, in_features)``.

        A vector of shape ``(in_features,)`` is treated as one point.
        The result has shape ``(n, out_features)``.
        """

        if not isinstance(coords, torch.Tensor):
            raise TypeError("coords must be a torch.Tensor")
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)
        if coords.ndim != 2:
            raise ValueError(
                f"coords must have shape (n, {self.in_features}), got {tuple(coords.shape)}"
            )
        if coords.shape[1] != self.in_features:
            raise ValueError(
                f"expected {self.in_features} input features, got {coords.shape[1]}"
            )
        return self.network(coords)


def _require_supported_spec(spec: EquationSpec) -> None:
    if not isinstance(spec, (HarmonicOscillatorSpec, Burgers1DSpec, PoissonToySpec)):
        raise TypeError("spec must be a HarmonicOscillatorSpec, Burgers1DSpec, or PoissonToySpec")


def _positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _hidden_widths(widths: Sequence[int]) -> tuple[int, ...]:
    values = tuple(widths)
    for width in values:
        _positive_int(width, label="hidden width")
    return values


def _activation_name(name: str) -> str:
    if name == "relu":
        raise ValueError(_RELU_ERROR)
    if not isinstance(name, str) or name not in _ACTIVATION_TYPES:
        known = ", ".join(ACTIVATIONS)
        raise ValueError(f"unknown activation {name!r}; known: {known}")
    return name


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
