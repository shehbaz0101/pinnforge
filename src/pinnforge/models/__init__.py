"""PINN multilayer perceptron.

Importing this package requires torch (the optional ``ml`` extra). The
package root does not import it.
"""

from pinnforge.models.mlp import ACTIVATIONS, MLP, input_features, mlp_from_spec

__all__ = [
    "ACTIVATIONS",
    "MLP",
    "input_features",
    "mlp_from_spec",
]
