"""Claim-to-evidence checks for user-facing PINNForge text.

``python -m pinnforge.claims check`` loads ``docs/claims.yaml``, reads the
committed score files, and fails when a published number, comparison, or
superiority phrase is not supported. Importing this package does not import
torch.
"""

from pinnforge.claims.check import Failure, check

__all__ = ["Failure", "check"]
