"""Pneumonia chest X-ray classification pipeline.

Rebuilt to actually match the original course report's claims: trained and
evaluated at the scale the report describes (thousands of images, not
dozens), with rigor the original notebook skipped (proper held-out test set,
multi-seed evaluation, confidence intervals).
"""

__version__ = "0.1.0"
