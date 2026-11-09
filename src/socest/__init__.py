"""socest: physics-based and data-driven state-of-charge estimation for Li-ion cells.

Sign convention used everywhere inside this package:
    current > 0  means DISCHARGE (current leaving the cell)
    current < 0  means CHARGE
Loaders convert raw tester data to this convention.
"""

__version__ = "0.1.0"
