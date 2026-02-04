"""Circle Packing problem for MangoEvolve.

Pack N circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.

This is the original benchmark problem from AlphaEvolve. MangoEvolve has achieved
a score of 2.6359850561 with 26 circles, exceeding the DeepMind benchmark of 2.635.
"""

from .evaluator import CirclePackingEvaluator

__all__ = ["CirclePackingEvaluator"]
