"""Inference tests: output mask shape and value-range sanity.

Phase 6 — implemented alongside src/app/inference.py. Skipped until then so the
suite stays green.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Phase 6: implement src/app/inference.py first")


def test_mask_shape_matches_input() -> None:
    ...


def test_mask_values_are_binary() -> None:
    ...
