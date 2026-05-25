"""API tests: GET /health returns 200; POST /predict returns a mask overlay.

Phase 6 — implemented alongside the FastAPI app. Skipped until then so the suite
stays green.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Phase 6: implement src/app/main.py first")


def test_health_returns_200() -> None:
    ...


def test_predict_returns_mask_for_sample_image() -> None:
    ...
