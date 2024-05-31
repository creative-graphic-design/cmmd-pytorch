import pathlib

import pytest

from cmmd.cmmd_score import compute_cmmd
from cmmd.testing import CmmdTestCase


class TestComputeCmmd(CmmdTestCase):
    @pytest.fixture
    def ref_dir(self) -> pathlib.Path:
        return self.FIXTURES_ROOT / "reference_images"

    @pytest.fixture
    def eval_dir(self) -> pathlib.Path:
        return self.FIXTURES_ROOT / "generated_images"

    def test_compute_cmmd(self, ref_dir: pathlib.Path, eval_dir: pathlib.Path):
        cmmd_score = compute_cmmd(ref_dir=ref_dir, eval_dir=eval_dir, batch_size=1)
        assert cmmd_score == pytest.approx(7.696, 0.0001)
