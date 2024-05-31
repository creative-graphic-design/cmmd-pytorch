import pathlib


class CmmdTestCase(object):
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
    MODULE_ROOT = PROJECT_ROOT / "cmmd"
    TEST_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
