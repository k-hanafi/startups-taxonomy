"""Keep production, crawler, and wayback CSV contracts aligned."""

from tavily_crawler.master_csv import CLASSIFIER_INPUT_COLUMNS as CRAWLER_INPUT_COLUMNS
from tavily_crawler.master_csv import is_valid_homepage_url as crawler_url_ok
from two_pass_classifier.input_contract import SOURCE_COLUMNS
from wayback_machine.cohort import CLASSIFIER_INPUT_COLUMNS as WAYBACK_INPUT_COLUMNS
from wayback_machine.cohort import is_valid_homepage_url as wayback_url_ok

_URL_SAMPLES = (
    "https://example.com",
    "http://example.com/path",
    "example.com",
    "",
    None,
    "nan",
    "ftp://example.com",
)


def test_production_source_columns_match_crawler_output() -> None:
    assert list(SOURCE_COLUMNS) == list(CRAWLER_INPUT_COLUMNS)


def test_wayback_classifier_input_columns_match_production() -> None:
    assert list(WAYBACK_INPUT_COLUMNS) == list(SOURCE_COLUMNS)


def test_homepage_url_validators_agree() -> None:
    for value in _URL_SAMPLES:
        assert crawler_url_ok(value) == wayback_url_ok(value)
