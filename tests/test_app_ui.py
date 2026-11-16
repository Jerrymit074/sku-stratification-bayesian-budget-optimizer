"""Headless UI smoke tests using Streamlit AppTest across all 5 pages and both languages."""

from pathlib import Path
from streamlit.testing.v1 import AppTest


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_all_pages_in_english_and_chinese():
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # Sidebar has two radio widgets: [0] language, [1] page navigation
    for lang_choice in ["English", "简体中文"]:
        at.sidebar.radio[0].set_value(lang_choice).run()
        assert not at.exception
        for page_key in ["overview", "clustering", "bayesian", "allocator", "monitor"]:
            at.sidebar.radio[1].set_value(page_key).run()
            assert not at.exception, f"Exception on page={page_key}, lang={lang_choice}: {at.exception}"
