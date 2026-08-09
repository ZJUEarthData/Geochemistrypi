import io
import sys

from geochemistrypi_mcp.lifecycle.console import configure_utf8_console


def test_lifecycle_console_handles_unicode_paths_under_a_legacy_code_page(
    monkeypatch,
) -> None:
    raw = io.BytesIO()
    legacy_stdout = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", legacy_stdout)

    configure_utf8_console()
    print("D:/GeochemistryPi/用户数据")
    legacy_stdout.flush()

    assert raw.getvalue().decode("utf-8").splitlines() == [
        "D:/GeochemistryPi/用户数据"
    ]
