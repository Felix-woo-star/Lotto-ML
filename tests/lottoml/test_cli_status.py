from pathlib import Path

from click.testing import CliRunner

from lottoml.cli import main


def test_status_handles_empty_state(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "데이터:" in result.output
    assert "모델:" in result.output


def test_report_with_empty_portfolio(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["report", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "이력 없음" in result.output
