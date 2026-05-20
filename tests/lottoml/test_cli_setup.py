from pathlib import Path

from click.testing import CliRunner

from lottoml.cli import main


V1_SAMPLE = """draw_no,draw_date,n1,n2,n3,n4,n5,n6,bonus,total_sell_amount,first_prize_amount,first_prize_winner_count,second_prize_amount,second_prize_winner_count,third_prize_amount,third_prize_winner_count,first_accum_amount
1,2002-12-07,10,23,29,33,37,40,16,3681782000,0,0,143934100,1,5140500,28,863604600
2,2002-12-14,9,13,21,25,32,42,2,4904274000,2002006800,1,94866800,2,1842000,103,0
"""


def test_setup_migrates_v1_seed(tmp_path: Path, monkeypatch) -> None:
    v1_path = tmp_path / "data" / "processed" / "lotto_draws.csv"
    v1_path.parent.mkdir(parents=True)
    v1_path.write_text(V1_SAMPLE, encoding="utf-8")

    # discover_latest_draw returns 2 (matches seed) so no fetch needed
    monkeypatch.setattr(
        "lottoml.cli.discover_latest_draw", lambda **_: 2
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup", "--root", str(tmp_path), "--skip-train"],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data" / "draws.csv").exists()
    assert "2회까지" in result.output or "2 회까지" in result.output
