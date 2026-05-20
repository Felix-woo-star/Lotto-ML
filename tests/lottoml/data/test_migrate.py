from pathlib import Path

from lottoml.data.migrate import migrate_v1_csv
from lottoml.data.storage import load_draws


V1_CSV_SAMPLE = """draw_no,draw_date,n1,n2,n3,n4,n5,n6,bonus,total_sell_amount,first_prize_amount,first_prize_winner_count,second_prize_amount,second_prize_winner_count,third_prize_amount,third_prize_winner_count,first_accum_amount
1,2002-12-07,10,23,29,33,37,40,16,3681782000,0,0,143934100,1,5140500,28,863604600
2,2002-12-14,9,13,21,25,32,42,2,4904274000,2002006800,1,94866800,2,1842000,103,0
"""


def test_migrate_v1_csv(tmp_path: Path) -> None:
    v1_path = tmp_path / "lotto_draws.csv"
    v1_path.write_text(V1_CSV_SAMPLE, encoding="utf-8")
    v2_path = tmp_path / "draws.csv"

    migrate_v1_csv(v1_path, v2_path)

    draws = load_draws(v2_path)
    assert len(draws) == 2
    assert draws[0].draw_no == 1
    assert draws[0].numbers == (10, 23, 29, 33, 37, 40)
    assert draws[0].second_prize == 143934100
    assert draws[1].first_prize == 2002006800
