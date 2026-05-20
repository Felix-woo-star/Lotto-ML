"""lotto CLI entrypoint."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from .data.fetch import backfill_range, discover_latest_draw
from .data.migrate import migrate_v1_csv
from .data.storage import load_draws, save_draws
from .model.train import train_ranker


@click.group()
def main() -> None:
    """주 5,000원 한국 로또 추천 시스템."""


@main.command()
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.cwd(),
    show_default=True,
    help="프로젝트 루트 경로",
)
@click.option(
    "--skip-train", is_flag=True, default=False, help="모델 학습 단계를 건너뛴다 (M1까지 테스트용)"
)
def setup(root: Path, skip_train: bool) -> None:
    """최초 1회 데이터 backfill 및 모델 학습."""
    draws_path = root / "data" / "draws.csv"
    v1_path = root / "data" / "processed" / "lotto_draws.csv"

    seed_count = 0
    if not draws_path.exists() and v1_path.exists():
        click.echo("[lotto] v1 데이터를 시드로 사용합니다")
        seed_count = migrate_v1_csv(v1_path, draws_path)
        click.echo(f"        {seed_count}회차 마이그레이션 완료")

    existing = load_draws(draws_path)
    last_known = existing[-1].draw_no if existing else 0

    click.echo("[lotto] 최신 회차 탐색 중...")
    latest = discover_latest_draw()
    click.echo(f"        최신 회차: {latest}")

    if latest > last_known:
        click.echo(f"[lotto] {last_known + 1}~{latest} 회차 수집")
        new_draws = backfill_range(
            start=last_known + 1,
            end=latest,
            on_progress=lambda current, total: click.echo(
                f"        {current}/{total}", err=True
            ),
        )
        all_draws = existing + new_draws
        save_draws(draws_path, all_draws)
        click.echo(f"[lotto] {latest}회까지 저장됨")
    else:
        click.echo(f"[lotto] {last_known}회까지 이미 보유")

    if skip_train:
        click.echo("[lotto] 학습 단계 건너뜀 (--skip-train)")
        return

    out_model = root / "data" / "models" / "ranker.pkl"
    out_meta = root / "data" / "models" / "ranker_meta.json"
    all_draws = load_draws(draws_path)
    click.echo("[lotto] 모델 학습 중...")
    metrics = train_ranker(all_draws, out_model=out_model, out_meta=out_meta)
    click.echo(
        f"        Brier: {metrics['brier']:.4f}  "
        f"ECE: {metrics['ece']:.4f}  "
        f"Top6: {metrics['top6_hit']:.4f}"
    )
    click.echo("[lotto] 준비 완료. `lotto recommend`로 시작하세요.")


@main.command()
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.cwd(),
    show_default=True,
)
@click.option("--holdout", type=int, default=50, show_default=True)
@click.option("--yes", "-y", is_flag=True, help="확인 프롬프트 건너뛰기")
def retrain(root: Path, holdout: int, yes: bool) -> None:
    """LightGBM ranker 재학습."""
    draws_path = root / "data" / "draws.csv"
    if not draws_path.exists():
        click.echo(f"[lotto] 데이터 파일이 없습니다: {draws_path}")
        raise SystemExit(1)

    draws = load_draws(draws_path)
    if len(draws) <= 100 + holdout:
        click.echo(f"[lotto] 학습 데이터가 부족합니다 ({len(draws)}회)")
        raise SystemExit(1)

    out_model = root / "data" / "models" / "ranker.pkl"
    out_meta = root / "data" / "models" / "ranker_meta.json"

    click.echo(f"[lotto] 학습 시작 ({len(draws)}회, holdout={holdout})")
    metrics = train_ranker(draws, out_model=out_model, out_meta=out_meta, holdout=holdout)
    click.echo(
        f"        Brier: {metrics['brier']:.4f}  "
        f"ECE: {metrics['ece']:.4f}  "
        f"Top6: {metrics['top6_hit']:.4f}"
    )

    if not yes:
        if not click.confirm("새 모델로 교체하시겠습니까?", default=True):
            click.echo("[lotto] 교체 취소됨")
            return
    click.echo(f"[lotto] 모델 저장: {out_model}")


if __name__ == "__main__":
    sys.exit(main())
