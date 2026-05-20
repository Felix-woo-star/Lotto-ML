"""lotto CLI entrypoint."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import click

from .data.fetch import backfill_range, discover_latest_draw, fetch_draw_urllib
from .data.migrate import migrate_v1_csv
from .data.portfolio import (
    PortfolioRecord,
    append_recommendation,
    load_portfolio_records,
    record_result,
)
from .data.storage import load_draws, save_draws
from .model.predict import predict_probabilities
from .model.train import train_ranker
from .reporting.history import render_history_markdown
from .reporting.scoring import classify, score_ticket
from .reporting.weekly import render_weekly_markdown
from .selection.hybrid import build_portfolio


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


@main.command()
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.cwd(),
    show_default=True,
)
@click.option("--pool-size", type=int, default=30, show_default=True)
@click.option("--time-limit", type=float, default=30.0, show_default=True, help="ILP 시간 한계 (초)")
def recommend(root: Path, pool_size: int, time_limit: float) -> None:
    """매주 1회 - 최신 fetch → 채점 → 다음 회차 추천."""
    draws_path = root / "data" / "draws.csv"
    portfolio_path = root / "data" / "portfolio.csv"
    model_path = root / "data" / "models" / "ranker.pkl"
    weekly_dir = root / "reports" / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    if not draws_path.exists():
        click.echo("[lotto] 데이터가 없습니다. `lotto setup`을 먼저 실행하세요.")
        raise SystemExit(1)
    if not model_path.exists():
        click.echo("[lotto] 모델이 없습니다. `lotto setup`을 먼저 실행하세요.")
        raise SystemExit(1)

    existing = load_draws(draws_path)
    last_known = existing[-1].draw_no
    click.echo("[lotto] 최신 회차 확인 중...")
    latest = discover_latest_draw()
    if latest > last_known:
        for draw_no in range(last_known + 1, latest + 1):
            try:
                new_draw = fetch_draw_urllib(draw_no)
                existing.append(new_draw)
                click.echo(f"        {draw_no}회 수집됨")
            except Exception as exc:  # noqa: BLE001
                click.echo(f"        {draw_no}회 수집 실패: {exc}")
        save_draws(draws_path, existing)

    # 채점: portfolio.csv 에서 미평가 + draws.csv 에 존재하는 회차 매칭
    draws_by_no = {d.draw_no: d for d in existing}
    records = load_portfolio_records(portfolio_path) if portfolio_path.exists() else []
    pending = [r for r in records if r.hits is None and r.draw_no in draws_by_no]
    if pending:
        click.echo("\n📊 미평가 추천 채점")
        scored_draws: dict[int, list] = {}
        for record in pending:
            draw = draws_by_no[record.draw_no]
            hits, tier = classify(record.numbers, draw.numbers, draw.bonus)
            actuals = {
                "first_prize": draw.first_prize,
                "second_prize": draw.second_prize,
                "third_prize": draw.third_prize,
            }
            payout = score_ticket(tier=tier, actuals=actuals)
            record_result(
                portfolio_path,
                draw_no=record.draw_no, ticket_idx=record.ticket_idx,
                actual_numbers=draw.numbers, actual_bonus=draw.bonus,
                hits=hits, tier=tier, payout=payout,
                recorded_at=dt.datetime.now(),
            )
            scored_draws.setdefault(draw.draw_no, []).append(
                {
                    "idx": record.ticket_idx,
                    "source": record.source,
                    "numbers": record.numbers,
                    "hits": hits,
                    "tier": tier,
                    "payout": payout,
                }
            )
        from .reporting.weekly import (
            render_weekly_results_section,
            update_weekly_markdown_with_results,
        )

        for draw_no, rows in scored_draws.items():
            d = draws_by_no[draw_no]
            click.echo(f"   {draw_no}회 당첨번호: {' '.join(str(n) for n in d.numbers)} (보너스 {d.bonus})")
            total_payout = 0
            for row in rows:
                nums = " ".join(str(n) for n in row["numbers"])
                tag = f"{row['tier']}등" if row["tier"] else f"{row['hits']}개 적중"
                click.echo(f"   {row['idx']}번 ({row['source']}): {nums} → {tag} ({row['payout']:,}원)")
                total_payout += row["payout"]
            click.echo(f"   회차 손익: {total_payout - 5000:+,}원")

            md_path = weekly_dir / f"{draw_no}.md"
            if md_path.exists():
                section = render_weekly_results_section(
                    winning=d.numbers,
                    bonus=d.bonus,
                    per_ticket=rows,
                    total_payout=total_payout,
                )
                update_weekly_markdown_with_results(
                    md_path, section, draw_date=d.draw_date.isoformat()
                )
                click.echo(f"   📝 {md_path.relative_to(root)} 결과 갱신됨")

    # 다음 회차 추천
    next_draw = existing[-1].draw_no + 1
    probs = predict_probabilities(existing, model_path=model_path)
    portfolio = build_portfolio(
        probs, seed=next_draw, pool_size=pool_size, time_limit_seconds=time_limit
    )
    ordered = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    pool = sorted(n for n, _ in ordered[:pool_size])

    click.echo(f"\n🎯 다음 회차({next_draw}) 추천")
    for entry in portfolio.tickets:
        nums = " ".join(str(n) for n in entry.numbers)
        click.echo(f"   {entry.ticket_idx}번 ({entry.source}): {nums}")
    if portfolio.covering.fallback_used:
        click.echo("   ⚠️  ILP 시간초과로 greedy fallback 사용")

    md = render_weekly_markdown(
        draw_no=next_draw,
        draw_date="TBD",
        portfolio=portfolio, probs=probs, pool=pool,
    )
    md_path = weekly_dir / f"{next_draw}.md"
    md_path.write_text(md, encoding="utf-8")
    click.echo(f"\n   📝 {md_path.relative_to(root)} 저장됨")

    now = dt.datetime.now()
    existing_records = {
        (r.draw_no, r.ticket_idx) for r in load_portfolio_records(portfolio_path)
    }
    for entry in portfolio.tickets:
        if (next_draw, entry.ticket_idx) in existing_records:
            continue
        append_recommendation(
            portfolio_path,
            PortfolioRecord(
                draw_no=next_draw,
                ticket_idx=entry.ticket_idx,
                numbers=entry.numbers,
                source=entry.source,
                recommended_at=now,
            ),
        )


@main.command()
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.cwd(),
    show_default=True,
)
def report(root: Path) -> None:
    """누적 통계 markdown 생성."""
    portfolio_path = root / "data" / "portfolio.csv"
    out_path = root / "reports" / "history.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_portfolio_records(portfolio_path) if portfolio_path.exists() else []
    md = render_history_markdown(records)
    out_path.write_text(md, encoding="utf-8")
    click.echo(md)
    click.echo(f"\n📝 {out_path.relative_to(root)} 저장됨")


@main.command()
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.cwd(),
    show_default=True,
)
def status(root: Path) -> None:
    """데이터 / 모델 / 추천 대기 상태."""
    draws_path = root / "data" / "draws.csv"
    model_meta = root / "data" / "models" / "ranker_meta.json"
    portfolio_path = root / "data" / "portfolio.csv"

    if not draws_path.exists():
        click.echo("데이터:     없음 (`lotto setup` 필요)")
    else:
        draws = load_draws(draws_path)
        click.echo(
            f"데이터:     1~{draws[-1].draw_no}회 "
            f"(마지막 추첨일 {draws[-1].draw_date.isoformat()})"
        )

    if not model_meta.exists():
        click.echo("모델:       없음 (`lotto setup` 또는 `lotto retrain` 필요)")
    else:
        import json
        meta = json.loads(model_meta.read_text(encoding="utf-8"))
        train_end = meta["train_draws"][1]
        click.echo(f"모델:       학습 회차 1~{train_end}")

    if portfolio_path.exists():
        records = load_portfolio_records(portfolio_path)
        pending = [r for r in records if r.hits is None]
        if pending:
            draw_nos = sorted({r.draw_no for r in pending})
            click.echo(f"미평가 추천: {draw_nos}")
        else:
            click.echo("미평가 추천: 없음")
    else:
        click.echo("추천 이력:  없음")


if __name__ == "__main__":
    sys.exit(main())
