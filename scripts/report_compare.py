#!/usr/bin/env python3
"""베이스라인과 모델 결과를 비교해 마크다운 리포트를 만든다."""

import argparse
import json
import os

ADVANCED_METRICS = [
    ("mrr", "MRR", True),
    ("mean_min_rank", "평균 최소 실제순위", False),
    ("brier", "Brier", False),
    ("log_loss", "LogLoss", False),
    ("ece", "ECE", False),
]


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="베이스라인과 모델 메트릭을 비교합니다.")
    parser.add_argument(
        "--baseline-json",
        default="reports/baseline.json",
        help="베이스라인 JSON 경로.",
    )
    parser.add_argument(
        "--model-json",
        default="reports/model.json",
        help="모델 JSON 경로(여러 개면 쉼표로 구분).",
    )
    parser.add_argument(
        "--out-md",
        default="reports/compare.md",
        help="비교 리포트 Markdown 경로.",
    )
    parser.add_argument(
        "--title",
        default="베이스라인 대비 모델 성능 리포트",
        help="리포트 제목.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    """파일 저장 전 상위 디렉터리를 만든다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def load_json(path: str) -> dict:
    """JSON 파일을 읽어 딕셔너리로 반환한다."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def format_span(span: object) -> str:
    """회차 범위 값을 문자열로 변환한다."""
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return f"{span[0]}..{span[1]}"
    return ""


def parse_thresholds(raw: object) -> list[int]:
    """쉼표 구분 임계값 문자열을 정수 리스트로 변환한다."""
    if raw is None:
        return []
    return [int(value.strip()) for value in str(raw).split(",") if value.strip()]


def collect_thresholds(configs: list[dict], metrics_list: list[dict]) -> list[int]:
    """설정/메트릭에서 적중 임계값 목록을 수집한다."""
    ordered: list[int] = []
    seen: set[int] = set()

    for config in configs:
        for threshold in parse_thresholds(config.get("hit_thresholds")):
            if threshold not in seen:
                ordered.append(threshold)
                seen.add(threshold)

    for metrics in metrics_list:
        for key in metrics.get("hit_rates", {}).keys():
            try:
                threshold = int(key)
            except (TypeError, ValueError):
                continue
            if threshold not in seen:
                ordered.append(threshold)
                seen.add(threshold)

    return ordered or [1, 2, 3, 4, 5]


def metric_value(metrics: dict, key: str) -> float:
    """메트릭 값을 float으로 반환한다."""
    return float(metrics.get(key, 0.0))


def hit_rate(metrics: dict, threshold: int) -> float:
    """임계값별 적중률을 int/str 키 모두 지원해 읽는다."""
    hit_rates = metrics.get("hit_rates", {})
    value = hit_rates.get(threshold)
    if value is None:
        value = hit_rates.get(str(threshold), 0.0)
    return float(value)


def to_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """헤더/행 데이터를 마크다운 테이블 문자열로 변환한다."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def status_from_delta(delta: float, *, higher_is_better: bool) -> str:
    """증감값을 개선/악화/동일 상태 문자열로 변환한다."""
    if delta == 0:
        return "동일"
    improved = delta > 0 if higher_is_better else delta < 0
    return "개선" if improved else "악화"


def metric_display_name(key: str, threshold: int | None = None) -> str:
    """메트릭 키를 한국어 표시 이름으로 변환한다."""
    if key == "average_hits":
        return "평균 일치 개수"
    if key == "hit_rate" and threshold is not None:
        return f"Hit@{threshold} 적중률"
    return key


def composite_score(metrics: dict, thresholds: list[int]) -> float:
    """모델 우열 판정을 위한 종합 점수를 계산한다."""
    score = metric_value(metrics, "average_hits")
    weight_by_threshold = {1: 0.03, 2: 0.07, 3: 0.20, 4: 0.05, 5: 0.02}
    for threshold in thresholds:
        score += hit_rate(metrics, threshold) * weight_by_threshold.get(threshold, 0.01)

    score += metric_value(metrics, "mrr") * 0.15
    mean_min_rank = metric_value(metrics, "mean_min_rank")
    if mean_min_rank > 0:
        score += (1.0 / mean_min_rank) * 0.40
    score -= metric_value(metrics, "brier") * 0.10
    score -= metric_value(metrics, "log_loss") * 0.05
    score -= metric_value(metrics, "ece") * 0.10
    return score


def build_basic_rows(
    baseline_metrics: dict,
    random_metrics: dict,
    improvement_metrics: dict,
    model_rows: list[tuple[str, dict]],
    thresholds: list[int],
) -> list[list[str]]:
    """베이스라인 비교용 기본 지표 테이블 행을 만든다."""
    rows = []
    for label, metrics in [
        ("베이스라인", baseline_metrics),
        ("랜덤", random_metrics),
        ("랜덤 대비 개선율", improvement_metrics),
        *model_rows,
    ]:
        row = [label, f"{metric_value(metrics, 'average_hits'):.4f}"]
        for threshold in thresholds:
            row.append(f"{hit_rate(metrics, threshold):.4f}")
        rows.append(row)
    return rows


def main() -> int:
    """CLI 실행 진입점."""
    args = parse_args()
    if not os.path.exists(args.baseline_json):
        print(f"베이스라인 파일이 없습니다: {args.baseline_json}")
        return 1

    model_paths = [path.strip() for path in args.model_json.split(",") if path.strip()]
    for path in model_paths:
        if not os.path.exists(path):
            print(f"모델 파일이 없습니다: {path}")
            return 1

    baseline_payload = load_json(args.baseline_json)
    baseline_config = baseline_payload.get("config", {})
    baseline_metrics = baseline_payload.get("metrics", {}).get("baseline", {})
    random_metrics = baseline_payload.get("metrics", {}).get("random", {})
    improvement_metrics = baseline_payload.get("metrics", {}).get("improvement", {})

    model_payloads = [load_json(path) for path in model_paths]
    model_configs = [payload.get("config", {}) for payload in model_payloads]
    model_metrics_list = [payload.get("metrics", {}) for payload in model_payloads]
    thresholds = collect_thresholds(
        [baseline_config, *model_configs],
        [baseline_metrics, random_metrics, improvement_metrics, *model_metrics_list],
    )

    model_config_lines = []
    warning_lines = []
    model_rows: list[tuple[str, dict]] = []
    baseline_protocol = baseline_config.get("eval_protocol", "")

    for path, payload in zip(model_paths, model_payloads):
        config = payload.get("config", {})
        metrics = payload.get("metrics", {})
        label = config.get("model") or os.path.basename(path).split(".")[0]
        model_protocol = config.get("eval_protocol", "single_top6")
        model_rows.append((label, metrics))

        model_config_lines.extend(
            [
                f"- 모델 데이터 ({label}): `{config.get('data_path', '')}`",
                f"- 모델 학습 구간 ({label}): {format_span(config.get('train_span'))}",
                f"- 모델 평가 구간 ({label}): {format_span(config.get('test_span'))}",
                f"- 모델 평가 방식 ({label}): {model_protocol}",
                f"- 피처 개수 ({label}): {config.get('feature_count', '')}",
            ]
        )
        if baseline_protocol and model_protocol != baseline_protocol:
            warning_lines.append(
                f"- 평가 방식 불일치: baseline={baseline_protocol}, {label}={model_protocol}"
            )

    basic_headers = ["구분", "평균 일치 개수"] + [f"Hit@{t} 적중률" for t in thresholds]
    basic_rows = build_basic_rows(
        baseline_metrics,
        random_metrics,
        improvement_metrics,
        model_rows,
        thresholds,
    )

    summary_lines = []
    baseline_avg = metric_value(baseline_metrics, "average_hits")
    for label, metrics in model_rows:
        avg_delta = metric_value(metrics, "average_hits") - baseline_avg
        parts = [
            f"{metric_display_name('average_hits')} {avg_delta:+.4f} "
            f"({status_from_delta(avg_delta, higher_is_better=True)})"
        ]
        for threshold in thresholds:
            delta = hit_rate(metrics, threshold) - hit_rate(baseline_metrics, threshold)
            parts.append(
                f"{metric_display_name('hit_rate', threshold)} {delta:+.4f} "
                f"({status_from_delta(delta, higher_is_better=True)})"
            )
        for key, display_name, _ in ADVANCED_METRICS:
            if key in metrics:
                parts.append(f"{display_name} {metric_value(metrics, key):.6f}")
            else:
                parts.append(f"{display_name} 미제공")
        parts.append(f"종합점수 {composite_score(metrics, thresholds):.4f}")
        summary_lines.append(f"- {label}: " + ", ".join(parts))

    ranking_rows = []
    ranked_models = sorted(
        model_rows,
        key=lambda item: composite_score(item[1], thresholds),
        reverse=True,
    )
    for rank, (label, metrics) in enumerate(ranked_models, start=1):
        row = [
            str(rank),
            label,
            f"{composite_score(metrics, thresholds):.4f}",
            f"{metric_value(metrics, 'average_hits'):.4f}",
        ]
        for threshold in thresholds:
            row.append(f"{hit_rate(metrics, threshold):.4f}")
        for key, _, _ in ADVANCED_METRICS:
            if key in metrics:
                row.append(f"{metric_value(metrics, key):.6f}")
            else:
                row.append("N/A")
        ranking_rows.append(row)

    ranking_headers = (
        ["순위", "모델", "종합점수", "평균 일치 개수"]
        + [f"Hit@{t}" for t in thresholds]
        + [display for _, display, _ in ADVANCED_METRICS]
    )

    best_model_line = ""
    if ranked_models:
        best_name, best_metrics = ranked_models[0]
        best_model_line = (
            f"- 추천 모델: **{best_name}** "
            f"(종합점수 {composite_score(best_metrics, thresholds):.4f})"
        )

    config_lines = [
        f"- 베이스라인 데이터: `{baseline_config.get('data_path', '')}`",
        f"- 베이스라인 학습 구간: {format_span(baseline_config.get('train_span'))}",
        f"- 베이스라인 평가 구간: {format_span(baseline_config.get('test_span'))}",
        f"- 베이스라인 평가 방식: {baseline_protocol or '미지정'}",
        *model_config_lines,
    ]

    report: list[str] = [
        f"# {args.title}",
        "",
        "## 설정",
        *config_lines,
    ]
    if warning_lines:
        report.extend(["", "## 경고", *warning_lines])
    if summary_lines:
        report.extend(["", "## 요약", *summary_lines])
    if best_model_line:
        report.extend(["", "## 판정", best_model_line])
    report.extend(
        [
            "",
            "## 베이스라인 비교 지표",
            to_markdown_table(basic_headers, basic_rows),
            "",
            "## 모델 종합 순위",
            to_markdown_table(ranking_headers, ranking_rows),
            "",
        ]
    )

    ensure_parent_dir(args.out_md)
    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))

    print(f"리포트를 저장했습니다: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
