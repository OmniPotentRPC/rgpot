#!/usr/bin/env python3
"""Export asv results and apply a per-benchmark gate.

Each calibration run writes asv results into its own directory. The
exporter keeps exactly one result for the requested commit, machine,
and environment. The PR gate compares those saved samples with each
benchmark's own factor. A regression is a head/base median ratio above
that factor.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import statistics
from pathlib import Path

OUT = Path("benchflow/out")
MACHINE_NAME = "bench-runner"
CANONICAL = {
    "arch": "x86_64",
    "cpu": "unspecified",
    "machine": MACHINE_NAME,
    "num_cpu": "1",
    "os": "Linux",
    "ram": "unspecified",
}
FLOOR = 0.05
SPREAD_MULTIPLIER = 3.0
MIN_SAMPLES = 10
MIN_RUNS = 10
RULE = (
    "relative_threshold = max(0.05, 3 * MAD, the 99th percentile of "
    "|ratio - 1|, the maximum |ratio - 1|) of the ratios between two "
    "separate asv runs of the same commit; factor = 1 + relative_threshold"
)


def runs_dir() -> Path:
    return OUT / "runs"


def memory_runs_dir() -> Path:
    return OUT / "memory-runs"


def clean_label(label):
    if isinstance(label, str) and len(label) >= 2 and label[0] == label[-1] and label[0] in "'\"":
        return label[1:-1]
    return label


def walk(param_axes, value, samples):
    """Expand asv's flat parameter vector.

    asv stores one value per parameter combination in a flat list whose
    length is the product of the axes, in the same order as
    itertools.product.
    """
    if not param_axes:
        if isinstance(value, list) and len(value) == 1 and not isinstance(value[0], list):
            value = value[0]
        if (
            isinstance(samples, list)
            and len(samples) == 1
            and isinstance(samples[0], list)
        ):
            samples = samples[0]
        yield [], value, samples
        return
    axes = [list(axis) for axis in param_axes]
    count = 1
    for axis in axes:
        count *= len(axis)
    if isinstance(value, list) and len(value) == count and (
        samples is None or (isinstance(samples, list) and len(samples) == count)
    ):
        labels = list(itertools.product(*axes))
        for index, label in enumerate(labels):
            samp = None if samples is None else samples[index]
            yield [clean_label(item) for item in label], value[index], samp
        return
    axis, *rest = axes
    if not isinstance(value, list) or len(value) != len(axis):
        yield [], value, samples
        return
    for index, label in enumerate(axis):
        child = None if samples is None else samples[index]
        for tail, item, samp in walk(rest, value[index], child):
            yield [clean_label(label), *tail], item, samp


def flatten_samples(samples):
    if samples is None:
        return []
    if isinstance(samples, (int, float)) and not isinstance(samples, bool):
        return [float(samples)]
    if isinstance(samples, list):
        flat = []
        for item in samples:
            flat.extend(flatten_samples(item))
        return flat
    return []


def keep_key(key: str, kind: str) -> bool:
    if kind == "time":
        return "time_" in key and "ray_" not in key
    if kind == "memory":
        return "ray_" in key
    raise SystemExit(f"unknown kind {kind}")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def mad(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def relative_threshold(ratios: list[float]) -> dict:
    """Gate that is at least as wide as every calibration ratio."""
    if not ratios:
        raise SystemExit("no ratios")
    abs_dev = [abs(ratio - 1.0) for ratio in ratios]
    scale = mad(ratios)
    three = SPREAD_MULTIPLIER * scale
    p99 = percentile(abs_dev, 99)
    worst = max(abs_dev)
    relative = max(FLOOR, three, p99, worst)
    if relative < worst:
        raise SystemExit("gate narrower than measured A/A")
    return {
        "factor": 1.0 + relative,
        "floor": FLOOR,
        "mad": scale,
        "max_abs_ratio_minus_1": worst,
        "p99_abs_ratio_minus_1": p99,
        "relative_threshold": relative,
        "three_mad": three,
    }


def machine_file_path() -> Path:
    return Path(os.environ["HOME"]) / ".asv-machine.json"


def assert_selected_record(record: dict, name: str) -> None:
    """Check one machine record. Other records in the collection are ignored."""
    if not isinstance(record, dict):
        raise SystemExit("selected machine record is not an object")
    if record.get("machine") != name:
        raise SystemExit(
            f"selected machine record is {record.get('machine')!r}, expected {name}"
        )
    node = platform.node()
    if node and node in json.dumps(record, sort_keys=True):
        raise SystemExit("selected machine record contains the node name")


def _write_machine(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["version"] = 1
    payload[MACHINE_NAME] = dict(CANONICAL)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare_machine(path: Path | None = None) -> None:
    path = path or machine_file_path()
    data: dict = {}
    if path.is_file():
        data = json.loads(path.read_text())
    _write_machine(path, data)


def check_machine(path: Path | None = None) -> None:
    path = path or machine_file_path()
    if not path.is_file():
        raise SystemExit(f"machine file is missing: {path.name}")
    data = json.loads(path.read_text())
    _write_machine(path, data)
    saved = json.loads(path.read_text())
    if MACHINE_NAME not in saved:
        raise SystemExit(f"machine file has no {MACHINE_NAME} record")
    assert_selected_record(saved[MACHINE_NAME], MACHINE_NAME)
    if saved[MACHINE_NAME] != CANONICAL:
        raise SystemExit("selected machine record is not the canonical record")


def derive_conf(src: Path, dest: Path, results_dir: str) -> None:
    data = json.loads(Path(src).read_text())
    data["results_dir"] = results_dir
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=4) + "\n")


def _load_result_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"cannot parse {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path.name} is not a result object")
    return data


def select_result(results_dir: Path, commit: str, machine: str, env_name: str | None = None):
    """Return the one result for this commit, machine, and environment.

    Zero matches and more than one match are errors. A different commit
    or a different machine directory is not a match.
    """
    folder = Path(results_dir) / machine
    if not folder.is_dir():
        raise SystemExit(f"missing results for machine {machine}")
    matches = []
    for path in sorted(folder.glob("*.json")):
        if path.name == "machine.json":
            continue
        data = _load_result_json(path)
        params = data.get("params") or {}
        got_machine = params.get("machine")
        got_commit = data.get("commit_hash")
        got_env = data.get("env_name")
        if got_commit != commit or got_machine != machine:
            continue
        if env_name is not None and got_env != env_name:
            continue
        matches.append((path, data))
    if env_name is None:
        envs = {data.get("env_name") for _, data in matches}
        if len(envs) != 1:
            found = ", ".join(sorted(str(item) for item in envs)) or "none"
            raise SystemExit(
                f"expected one environment for commit {commit[:12]} on {machine}, found {found}"
            )
    if len(matches) != 1:
        names = ", ".join(path.name for path, _ in matches) or "none"
        label = env_name if env_name is not None else matches[0][1].get("env_name") if matches else "*"
        raise SystemExit(
            f"expected one result for commit {commit[:12]} machine {machine} "
            f"env {label}, found {len(matches)} ({names})"
        )
    path, data = matches[0]
    assert_selected_record(data.get("params") or {}, machine)
    return path, data


def rows_from_file(path: Path, kind: str, machine: str):
    from asv.results import Results

    result = Results.load(str(path), machine)
    rows = []
    for key in sorted(result._results):
        if not keep_key(key, kind):
            continue
        params = result.get_result_params(key)
        values = result.get_result_value(key, params)
        samples = result._samples.get(key)
        for labels, value, samp in walk(params, values, samples):
            rows.append(
                {
                    "commit": result.commit_hash,
                    "env_name": result.env_name,
                    "machine": result.params.get("machine"),
                    "name": key,
                    "params": labels,
                    "samples": flatten_samples(samp),
                    "value": value,
                }
            )
    rows.sort(key=lambda row: (row["name"], [str(item) for item in row["params"]]))
    return rows


def pair_ratio(earlier_samples: list[float], later_samples: list[float]) -> dict:
    """Ratio of two independent asv runs of one commit.

    The later median is divided by the earlier median. The two halves of
    one sample list are not a calibration ratio.
    """
    if not earlier_samples or not later_samples:
        raise SystemExit("a calibration ratio needs two non-empty runs")
    earlier = statistics.median(earlier_samples)
    later = statistics.median(later_samples)
    if earlier == 0:
        raise SystemExit("zero earlier median")
    return {
        "earlier_median": earlier,
        "later_median": later,
        "n_samples": len(earlier_samples) + len(later_samples),
        "ratio": later / earlier,
        "seconds": statistics.median([*earlier_samples, *later_samples]),
    }


def save_pair(
    index: int,
    earlier_dir: Path,
    later_dir: Path,
    commit: str,
    machine: str,
) -> None:
    earlier_path, _earlier_meta = select_result(earlier_dir, commit, machine)
    later_path, _later_meta = select_result(later_dir, commit, machine)
    earlier_rows = rows_from_file(earlier_path, "time", machine)
    later_rows = rows_from_file(later_path, "time", machine)
    later_by = {row_key(row): row for row in later_rows}
    saved = []
    for row in earlier_rows:
        if row["commit"] != commit or row["machine"] != machine:
            raise SystemExit("selected timing row does not match the requested commit and machine")
        other = later_by.get(row_key(row))
        if other is None:
            raise SystemExit(f"second run is missing {row['name']} {row['params']}")
        if other["commit"] != commit or other["machine"] != machine:
            raise SystemExit("second timing row does not match the requested commit and machine")
        try:
            stats = pair_ratio(row["samples"], other["samples"])
        except SystemExit as exc:
            raise SystemExit(f"{row['name']} {row['params']}: {exc}") from exc
        saved.append(
            {
                "commit": row["commit"],
                "env_name": row["env_name"],
                "machine": row["machine"],
                "name": row["name"],
                "params": row["params"],
                "samples": [*row["samples"], *other["samples"]],
                **stats,
            }
        )
    if not saved:
        raise SystemExit("no timing rows in the selected result")
    if len(later_by) != len(saved):
        raise SystemExit("the two calibration runs do not cover the same benchmarks")
    envs = {row["env_name"] for row in saved}
    if len(envs) != 1:
        raise SystemExit("selected timing rows span more than one environment")
    write_json(runs_dir() / f"{index:02d}.json", saved)


def aggregate() -> None:
    files = sorted(runs_dir().glob("*.json"))
    if len(files) < MIN_RUNS:
        raise SystemExit(f"{len(files)} runs, need {MIN_RUNS}")
    grouped: dict[tuple, list] = {}
    for path in files:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise SystemExit(f"{path.name} is not a run list")
        for row in payload:
            grouped.setdefault((row["name"], tuple(row["params"])), []).append(row)
    timing = []
    noise = []
    thresholds = []
    for key in sorted(grouped, key=lambda item: (item[0], [str(part) for part in item[1]])):
        rows = grouped[key]
        if len(rows) < MIN_RUNS:
            raise SystemExit(f"{key[0]} {list(key[1])}: {len(rows)} runs, need {MIN_RUNS}")
        commits = {row["commit"] for row in rows}
        machines = {row["machine"] for row in rows}
        envs = {row["env_name"] for row in rows}
        if len(commits) != 1 or len(machines) != 1 or len(envs) != 1:
            raise SystemExit(f"{key[0]} {list(key[1])}: mixed commit, machine, or environment")
        ratios = [float(row["ratio"]) for row in rows]
        samples = [sample for row in rows for sample in row["samples"]]
        stats = relative_threshold(ratios)
        name, params = key
        timing.append(
            {
                "commit": rows[-1]["commit"],
                "env_name": rows[-1]["env_name"],
                "machine": rows[-1]["machine"],
                "n_runs": len(rows),
                "n_samples": len(samples),
                "name": name,
                "params": list(params),
                "seconds": statistics.median(samples),
            }
        )
        noise.append(
            {
                "kind": "time",
                "mad": stats["mad"],
                "max_abs_ratio_minus_1": stats["max_abs_ratio_minus_1"],
                "median_ratio": statistics.median(ratios),
                "name": name,
                "p99_abs_ratio_minus_1": stats["p99_abs_ratio_minus_1"],
                "params": list(params),
                "ratios": ratios,
                "seconds": statistics.median(samples),
                "three_mad": stats["three_mad"],
            }
        )
        thresholds.append(
            {
                "estimator": "max(floor, 3xMAD, p99, max_abs)",
                "factor": stats["factor"],
                "kind": "time",
                "mad": stats["mad"],
                "max_abs_ratio_minus_1": stats["max_abs_ratio_minus_1"],
                "name": name,
                "p99_abs_ratio_minus_1": stats["p99_abs_ratio_minus_1"],
                "params": list(params),
                "relative_threshold": stats["relative_threshold"],
                "three_mad": stats["three_mad"],
            }
        )
    write_json(OUT / "timing.json", timing)
    write_json(OUT / "noise.json", noise)
    write_json(
        OUT / "thresholds.json",
        {
            "benchmarks": thresholds,
            "estimator": "max(floor, 3xMAD, p99, max_abs)",
            "floor": FLOOR,
            "rounds_per_run": MIN_SAMPLES,
            "rule": RULE,
            "runs": len(files),
        },
    )


def as_number(value):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (int, float)):
        return value[0]
    return value


def memory_save(index: int, results_dir: Path, commit: str, machine: str) -> None:
    path, _meta = select_result(results_dir, commit, machine)
    rows = rows_from_file(path, "memory", machine)
    saved = []
    for row in rows:
        if row["commit"] != commit or row["machine"] != machine:
            raise SystemExit("selected memory row does not match the requested commit and machine")
        saved.append(
            {
                "bytes": as_number(row["value"]),
                "commit": row["commit"],
                "env_name": row["env_name"],
                "machine": row["machine"],
                "name": row["name"],
                "params": row["params"],
                "samples": row["samples"],
            }
        )
    if not saved:
        raise SystemExit("no memory rows in the selected result")
    write_json(memory_runs_dir() / f"{index:02d}.json", saved)


def _memory_threshold(spread):
    if spread is None:
        raise SystemExit("memory pair is missing a spread")
    relative = max(FLOOR, float(spread))
    if relative < float(spread):
        raise SystemExit("memory gate narrower than measured A/A")
    return relative


def memory_pair() -> None:
    first_path = memory_runs_dir() / "01.json"
    second_path = memory_runs_dir() / "02.json"
    if not first_path.is_file() or not second_path.is_file():
        raise SystemExit("memory pair needs two selected runs")
    first = json.loads(first_path.read_text())
    second = json.loads(second_path.read_text())
    first_by = {(row["name"], tuple(row["params"])): row for row in first}
    memory = []
    extra_noise = []
    extra_thresholds = []
    for row in second:
        key = (row["name"], tuple(row["params"]))
        earlier_row = first_by.get(key)
        if earlier_row is None:
            raise SystemExit(f"memory pair is missing {row['name']} {row['params']}")
        earlier = as_number(earlier_row["bytes"])
        later = as_number(row["bytes"])
        if earlier in (None, 0) or later is None:
            raise SystemExit(f"memory pair has no bytes for {row['name']} {row['params']}")
        ratio = later / earlier
        spread = abs(ratio - 1.0)
        relative = _memory_threshold(spread)
        memory.append(
            {
                "bytes": later,
                "commit": row["commit"],
                "env_name": row["env_name"],
                "machine": row["machine"],
                "name": row["name"],
                "params": row["params"],
            }
        )
        extra_noise.append(
            {
                "earlier_bytes": earlier,
                "kind": "memory",
                "later_bytes": later,
                "name": row["name"],
                "params": row["params"],
                "ratio_later_over_earlier": ratio,
                "spread": spread,
            }
        )
        extra_thresholds.append(
            {
                "estimator": "max(floor, abs deviation)",
                "factor": 1.0 + relative,
                "kind": "memory",
                "name": row["name"],
                "params": row["params"],
                "relative_threshold": relative,
                "spread": spread,
            }
        )
    write_json(OUT / "memory.json", memory)
    noise_path = OUT / "noise.json"
    previous = json.loads(noise_path.read_text()) if noise_path.is_file() else []
    previous = [row for row in previous if row.get("kind") != "memory"]
    write_json(noise_path, previous + extra_noise)
    payload = json.loads((OUT / "thresholds.json").read_text())
    kept = [row for row in payload.get("benchmarks", []) if row.get("kind") != "memory" and "ray_" not in row.get("name", "")]
    payload["benchmarks"] = kept + extra_thresholds
    write_json(OUT / "thresholds.json", payload)


def row_key(row) -> tuple:
    return (row["name"], tuple(row["params"]))


def gate_rows(measurements: list[dict], thresholds: list[dict]) -> list[dict]:
    """Fail when a row's ratio exceeds that row's own factor.

    The decision does not use the maximum factor across rows.
    """
    bounds = {}
    for row in thresholds:
        if row.get("kind") == "memory" or "ray_" in row.get("name", ""):
            continue
        if "time_" not in row.get("name", "") and row.get("kind") != "time":
            continue
        bounds[row_key(row)] = row
    seen = {}
    for row in measurements:
        seen[row_key(row)] = row
    failures = []
    if not bounds:
        raise SystemExit("no per-benchmark timing thresholds")
    for key, bound in sorted(bounds.items(), key=lambda item: (item[0][0], [str(part) for part in item[0][1]])):
        measured = seen.get(key)
        factor = float(bound["factor"])
        if measured is None or measured.get("ratio") is None:
            failures.append(
                {
                    "factor": factor,
                    "name": key[0],
                    "params": list(key[1]),
                    "ratio": None,
                    "reason": "missing",
                }
            )
            continue
        ratio = float(measured["ratio"])
        if ratio > factor:
            failures.append(
                {
                    "factor": factor,
                    "name": key[0],
                    "params": list(key[1]),
                    "ratio": ratio,
                    "reason": "regression",
                }
            )
    for key, measured in sorted(
        seen.items(),
        key=lambda item: (item[0][0], [str(part) for part in item[0][1]]),
    ):
        if key in bounds or not _is_timing_name(key[0]):
            continue
        failures.append(
            {
                "factor": None,
                "name": key[0],
                "params": list(key[1]),
                "ratio": measured.get("ratio"),
                "reason": "no-threshold",
            }
        )
    return failures


def _is_timing_name(name: str) -> bool:
    return "time_" in name and "ray_" not in name


def median_ratio(base_samples: list[float], head_samples: list[float]) -> float:
    if not base_samples or not head_samples:
        raise SystemExit("missing samples")
    base = statistics.median(base_samples)
    head = statistics.median(head_samples)
    if base == 0:
        raise SystemExit("zero base median")
    return head / base


def measurements_from_rows(base_rows: list[dict], head_rows: list[dict]) -> list[dict]:
    head_by = {row_key(row): row for row in head_rows}
    measurements = []
    for row in base_rows:
        key = row_key(row)
        other = head_by.get(key)
        if other is None:
            measurements.append({"name": row["name"], "params": row["params"], "ratio": None})
            continue
        measurements.append(
            {
                "name": row["name"],
                "params": row["params"],
                "ratio": median_ratio(row["samples"], other["samples"]),
            }
        )
    base_keys = {row_key(row) for row in base_rows}
    for row in head_rows:
        if row_key(row) not in base_keys:
            measurements.append({"name": row["name"], "params": row["params"], "ratio": None})
    return measurements


def scrub_node(text: str) -> str:
    node = platform.node()
    if node and len(node) >= 3:
        return text.replace(node, MACHINE_NAME)
    return text


def _fmt(value) -> str:
    if value is None:
        return "missing"
    return format(float(value), ".17g")


def format_gate_lines(measurements: list[dict], thresholds: list[dict], failures: list[dict]) -> list[str]:
    failed = {row_key(row) for row in failures}
    bounds = {row_key(row): row for row in thresholds if row.get("kind") != "memory" and "ray_" not in row.get("name", "")}
    lines = ["name\tparams\tratio\tfactor\tresult"]
    seen = {row_key(row): row for row in measurements}
    keys = sorted(set(bounds) | set(seen), key=lambda item: (item[0], [str(part) for part in item[1]]))
    for key in keys:
        if "ray_" in key[0]:
            continue
        bound = bounds.get(key)
        measured = seen.get(key)
        factor = None if bound is None else bound.get("factor")
        ratio = None if measured is None else measured.get("ratio")
        if key in failed:
            result = "fail"
        elif bound is None or measured is None:
            result = "fail"
        else:
            result = "ok"
        lines.append(
            f"{key[0]}\t{list(key[1])}\t{_fmt(ratio)}\t{_fmt(factor)}\t{result}"
        )
    return lines


def decide_status(base: str, head: str, failures: list[dict] | None, results_ready: bool) -> int:
    if base == head or not results_ready or failures:
        return 1
    return 0


def command_gate(args: argparse.Namespace) -> None:
    lines = [
        f"base_hash={args.base}",
        f"head_hash={args.head}",
        f"machine={args.machine}",
        "interleave=yes quick=no skip_existing=no",
        "gate=per-benchmark",
    ]
    log_text = ""
    if args.asv_log:
        log_path = Path(args.asv_log)
        if log_path.is_file():
            log_text = scrub_node(log_path.read_text(errors="replace"))
    failures = None
    results_ready = False
    if args.base == args.head:
        lines.append("PR gate requires distinct base and head revisions")
        lines.append("asv_continuous=not-run")
    else:
        lines.append("asv_continuous=ran")
        lines.append(f"asv_status={args.asv_status}")
        if args.results:
            try:
                base_path, _base_meta = select_result(Path(args.results), args.base, args.machine)
                head_path, _head_meta = select_result(Path(args.results), args.head, args.machine)
                base_rows = rows_from_file(base_path, "time", args.machine)
                head_rows = rows_from_file(head_path, "time", args.machine)
                thresholds = json.loads(Path(args.thresholds).read_text())["benchmarks"]
                measurements = measurements_from_rows(base_rows, head_rows)
                failures = gate_rows(measurements, thresholds)
                lines.extend(format_gate_lines(measurements, thresholds, failures))
                results_ready = True
            except SystemExit as exc:
                lines.append(f"gate_error={exc}")
                results_ready = False
        else:
            lines.append("gate_error=missing results directory")
    if log_text:
        lines.append("asv_log<<")
        lines.append(log_text.rstrip())
        lines.append(">>")
    status = decide_status(args.base, args.head, failures, results_ready)
    if args.asv_status:
        status = args.asv_status
    lines.append(f"gate_status={status}")
    report = Path(args.report)
    status_path = Path(args.status)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    status_path.write_text(f"{status}\n")


def _add_result_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("index", type=int)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--machine", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare-machine").set_defaults(func=lambda _args: prepare_machine())
    sub.add_parser("check-machine").set_defaults(func=lambda _args: check_machine())

    derive = sub.add_parser("derive-conf")
    derive.add_argument("src", type=Path)
    derive.add_argument("dest", type=Path)
    derive.add_argument("results_dir")
    derive.set_defaults(func=lambda args: derive_conf(args.src, args.dest, args.results_dir))

    pair = sub.add_parser("save-pair")
    pair.add_argument("index", type=int)
    pair.add_argument("--earlier", required=True, type=Path)
    pair.add_argument("--later", required=True, type=Path)
    pair.add_argument("--commit", required=True)
    pair.add_argument("--machine", required=True)
    pair.set_defaults(
        func=lambda args: save_pair(
            args.index, args.earlier, args.later, args.commit, args.machine
        )
    )

    sub.add_parser("aggregate").set_defaults(func=lambda _args: aggregate())

    memory = sub.add_parser("memory-save")
    _add_result_args(memory)
    memory.set_defaults(
        func=lambda args: memory_save(args.index, args.results, args.commit, args.machine)
    )

    sub.add_parser("memory-pair").set_defaults(func=lambda _args: memory_pair())

    gate = sub.add_parser("gate")
    gate.add_argument("--base", required=True)
    gate.add_argument("--head", required=True)
    gate.add_argument("--machine", required=True)
    gate.add_argument("--results", default=None)
    gate.add_argument("--thresholds", default="benchflow/out/thresholds.json")
    gate.add_argument("--asv-log", default=None)
    gate.add_argument("--asv-status", type=int, default=0)
    gate.add_argument("--report", required=True)
    gate.add_argument("--status", required=True)
    gate.set_defaults(func=command_gate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
