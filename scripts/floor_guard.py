#!/usr/bin/env python3
"""Diff-scoped enforcement of the CONSTRAINTS.md floor.

Usage: python3 scripts/floor_guard.py [--base <ref>]  (default base: main)
Exit codes:
  0: clean
  1: at least one floor violation detected
  2: guard could not run (not a git repo, missing merge base, etc.)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

GIT_EXE = shutil.which("git")

SUPPRESSIONS_RE = re.compile(
    r"#\s*noqa\b|#\s*type:\s*ignore\b|#\s*pyright:\s*ignore\b|\bnosemgrep\b|\bgitleaks:allow\b"
)
STUBS_RE = re.compile(
    r"raise\s+NotImplementedError\b"
    r"|#\s*TODO\b"
    r"|\bTODO\b:"
    r"|\bpass\s*#\s*stub\b"
    r"|except\s*(?:[\w.(),\s]+)?:\s*pass\b"
)
SKIPS_RE = re.compile(r"@pytest\.mark\.(?:skip|skipif|xfail)\b|\bpytest\.skip\(")
ASSERT_RE = re.compile(r"\bassert\b")
TEST_FILE_RE = re.compile(r"(?:^|/)(?:test_[^/]+\.py|[^/]+_test\.py)$")
EXCEPTION_ROW_RE = re.compile(r"^\|\s*(?:W|E)\d+\s*\|")

MIN_BEFORE_RE = re.compile(r"(>=|>|≥|at least|minimum|\bmin\b|no less than|not fall|not drop)\s*$")
MAX_BEFORE_RE = re.compile(
    r"(<=|<|≤|at most|maximum|\bmax\b|no more than|under|below|not grow|not exceed)\s*$"
)
MIN_AFTER_RE = re.compile(r"^\s*\S*\s*(or more|or higher|must not fall|must not drop)")
MAX_AFTER_RE = re.compile(r"^\s*\S*\s*(or less|or lower|must not grow|must not exceed)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class DiffLine:
    file: str
    text: str


@dataclass(frozen=True)
class Finding:
    rule: str
    file: str
    text: str


@dataclass(frozen=True)
class Threshold:
    value: float
    direction: str | None


def _bail(msg: str) -> int:
    print(f"floor-guard: {msg}", file=sys.stderr)
    return 2


def _run_git(args: list[str], *, allow_diff_exit: bool = False) -> str | None:
    if GIT_EXE is None:
        return None
    proc = subprocess.run(
        [GIT_EXE, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return proc.stdout
    if allow_diff_exit and proc.returncode == 1:
        return proc.stdout
    return None


def _path_of(header_path: str) -> str:
    if header_path.startswith(("a/", "b/")):
        return header_path[2:]
    return header_path


def _is_test_file(path: str) -> bool:
    return bool(TEST_FILE_RE.search(path))


def _is_constraints_file(path: str) -> bool:
    return Path(path).name == "CONSTRAINTS.md"


def _rule_key(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("|"):
        cells = [c.strip() for c in stripped.split("|") if c.strip()]
        if not cells:
            return None
        if cells[0] in {"Dimension", "Metric", "ID"} or set(cells[0]) <= {"-", ":"}:
            return None
        return cells[0]
    if stripped.startswith(("- ", "* ")):
        return stripped[2:].split(":", maxsplit=1)[0].strip()
    return None


def _is_exception_row(text: str) -> bool:
    return bool(EXCEPTION_ROW_RE.match(text.strip()))


def _extract_thresholds(text: str) -> list[Threshold]:
    out: list[Threshold] = []
    for match in NUMBER_RE.finditer(text):
        before = text[max(0, match.start() - 24) : match.start()].lower()
        after = text[match.end() : match.end() + 40].lower()
        if MIN_BEFORE_RE.search(before) or MIN_AFTER_RE.search(after):
            direction: str | None = "min"
        elif MAX_BEFORE_RE.search(before) or MAX_AFTER_RE.search(after):
            direction = "max"
        else:
            direction = None
        out.append(Threshold(value=float(match.group(0)), direction=direction))
    return out


def _parse_diff(diff_text: str) -> tuple[list[DiffLine], list[DiffLine], list[str]]:
    added: list[DiffLine] = []
    removed: list[DiffLine] = []
    deleted: list[str] = []

    current_file = ""
    old_file = ""
    in_header = False

    for line in diff_text.splitlines():
        if line.startswith("diff "):
            in_header = True
        elif line.startswith("@@"):
            in_header = False
        elif in_header:
            if line.startswith("--- "):
                old_file = _path_of(line[4:])
            elif line.startswith("+++ "):
                new_file = _path_of(line[4:])
                current_file = old_file if new_file == "/dev/null" else new_file
                if new_file == "/dev/null":
                    deleted.append(current_file)
        elif line.startswith("+"):
            added.append(DiffLine(file=current_file, text=line[1:]))
        elif line.startswith("-"):
            removed.append(DiffLine(file=current_file, text=line[1:]))

    return added, removed, deleted


def _scan_findings(
    added: list[DiffLine],
    removed: list[DiffLine],
    deleted: list[str],
) -> list[Finding]:
    findings: list[Finding] = []

    def flag(rule: str, file: str, text: str) -> None:
        findings.append(Finding(rule=rule, file=file, text=text.strip()[:120]))

    for item in added:
        if not _is_constraints_file(item.file) and item.file != "scripts/floor_guard.py":
            if SUPPRESSIONS_RE.search(item.text):
                flag("silenced-checker", item.file, item.text)
            if STUBS_RE.search(item.text):
                flag("unfinished-work", item.file, item.text)
            if SKIPS_RE.search(item.text):
                flag("test-made-easier", item.file, item.text)
        if _is_constraints_file(item.file) and _is_exception_row(item.text):
            flag("new-exception", item.file, item.text)

    deleted_set = set(deleted)
    for path in deleted:
        if _is_test_file(path):
            flag("test-deleted", path, "file deleted")

    for item in removed:
        if (
            _is_test_file(item.file)
            and item.file not in deleted_set
            and ASSERT_RE.search(item.text)
        ):
            flag("assertion-removed", item.file, item.text)

    removed_rules = [
        item for item in removed if _is_constraints_file(item.file) and _rule_key(item.text)
    ]
    added_rules = [
        item for item in added if _is_constraints_file(item.file) and _rule_key(item.text)
    ]

    for old_rule in removed_rules:
        key = _rule_key(old_rule.text)
        new_rule = next((cand for cand in added_rules if _rule_key(cand.text) == key), None)
        if new_rule is None:
            if not _is_exception_row(old_rule.text):
                flag("rule-removed", old_rule.file, old_rule.text)
            continue

        before = _extract_thresholds(old_rule.text)
        after = _extract_thresholds(new_rule.text)
        verdict: str | None = None
        for direction in ("min", "max", None):
            was = [t for t in before if t.direction == direction]
            now = [t for t in after if t.direction == direction]
            for idx, old_t in enumerate(was):
                if verdict is not None:
                    break
                if idx >= len(now):
                    verdict = "threshold-removed"
                elif now[idx].value == old_t.value:
                    continue
                elif direction == "min":
                    if now[idx].value < old_t.value:
                        verdict = "threshold-loosened"
                elif direction == "max":
                    if now[idx].value > old_t.value:
                        verdict = "threshold-loosened"
                else:
                    verdict = "threshold-changed"
        if verdict is not None:
            flag(verdict, old_rule.file, f"{old_rule.text}  ->  {new_rule.text}")

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff-scoped CONSTRAINTS.md floor guard.")
    parser.add_argument("--base", default="main", help="Git ref to compute merge-base against.")
    args = parser.parse_args(argv)

    top = _run_git(["rev-parse", "--show-toplevel"])
    if top is None or not top.strip():
        return _bail("not inside a git work tree")
    os.chdir(top.strip())

    merge_base = _run_git(["merge-base", args.base, "HEAD"])
    if merge_base is None or not merge_base.strip():
        return _bail(f"no merge base against {args.base}")
    base_sha = merge_base.strip()

    tracked = _run_git(["diff", "--unified=0", base_sha, "--"])
    if tracked is None:
        return _bail(f"could not diff against {base_sha}")

    untracked_raw = _run_git(["ls-files", "--others", "--exclude-standard"])
    if untracked_raw is None:
        return _bail("could not list untracked files")

    untracked_parts: list[str] = []
    for rel_path in untracked_raw.splitlines():
        if not rel_path:
            continue
        diff_part = _run_git(
            ["diff", "--no-index", "--unified=0", "/dev/null", rel_path],
            allow_diff_exit=True,
        )
        if diff_part is None:
            return _bail(f"could not diff untracked file {rel_path}")
        untracked_parts.append(diff_part)

    full_diff = tracked + "\n" + "\n".join(untracked_parts)
    added, removed, deleted = _parse_diff(full_diff)
    findings = _scan_findings(added, removed, deleted)

    if not findings:
        print("floor-guard: clean")
        return 0

    print(f"floor-guard: {len(findings)} floor violation(s):", file=sys.stderr)
    for finding in findings:
        print(f"  [{finding.rule}] {finding.file}: {finding.text}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
