#!/usr/bin/env python3
"""Run one skill command with progress, wall-clock, and peak-RSS monitoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from typing import Any


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def _gpu_info() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            result.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_free_mib": int(fields[3]),
                }
            )
    return result


def _hardware() -> dict[str, Any]:
    memory = _meminfo()
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    return {
        "cpu_affinity_count": len(affinity),
        "cpu_affinity": affinity,
        "memory_total_bytes": memory.get("MemTotal"),
        "memory_available_bytes": memory.get("MemAvailable"),
        "gpus": _gpu_info(),
    }


def _artifact_signature(root: Path | None) -> tuple[int, int, int]:
    if root is None or not root.exists():
        return (0, 0, 0)
    count = 0
    total_size = 0
    newest = 0
    for directory, _, files in os.walk(root):
        for name in files:
            try:
                stat = os.stat(os.path.join(directory, name))
            except OSError:
                continue
            count += 1
            total_size += stat.st_size
            newest = max(newest, stat.st_mtime_ns)
    return (count, total_size, newest)


def _process_group_rss_bytes(process_group: int) -> int:
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = 0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            text = Path(entry.path, "stat").read_text(encoding="utf-8")
            fields = text[text.rfind(")") + 2 :].split()
            if int(fields[2]) == process_group:
                total += int(fields[21]) * page_size
        except (OSError, ValueError, IndexError):
            continue
    return total


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--idle-seconds", type=int, default=1200)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.idle_seconds < 1 or args.wall_seconds < 1 or args.poll_seconds <= 0:
        parser.error("watchdog time limits must be positive")

    hardware = _hardware()
    print(
        "[hardware] "
        f"cpus={hardware['cpu_affinity_count']} "
        f"ram_total={hardware['memory_total_bytes']} "
        f"ram_available={hardware['memory_available_bytes']} "
        f"gpus={hardware['gpus']}",
        flush=True,
    )
    print(f"[watchdog] label={args.label} command={command}", flush=True)

    artifact_root = (
        Path(args.artifact_root).expanduser().resolve()
        if args.artifact_root
        else None
    )
    signature = _artifact_signature(artifact_root)
    started = time.monotonic()
    last_progress = started
    last_artifact_scan = started
    peak_rss = 0
    trigger = "exit"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    try:
        while process.poll() is None:
            for key, _ in selector.select(timeout=args.poll_seconds):
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    chunk = b""
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    last_progress = time.monotonic()
            now = time.monotonic()
            peak_rss = max(peak_rss, _process_group_rss_bytes(process.pid))
            if now - last_artifact_scan >= min(30.0, args.idle_seconds / 4):
                current_signature = _artifact_signature(artifact_root)
                if current_signature != signature:
                    signature = current_signature
                    last_progress = now
                last_artifact_scan = now
            if now - started > args.wall_seconds:
                trigger = "wall_timeout"
                _terminate_group(process)
                break
            if now - last_progress > args.idle_seconds:
                trigger = "idle_timeout"
                _terminate_group(process)
                break
    finally:
        selector.close()

    remaining = process.stdout.read()
    if remaining:
        sys.stdout.buffer.write(remaining)
        sys.stdout.buffer.flush()
    return_code = process.wait()
    ended = time.monotonic()
    if trigger == "exit" and return_code != 0:
        trigger = "nonzero_exit"
    report = {
        "label": args.label,
        "command": command,
        "trigger": trigger,
        "return_code": return_code,
        "wall_seconds": ended - started,
        "peak_rss_bytes": peak_rss,
        "artifact_root": str(artifact_root) if artifact_root else None,
        "artifact_signature": signature,
        "hardware": hardware,
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    print(f"[watchdog] report={report_path} trigger={trigger}", flush=True)
    if trigger == "idle_timeout":
        return 124
    if trigger == "wall_timeout":
        return 125
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
