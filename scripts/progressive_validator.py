#!/usr/bin/env python3
"""
Progressive validation daemon for 3D-Ising-CFT-Bootstrap pipeline.

Monitors Stage A/B results as they complete and detects anomaly patterns early.
Saves 15-25 hours by catching systematic failures before all tasks finish.

Usage:
    python scripts/progressive_validator.py \\
        --stage a \\
        --job-id 12345678 \\
        --poll-interval 60 \\
        --output logs/validation_state.json

Features:
    - Polls data/ directory for new CSV files every 60s
    - Analyzes each result incrementally
    - Detects patterns: all NaN, all ~0.5 (unitarity floor), all ~2.5 (upper bound)
    - Triggers alerts at configurable thresholds (warning/critical)
    - Optional auto-cancellation of parent job on critical anomaly
    - Saves state to JSON for post-mortem analysis
"""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Import notification module
try:
    from notification import notify
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
    print("⚠ Warning: notification.py not found, notifications disabled", file=sys.stderr)


@dataclass
class AnomalyPattern:
    """Detected anomaly pattern."""
    pattern_type: str  # "all_lower", "all_upper", "all_nan", "timeout_cascade"
    affected_tasks: List[int]
    severity: str  # "warning", "critical"
    message: str
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationState:
    """Current state of progressive validation."""
    stage: str  # "a" or "b"
    job_id: str
    total_expected: int  # usually 51
    analyzed_tasks: Set[int] = field(default_factory=set)
    valid_results: Dict[int, float] = field(default_factory=dict)  # task_id -> value
    anomalous_tasks: Dict[int, str] = field(default_factory=dict)  # task_id -> reason
    patterns: List[AnomalyPattern] = field(default_factory=list)
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "stage": self.stage,
            "job_id": self.job_id,
            "total_expected": self.total_expected,
            "analyzed_tasks": sorted(list(self.analyzed_tasks)),
            "valid_results": {str(k): v for k, v in self.valid_results.items()},
            "anomalous_tasks": {str(k): v for k, v in self.anomalous_tasks.items()},
            "patterns": [p.to_dict() for p in self.patterns],
            "last_update": self.last_update,
            "started_at": self.started_at,
            "progress": f"{len(self.analyzed_tasks)}/{self.total_expected}"
        }


class ProgressiveValidator:
    """Progressive validation daemon for Stage A/B results."""

    TERMINAL_JOB_STATES = {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }

    def __init__(
        self,
        stage: str,
        job_id: str,
        data_dir: Path = Path("data"),
        poll_interval: int = 60,
        total_expected: int = 51,
        anomaly_threshold_warning: int = 10,
        anomaly_threshold_critical: int = 20,
        lower_bound_tolerance: float = 0.01,
        upper_bound_tolerance: float = 0.01,
        cancel_on_critical: bool = False
    ):
        self.stage = stage.lower()
        self.job_id = job_id
        self.data_dir = data_dir
        self.poll_interval = poll_interval
        self.total_expected = total_expected
        self.anomaly_threshold_warning = anomaly_threshold_warning
        self.anomaly_threshold_critical = anomaly_threshold_critical
        self.lower_bound_tolerance = lower_bound_tolerance
        self.upper_bound_tolerance = upper_bound_tolerance
        self.cancel_on_critical = cancel_on_critical
        # Default chosen to include normal Stage A/B single-row CSVs (~44-70 bytes).
        self.min_file_size_bytes = int(os.getenv("VALIDATOR_MIN_FILE_SIZE_BYTES", "20"))

        # File pattern based on stage
        if self.stage == "a":
            self.file_pattern = "eps_bound_*.csv"
            self.value_column = "delta_eps_max"
        elif self.stage == "b":
            self.file_pattern = "epsprime_bound_*.csv"
            self.value_column = "delta_eps_prime_max"
        else:
            raise ValueError(f"Invalid stage: {stage} (must be 'a' or 'b')")

        # Validation state
        self.state = ValidationState(
            stage=self.stage,
            job_id=self.job_id,
            total_expected=self.total_expected
        )

        # Known bounds
        self.UNITARITY_FLOOR = 0.5
        self.UPPER_BOUND = 2.5

    def discover_new_files(self) -> List[Tuple[int, Path]]:
        """Find CSV files that haven't been analyzed yet."""
        all_files = glob.glob(str(self.data_dir / self.file_pattern))
        new_files: List[Tuple[int, Path]] = []

        for file_path in all_files:
            # Extract task ID from filename
            # Format: eps_bound_42.csv or epsprime_bound_42.csv
            basename = Path(file_path).stem  # e.g., "eps_bound_42"
            try:
                task_id = int(basename.split("_")[-1])
            except (ValueError, IndexError):
                continue

            # Skip if already analyzed
            if task_id in self.state.analyzed_tasks:
                continue

            # Skip tiny files that are likely incomplete writes.
            try:
                if Path(file_path).stat().st_size < self.min_file_size_bytes:
                    continue
            except OSError:
                continue

            new_files.append((task_id, Path(file_path)))

        return new_files

    def analyze_result(self, task_id: int, file_path: Path) -> Tuple[str, Optional[float], Optional[str]]:
        """
        Analyze a single result file.

        Returns:
            Tuple(status, value, reason):
              - status="valid": value parsed successfully
              - status="anomalous": value is definitely bad (NaN/inf)
              - status="pending": file looks incomplete; retry later
        """
        try:
            with open(file_path) as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row is None:
                    return "pending", None, "no_data_rows_yet"

                # Get value from first row
                if self.value_column not in row:
                    return "pending", None, f"missing_column_{self.value_column}"

                value = float(row[self.value_column])

                # Check for NaN/inf
                if not math.isfinite(value):
                    self.state.anomalous_tasks[task_id] = "nan_or_inf"
                    return "anomalous", None, "nan_or_inf"

                # Valid result
                self.state.valid_results[task_id] = value
                return "valid", value, None

        except Exception as e:
            # Parsing may race with in-progress file writes; retry on next poll.
            return "pending", None, f"parse_retry_{type(e).__name__}"

    def get_parent_job_state(self) -> Optional[str]:
        """
        Get SLURM state for the monitored parent job.

        Returns:
            State string (e.g., RUNNING, COMPLETED, FAILED) or None if unavailable.
        """
        # Prefer sacct for terminal states/history.
        try:
            proc = subprocess.run(
                [
                    "sacct",
                    "-X",
                    "-j",
                    self.job_id,
                    "--format=JobIDRaw,State",
                    "--noheader",
                    "--parsable2",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                rows: List[Tuple[str, str]] = []
                for line in proc.stdout.splitlines():
                    if "|" not in line:
                        continue
                    job_id_raw, state_raw = line.split("|", 1)
                    job_id_raw = job_id_raw.strip()
                    state = state_raw.strip().split()[0].split("+", 1)[0].upper()
                    if not state:
                        continue
                    rows.append((job_id_raw, state))

                if rows:
                    for job_id_raw, state in rows:
                        if job_id_raw == self.job_id:
                            return state
                    return rows[0][1]
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback to squeue for active states.
        try:
            proc = subprocess.run(
                ["squeue", "-h", "-j", self.job_id, "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                state = proc.stdout.strip().splitlines()
                if state:
                    return state[0].strip().upper()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None

    def detect_patterns(self) -> List[AnomalyPattern]:
        """
        Detect anomaly patterns in accumulated results.

        Returns:
            List of detected patterns (new patterns only, not previously detected)
        """
        patterns = []

        # Need at least 5 results to detect patterns
        if len(self.state.analyzed_tasks) < 5:
            return patterns

        # Count values in different categories
        valid_count = len(self.state.valid_results)
        anomalous_count = len(self.state.anomalous_tasks)
        analyzed_count = len(self.state.analyzed_tasks)

        # Pattern 1: All NaN (SDPB timeout failures)
        if anomalous_count >= self.anomaly_threshold_warning:
            nan_tasks = [tid for tid, reason in self.state.anomalous_tasks.items()
                        if "nan" in reason.lower()]

            if len(nan_tasks) >= self.anomaly_threshold_warning:
                severity = "critical" if len(nan_tasks) >= self.anomaly_threshold_critical else "warning"

                # Check if already detected
                if not any(p.pattern_type == "all_nan" for p in self.state.patterns):
                    patterns.append(AnomalyPattern(
                        pattern_type="all_nan",
                        affected_tasks=nan_tasks,
                        severity=severity,
                        message=f"{len(nan_tasks)}/{analyzed_count} tasks returned NaN/inf values. "
                               f"This indicates SDPB timeout or solver failures. "
                               f"SDPB_TIMEOUT may be too short ({os.getenv('SDPB_TIMEOUT', '18000')}s)."
                    ))

        # Pattern 2: All near unitarity floor (scipy bug signature)
        if valid_count >= self.anomaly_threshold_warning:
            lower_tasks = [tid for tid, value in self.state.valid_results.items()
                          if abs(value - self.UNITARITY_FLOOR) < self.lower_bound_tolerance]

            if len(lower_tasks) >= self.anomaly_threshold_warning:
                severity = "critical" if len(lower_tasks) >= self.anomaly_threshold_critical else "warning"

                if not any(p.pattern_type == "all_lower" for p in self.state.patterns):
                    patterns.append(AnomalyPattern(
                        pattern_type="all_lower",
                        affected_tasks=lower_tasks,
                        message=f"{len(lower_tasks)}/{analyzed_count} tasks stuck at unitarity floor (~{self.UNITARITY_FLOOR}). "
                               f"This indicates LP solver conditioning bug (scipy/HiGHS). "
                               f"Ensure SDPB backend is being used, not scipy.",
                        severity=severity
                    ))

        # Pattern 3: All near upper bound (solver broken)
        if valid_count >= self.anomaly_threshold_warning:
            upper_tasks = [tid for tid, value in self.state.valid_results.items()
                          if abs(value - self.UPPER_BOUND) < self.upper_bound_tolerance]

            if len(upper_tasks) >= self.anomaly_threshold_warning:
                severity = "critical" if len(upper_tasks) >= self.anomaly_threshold_critical else "warning"

                if not any(p.pattern_type == "all_upper" for p in self.state.patterns):
                    patterns.append(AnomalyPattern(
                        pattern_type="all_upper",
                        affected_tasks=upper_tasks,
                        severity=severity,
                        message=f"{len(upper_tasks)}/{analyzed_count} tasks stuck at upper bound (~{self.UPPER_BOUND}). "
                               f"This indicates LP solver is broken or constraints are incorrect."
                    ))

        return patterns

    def handle_anomaly_detection(self, pattern: AnomalyPattern):
        """
        Handle detected anomaly pattern.

        Actions:
            - Log to console
            - Send notification (if available)
            - Optionally cancel job (if CANCEL_ON_CRITICAL=1 and severity=critical)
        """
        print("", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"[{pattern.severity.upper()}] ANOMALY DETECTED", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"Pattern: {pattern.pattern_type}", file=sys.stderr)
        print(f"Affected tasks: {len(pattern.affected_tasks)}/{self.total_expected}", file=sys.stderr)
        print(f"Message: {pattern.message}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("", file=sys.stderr)

        # Send notification
        if NOTIFICATIONS_AVAILABLE:
            stage_name = "Stage A" if self.stage == "a" else "Stage B"
            event_type = "anomaly_critical" if pattern.severity == "critical" else "anomaly_warning"

            notify(
                title=f"Anomaly Detected: {pattern.pattern_type}",
                message=pattern.message,
                severity=pattern.severity,
                event_type=event_type,
                context={
                    "stage": stage_name,
                    "job_id": self.job_id,
                    "pattern_type": pattern.pattern_type,
                    "affected_tasks": len(pattern.affected_tasks),
                    "analyzed_tasks": len(self.state.analyzed_tasks),
                    "total_tasks": self.total_expected
                }
            )

        # Cancel job if critical and enabled
        if pattern.severity == "critical" and self.cancel_on_critical:
            print(f"⚠ CANCEL_ON_CRITICAL=1: Cancelling job {self.job_id}", file=sys.stderr)
            try:
                subprocess.run(["scancel", self.job_id], check=True)
                print(f"✓ Job {self.job_id} cancelled successfully", file=sys.stderr)
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to cancel job {self.job_id}: {e}", file=sys.stderr)
            except FileNotFoundError:
                print("✗ scancel command not found", file=sys.stderr)

    def save_state(self, output_path: Path):
        """Save validation state to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def run(self, output_path: Path, max_iterations: Optional[int] = None):
        """
        Main validation loop.

        Args:
            output_path: Path to save state JSON
            max_iterations: Maximum iterations (None = infinite, for testing)
        """
        print(f"=== Progressive Validator Started ===", file=sys.stderr)
        print(f"Stage: {self.stage.upper()}", file=sys.stderr)
        print(f"Job ID: {self.job_id}", file=sys.stderr)
        print(f"File pattern: {self.file_pattern}", file=sys.stderr)
        print(f"Poll interval: {self.poll_interval}s", file=sys.stderr)
        print(f"Anomaly thresholds: warning={self.anomaly_threshold_warning}, critical={self.anomaly_threshold_critical}", file=sys.stderr)
        print(f"Cancel on critical: {self.cancel_on_critical}", file=sys.stderr)
        print(f"Min file size: {self.min_file_size_bytes} bytes", file=sys.stderr)
        print(f"Output: {output_path}", file=sys.stderr)
        print("", file=sys.stderr)

        iteration = 0

        while True:
            iteration += 1

            # Check iteration limit (for testing)
            if max_iterations and iteration > max_iterations:
                print(f"Reached max iterations ({max_iterations}), exiting", file=sys.stderr)
                break

            # Discover new files
            new_files = self.discover_new_files()

            if new_files:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(new_files)} new file(s)", file=sys.stderr)

                # Analyze each new file
                for task_id, file_path in new_files:
                    status, value, reason = self.analyze_result(task_id, file_path)
                    if status == "pending":
                        print(f"  Task {task_id:2d}: PENDING ({reason})", file=sys.stderr)
                        continue

                    self.state.analyzed_tasks.add(task_id)

                    if status == "valid" and value is not None:
                        print(f"  Task {task_id:2d}: {value:.4f}", file=sys.stderr)
                    else:
                        reason = self.state.anomalous_tasks.get(task_id, "unknown")
                        print(f"  Task {task_id:2d}: ANOMALOUS ({reason})", file=sys.stderr)

                # Update timestamp
                self.state.last_update = datetime.now().isoformat()

                # Detect patterns
                new_patterns = self.detect_patterns()

                if new_patterns:
                    for pattern in new_patterns:
                        self.state.patterns.append(pattern)
                        self.handle_anomaly_detection(pattern)

                # Save state
                self.save_state(output_path)

                # Check if all tasks analyzed
                if len(self.state.analyzed_tasks) >= self.total_expected:
                    print("", file=sys.stderr)
                    print(f"✓ All {self.total_expected} tasks analyzed. Validation complete.", file=sys.stderr)
                    break

            # Stop if monitored parent job has ended and there is nothing else to wait for.
            parent_state = self.get_parent_job_state()
            if parent_state in self.TERMINAL_JOB_STATES and len(self.state.analyzed_tasks) < self.total_expected:
                print(
                    f"Parent job {self.job_id} reached terminal state {parent_state}; "
                    f"stopping monitor at {len(self.state.analyzed_tasks)}/{self.total_expected}.",
                    file=sys.stderr,
                )
                self.state.last_update = datetime.now().isoformat()
                self.save_state(output_path)
                break

            # Wait before next poll
            time.sleep(self.poll_interval)


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="Progressive validation daemon")
    parser.add_argument("--stage", required=True, choices=["a", "b"], help="Stage to monitor (a or b)")
    parser.add_argument("--job-id", required=True, help="SLURM job ID to monitor")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--total-expected", type=int, default=51, help="Expected number of tasks")
    parser.add_argument("--anomaly-threshold-warning", type=int, default=10, help="Tasks for warning")
    parser.add_argument("--anomaly-threshold-critical", type=int, default=20, help="Tasks for critical")
    parser.add_argument("--lower-bound-tolerance", type=float, default=0.01, help="Tolerance for 0.5 detection")
    parser.add_argument("--upper-bound-tolerance", type=float, default=0.01, help="Tolerance for 2.5 detection")
    parser.add_argument("--cancel-on-critical", action="store_true", help="Auto-cancel job on critical anomaly")
    parser.add_argument("--output", default="logs/validation_state.json", help="Output JSON path")
    parser.add_argument("--max-iterations", type=int, help="Max iterations (for testing)")

    args = parser.parse_args()

    # Load config from environment
    anomaly_threshold_warning = int(os.getenv("ANOMALY_THRESHOLD_WARNING", args.anomaly_threshold_warning))
    anomaly_threshold_critical = int(os.getenv("ANOMALY_THRESHOLD_CRITICAL", args.anomaly_threshold_critical))
    lower_bound_tolerance = float(os.getenv("LOWER_BOUND_TOLERANCE", args.lower_bound_tolerance))
    upper_bound_tolerance = float(os.getenv("UPPER_BOUND_TOLERANCE", args.upper_bound_tolerance))
    cancel_on_critical = args.cancel_on_critical or os.getenv("CANCEL_ON_CRITICAL", "0") == "1"

    # Create validator
    validator = ProgressiveValidator(
        stage=args.stage,
        job_id=args.job_id,
        data_dir=Path(args.data_dir),
        poll_interval=args.poll_interval,
        total_expected=args.total_expected,
        anomaly_threshold_warning=anomaly_threshold_warning,
        anomaly_threshold_critical=anomaly_threshold_critical,
        lower_bound_tolerance=lower_bound_tolerance,
        upper_bound_tolerance=upper_bound_tolerance,
        cancel_on_critical=cancel_on_critical
    )

    # Run validation loop
    try:
        validator.run(Path(args.output), max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user", file=sys.stderr)
        validator.save_state(Path(args.output))
        sys.exit(1)
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        validator.save_state(Path(args.output))
        sys.exit(1)


if __name__ == "__main__":
    main()
