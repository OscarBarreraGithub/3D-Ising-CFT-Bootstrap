#!/usr/bin/env python3
"""
Detailed log analyzer for bootstrap SLURM jobs.

Usage:
    python scripts/analyze_logs.py --job 59766814
    python scripts/analyze_logs.py --job 59766814 --stage a
    python scripts/analyze_logs.py --job 59766814 --task 0 --verbose
    python scripts/analyze_logs.py --job 59766814 --output report.json

Features:
- Parse all logs for a given job ID
- Extract SDPB iterations before crashes
- Count signal 9 events per task
- Report memory statistics from sacct
- Detect retry loops (multiple PMP writes per task)
- Generate structured summary report
"""

import argparse
import re
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

@dataclass
class TaskAnalysis:
    """Analysis result for a single task."""
    task_id: int
    log_file: str
    status: str  # "completed", "oom_killed", "mpi_error", "timeout", "unknown"

    # SDPB execution tracking
    sdpb_calls: int  # Number of SDPB invocations (binary search iterations)
    pmp_writes: int  # Number of PMP JSON writes
    last_iteration: Optional[int]  # Last iteration count before crash

    # Error detection
    oom_kills: int  # Count of "signal 9" or "oom" messages
    mpi_errors: int  # Count of MPI-related errors

    # Resource usage (from sacct)
    max_rss_mb: Optional[float]  # Peak memory usage
    elapsed_time: Optional[str]  # Wall clock time

    # Result validation
    result_file_exists: bool
    result_file_size: int  # bytes
    result_value: Optional[float]  # Parsed Δε_max or Δε'_max

@dataclass
class JobAnalysis:
    """Overall analysis for entire job."""
    job_id: int
    stage: str  # "a" or "b"
    total_tasks: int

    # Task-level statistics
    completed_tasks: int
    oom_killed_tasks: int
    mpi_error_tasks: int
    timeout_tasks: int
    unknown_status_tasks: int

    # Error aggregates
    total_oom_kills: int
    total_mpi_errors: int

    # Resource statistics
    max_memory_mb: float
    avg_memory_mb: float

    # Result validation
    valid_results: int
    empty_results: int
    missing_results: int

    # Per-task details
    tasks: List[TaskAnalysis]

class LogAnalyzer:
    """Main analyzer class."""

    def __init__(self, job_id: int, stage: str = "a"):
        self.job_id = job_id
        self.stage = stage
        self.log_dir = Path("/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap/logs")
        self.data_dir = Path("/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap/data")

    def find_log_files(self) -> List[Path]:
        """Find all log files for this job."""
        pattern = f"stage_{self.stage}_sdpb_{self.job_id}_*.log"
        return sorted(self.log_dir.glob(pattern))

    def parse_log_file(self, log_file: Path) -> TaskAnalysis:
        """Parse a single log file."""
        # Extract task ID from filename: stage_a_sdpb_59766814_0.log -> 0
        task_id = int(log_file.stem.split("_")[-1])

        # Initialize counters
        sdpb_calls = 0
        pmp_writes = 0
        oom_kills = 0
        mpi_errors = 0
        last_iteration = None

        # Patterns to match
        patterns = {
            'sdpb_call': re.compile(r'Running sdpb.*precision=(\d+)'),
            'pmp_write': re.compile(r'Writing PMP JSON \((\d+) blocks'),
            'oom_kill': re.compile(r'(signal 9|oom_kill|Killed)', re.IGNORECASE),
            'mpi_error': re.compile(r'(mpirun.*error|not enough slots|oversubscribe)', re.IGNORECASE),
            'iteration': re.compile(r'iter\s+(\d+)', re.IGNORECASE),
        }

        try:
            with open(log_file) as f:
                for line in f:
                    if patterns['sdpb_call'].search(line):
                        sdpb_calls += 1
                    if patterns['pmp_write'].search(line):
                        pmp_writes += 1
                    if patterns['oom_kill'].search(line):
                        oom_kills += 1
                    if patterns['mpi_error'].search(line):
                        mpi_errors += 1

                    # Extract last iteration count
                    match = patterns['iteration'].search(line)
                    if match:
                        last_iteration = int(match.group(1))
        except (IOError, UnicodeDecodeError) as e:
            # Log file might be corrupted or still being written
            pass

        # Determine status
        if oom_kills > 0:
            status = "oom_killed"
        elif mpi_errors > 0:
            status = "mpi_error"
        elif sdpb_calls > 0 and pmp_writes >= sdpb_calls:
            status = "completed"
        else:
            status = "unknown"

        # Check result file
        result_file = self.get_result_file(task_id)
        result_exists = result_file.exists()
        result_size = result_file.stat().st_size if result_exists else 0
        result_value = self.parse_result_value(result_file) if result_exists and result_size > 100 else None

        return TaskAnalysis(
            task_id=task_id,
            log_file=str(log_file),
            status=status,
            sdpb_calls=sdpb_calls,
            pmp_writes=pmp_writes,
            last_iteration=last_iteration,
            oom_kills=oom_kills,
            mpi_errors=mpi_errors,
            max_rss_mb=None,  # Filled by sacct query
            elapsed_time=None,
            result_file_exists=result_exists,
            result_file_size=result_size,
            result_value=result_value,
        )

    def get_result_file(self, task_id: int) -> Path:
        """Get path to result CSV for a task."""
        if self.stage == "a":
            return self.data_dir / f"eps_bound_{task_id}.csv"
        else:
            return self.data_dir / f"epsprime_bound_{task_id}.csv"

    def parse_result_value(self, csv_file: Path) -> Optional[float]:
        """Parse the Δε or Δε' value from CSV."""
        try:
            with open(csv_file) as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    # Expected format: "0.500000,1.234567"
                    value_str = lines[1].strip().split(',')[-1]
                    return float(value_str)
        except (IOError, ValueError):
            pass
        return None

    def query_sacct(self) -> Dict[int, Dict[str, str]]:
        """Query SLURM accounting for resource usage."""
        cmd = [
            "sacct", "-j", str(self.job_id),
            "--format=JobID,MaxRSS,Elapsed,State",
            "--units=M",
            "--noheader",
            "--parsable2"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            print("Warning: sacct query timed out")
            return {}

        # Parse output: "59766814_0.batch|12345M|01:23:45|COMPLETED"
        usage = {}
        for line in result.stdout.strip().split('\n'):
            if '.batch' in line:
                parts = line.split('|')
                jobid = parts[0]
                try:
                    task_id = int(jobid.split('_')[1].split('.')[0])
                except (IndexError, ValueError):
                    continue

                usage[task_id] = {
                    'max_rss': parts[1] if len(parts) > 1 else '',
                    'elapsed': parts[2] if len(parts) > 2 else '',
                    'state': parts[3] if len(parts) > 3 else '',
                }

        return usage

    def analyze(self) -> JobAnalysis:
        """Perform full analysis."""
        log_files = self.find_log_files()

        if not log_files:
            print(f"Warning: No log files found for job {self.job_id} stage {self.stage}")
            # Return empty analysis
            return JobAnalysis(
                job_id=self.job_id,
                stage=self.stage,
                total_tasks=0,
                completed_tasks=0,
                oom_killed_tasks=0,
                mpi_error_tasks=0,
                timeout_tasks=0,
                unknown_status_tasks=0,
                total_oom_kills=0,
                total_mpi_errors=0,
                max_memory_mb=0.0,
                avg_memory_mb=0.0,
                valid_results=0,
                empty_results=0,
                missing_results=0,
                tasks=[],
            )

        tasks = [self.parse_log_file(f) for f in log_files]

        # Query sacct for resource data
        sacct_data = self.query_sacct()

        # Merge sacct data into task analyses
        for task in tasks:
            if task.task_id in sacct_data:
                data = sacct_data[task.task_id]
                # Parse MaxRSS: "12345M" -> 12345.0
                rss_str = data['max_rss'].rstrip('M')
                task.max_rss_mb = float(rss_str) if rss_str and rss_str != '' else None
                task.elapsed_time = data['elapsed']

        # Compute aggregates
        completed = sum(1 for t in tasks if t.status == "completed")
        oom_killed = sum(1 for t in tasks if t.status == "oom_killed")
        mpi_error = sum(1 for t in tasks if t.status == "mpi_error")

        total_oom = sum(t.oom_kills for t in tasks)
        total_mpi = sum(t.mpi_errors for t in tasks)

        valid_results = sum(1 for t in tasks if t.result_file_size > 100)
        empty_results = sum(1 for t in tasks if t.result_file_exists and t.result_file_size <= 100)
        missing_results = sum(1 for t in tasks if not t.result_file_exists)

        # Memory statistics
        memory_values = [t.max_rss_mb for t in tasks if t.max_rss_mb is not None]
        max_memory = max(memory_values) if memory_values else 0.0
        avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0.0

        return JobAnalysis(
            job_id=self.job_id,
            stage=self.stage,
            total_tasks=len(tasks),
            completed_tasks=completed,
            oom_killed_tasks=oom_killed,
            mpi_error_tasks=mpi_error,
            timeout_tasks=0,  # TODO: detect from sacct
            unknown_status_tasks=len(tasks) - completed - oom_killed - mpi_error,
            total_oom_kills=total_oom,
            total_mpi_errors=total_mpi,
            max_memory_mb=max_memory,
            avg_memory_mb=avg_memory,
            valid_results=valid_results,
            empty_results=empty_results,
            missing_results=missing_results,
            tasks=tasks,
        )

    def print_summary(self, analysis: JobAnalysis):
        """Print human-readable summary."""
        print(f"\n{'='*70}")
        print(f"Log Analysis Report: Job {analysis.job_id} (Stage {analysis.stage.upper()})")
        print(f"{'='*70}\n")

        print(f"TASK STATUS:")
        print(f"  Total tasks:        {analysis.total_tasks}")
        print(f"  Completed:          {analysis.completed_tasks}")
        print(f"  OOM killed:         {analysis.oom_killed_tasks}")
        print(f"  MPI errors:         {analysis.mpi_error_tasks}")
        print(f"  Unknown:            {analysis.unknown_status_tasks}")
        print()

        print(f"ERROR SUMMARY:")
        print(f"  Total OOM kills:    {analysis.total_oom_kills} events")
        print(f"  Total MPI errors:   {analysis.total_mpi_errors} events")
        print()

        if analysis.max_memory_mb > 0:
            print(f"MEMORY USAGE:")
            print(f"  Peak:               {analysis.max_memory_mb:.0f} MB")
            print(f"  Average:            {analysis.avg_memory_mb:.0f} MB")
            print()

        print(f"RESULTS:")
        print(f"  Valid:              {analysis.valid_results} files")
        print(f"  Empty:              {analysis.empty_results} files")
        print(f"  Missing:            {analysis.missing_results} files")
        print()

        # Show worst offenders
        oom_tasks = [t for t in analysis.tasks if t.oom_kills > 0]
        if oom_tasks:
            print(f"OOM KILLED TASKS:")
            for t in sorted(oom_tasks, key=lambda x: x.oom_kills, reverse=True)[:10]:
                rss_str = f"MaxRSS={t.max_rss_mb:.0f}MB" if t.max_rss_mb else "MaxRSS=N/A"
                print(f"  Task {t.task_id}: {t.oom_kills} kills, {rss_str}")
            print()

        # Show tasks with missing results
        missing_tasks = [t for t in analysis.tasks if not t.result_file_exists]
        if missing_tasks:
            print(f"MISSING RESULTS (first 10):")
            for t in missing_tasks[:10]:
                print(f"  Task {t.task_id}: status={t.status}")
            print()

def main():
    parser = argparse.ArgumentParser(description="Analyze SLURM job logs")
    parser.add_argument("--job", type=int, required=True, help="Job ID to analyze")
    parser.add_argument("--stage", choices=["a", "b"], default="a", help="Pipeline stage")
    parser.add_argument("--task", type=int, help="Analyze single task only")
    parser.add_argument("--output", help="Save JSON report to file")
    parser.add_argument("--verbose", action="store_true", help="Show per-task details")

    args = parser.parse_args()

    analyzer = LogAnalyzer(args.job, args.stage)
    analysis = analyzer.analyze()

    analyzer.print_summary(analysis)

    if args.verbose and analysis.tasks:
        print(f"\nPER-TASK DETAILS:")
        for t in analysis.tasks[:10]:  # Show first 10
            print(f"Task {t.task_id}: status={t.status}, sdpb_calls={t.sdpb_calls}, oom_kills={t.oom_kills}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(analysis), f, indent=2)
        print(f"\nJSON report saved to: {args.output}")

if __name__ == "__main__":
    main()
