#!/bin/bash
# Quick pipeline monitoring script
# Usage: bash MONITOR.sh

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         3D Ising CFT Bootstrap Pipeline Monitor           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "🔍 JOB STATUS:"
squeue -u obarrera -o "%.10i %.12P %.20j %.8T %.10M %.6D %R"
echo ""

echo "📊 STAGE A PROGRESS:"
n_stage_a=$(ls data/eps_bound_*.csv 2>/dev/null | wc -l)
echo "  Results: ${n_stage_a}/51 tasks completed"

if [ "$n_stage_a" -gt 0 ]; then
    echo ""
    echo "  Sample results:"
    head -1 data/eps_bound_0.csv 2>/dev/null
    head -2 data/eps_bound_0.csv 2>/dev/null | tail -1
    if [ -f data/eps_bound_32.csv ]; then
        head -2 data/eps_bound_32.csv 2>/dev/null | tail -1
        echo "  (task 32 ≈ Ising point, expect Δε_max ≈ 1.41)"
    fi
fi
echo ""

echo "📊 STAGE B PROGRESS:"
n_stage_b=$(ls data/epsprime_bound_*.csv 2>/dev/null | wc -l)
echo "  Results: ${n_stage_b}/51 tasks completed"
echo ""

echo "🎨 FINAL OUTPUT:"
if [ -f figures/fig6_reproduction.png ]; then
    ls -lh figures/fig6_reproduction.*
    echo "  ✅ Figure 6 generated!"
else
    echo "  ⏳ Not yet generated"
fi
echo ""

echo "⚠️  ERROR CHECK:"
n_oom=$(grep -l "oom" logs/stage_a_sdpb_59716207_*.log 2>/dev/null | wc -l)
n_mpi=$(grep -l "not enough slots" logs/stage_a_sdpb_59716207_*.log 2>/dev/null | wc -l)
echo "  OOM kills: ${n_oom}"
echo "  MPI errors: ${n_mpi}"

if [ "$n_stage_a" -gt 0 ]; then
    # Check if results are the bad values
    n_upper=$(tail -n +2 data/eps_bound_*.csv 2>/dev/null | awk -F',' '$2 > 2.4 {count++} END {print count+0}')
    n_lower=$(tail -n +2 data/eps_bound_*.csv 2>/dev/null | awk -F',' '$2 < 0.51 {count++} END {print count+0}')
    if [ "$n_upper" -gt 10 ]; then
        echo "  ⚠️  WARNING: ${n_upper} results near 2.5 (upper bound bug?)"
    fi
    if [ "$n_lower" -gt 10 ]; then
        echo "  ⚠️  WARNING: ${n_lower} results near 0.5 (scipy bug?)"
    fi
fi
echo ""

echo "📝 LATEST LOG ACTIVITY:"
latest_log=$(ls -t logs/stage_a_sdpb_59716207_*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "  File: $latest_log"
    echo "  Last 3 lines:"
    tail -3 "$latest_log" | sed 's/^/    /'
fi
echo ""

echo "💡 QUICK COMMANDS:"
echo "  Watch live log:    tail -f logs/stage_a_sdpb_59716207_0.log"
echo "  Check all results: ls data/eps_bound_*.csv | wc -l"
echo "  Resource usage:    sacct -j 59716207 --format=JobID,State,MaxRSS,Elapsed"
echo ""
