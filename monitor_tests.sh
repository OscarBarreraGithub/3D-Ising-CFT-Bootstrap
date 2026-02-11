#!/bin/bash
# Monitor both test jobs

echo "Monitoring test jobs..."
echo "TEST 1 (16G): Should FAIL with error detection"
echo "TEST 2 (128G): Should SUCCEED with valid result"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Test Job Status - $(date)"
    echo "=========================================="
    echo ""

    sacct -j 59829132,59829443 --format=JobID%12,JobName%20,State%12,ExitCode%10,Elapsed%12,MaxRSS%12,ReqMem%8

    echo ""
    echo "=========================================="
    echo "Recent Log Output"
    echo "=========================================="

    echo ""
    echo "--- TEST 1 (16G - should fail) ---"
    tail -15 logs/test_error_detection_59829132.log 2>/dev/null | grep -v "^$"

    echo ""
    echo "--- TEST 2 (128G - should succeed) ---"
    tail -15 logs/test_sufficient_memory_59829443.log 2>/dev/null | grep -v "^$"

    echo ""
    echo "Checking again in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
