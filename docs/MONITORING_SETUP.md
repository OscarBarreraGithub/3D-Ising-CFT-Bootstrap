# Monitoring Setup Guide

This guide explains how to configure and use the progressive validation and notification features for the 3D-Ising-CFT-Bootstrap pipeline.

## Overview

The monitoring system provides two key enhancements for long-running (60+ hour) pipeline runs:

1. **Progressive Result Validation**: Detects anomaly patterns early during Stage A/B execution
2. **Email/Slack Notifications**: Automated alerts for key pipeline events

## Quick Start

### 1. Configure Notifications (Recommended)

Run the interactive setup script:

```bash
cd /n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap
bash scripts/setup_notifications.sh
```

This will:
- Prompt for your email address (default: `${USER}@fas.harvard.edu`)
- Optionally configure Slack webhook URL
- Create `.env.notifications` config file
- Test the notification system

### 2. Run Pipeline with Monitoring

```bash
# Enable progressive validation (enabled by default)
ENABLE_PROGRESSIVE_VALIDATION=1 bash jobs/run_pipeline.sh

# Or use the quick launcher
bash START_OVERNIGHT.sh
```

That's it! You'll now receive notifications at key pipeline milestones and early warnings if anomalies are detected.

---

## Notification System

### Features

- **Email notifications** via `/usr/bin/mail` command (available on cluster)
- **Slack notifications** via webhook (optional)
- **Configurable filters** to enable/disable specific notification types
- **Best-effort delivery** (failures don't crash pipeline)

### Notification Triggers

1. **Stage A Complete**: All 51 tasks finished
2. **Stage A Validation Failed**: Merge validation detected anomalies (NaN, all ~0.5, all ~2.5)
3. **Stage A Validation Passed**: Validation succeeded, Stage B launching
4. **Stage B Submitted**: Stage B job launched
5. **Figure 6 Complete**: Final plot generated successfully
6. **Anomaly Alerts**: Progressive validation detected patterns (warning/critical)

### Configuration

#### Interactive Setup (Recommended)

```bash
bash scripts/setup_notifications.sh
```

#### Manual Configuration

1. Copy template:
   ```bash
   cp config/.env.notifications.example .env.notifications
   ```

2. Edit `.env.notifications`:
   ```bash
   EMAIL_ENABLED=1
   EMAIL_RECIPIENT="your_email@fas.harvard.edu"

   # Optional: Slack integration
   SLACK_ENABLED=0
   SLACK_WEBHOOK_URL=""
   ```

3. Test configuration:
   ```bash
   python scripts/notification.py --test
   ```

### Slack Setup (Optional)

1. Visit https://api.slack.com/messaging/webhooks
2. Create a Slack app for your workspace
3. Enable "Incoming Webhooks"
4. Add webhook to your desired channel
5. Copy the webhook URL (starts with `https://hooks.slack.com/...`)
6. Add to `.env.notifications`:
   ```bash
   SLACK_ENABLED=1
   SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
   ```

### Testing Notifications

```bash
# Test all configured channels
python scripts/notification.py --test

# Test email only
python scripts/notification.py --test-email your_email@fas.harvard.edu

# Test Slack only
python scripts/notification.py --test-slack "https://hooks.slack.com/..."

# Send custom notification
python scripts/notification.py \
    --title "Test Alert" \
    --message "This is a test message" \
    --severity info \
    --context "key1=value1,key2=value2"
```

---

## Progressive Validation

### Features

- **Real-time monitoring**: Polls for new CSV files every 60s
- **Anomaly detection**: Detects patterns in completed tasks
  - All NaN/inf (SDPB timeout failures)
  - All ~0.5 (unitarity floor - scipy bug signature)
  - All ~2.5 (upper bound - solver broken)
- **Thresholds**: Warning at 10 tasks, Critical at 20 tasks
- **Optional auto-cancellation**: Stop job automatically if critical anomaly detected
- **State persistence**: Saves validation state to JSON for post-mortem analysis

### How It Works

1. **Daemon Launch**: When you submit Stage A/B, a separate validation daemon job starts
2. **Polling**: Daemon checks `data/` directory every 60s for new `eps_bound_*.csv` files
3. **Analysis**: Each completed task is analyzed immediately
4. **Pattern Detection**: After every 10 tasks, checks for anomaly patterns
5. **Alerting**: If threshold exceeded, sends notification with severity level
6. **Termination**: Daemon exits when all tasks analyzed or parent job completes

### Configuration

#### Default Settings

Progressive validation is **enabled by default** with conservative settings:
- Poll interval: 60 seconds
- Warning threshold: 10 tasks
- Critical threshold: 20 tasks
- Auto-cancellation: **Disabled** (alerts only)

#### Custom Configuration

1. Copy template:
   ```bash
   cp config/.env.validation.example .env.validation
   ```

2. Edit `.env.validation`:
   ```bash
   ENABLE_PROGRESSIVE_VALIDATION=1
   POLL_INTERVAL=60  # seconds between checks
   ANOMALY_THRESHOLD_WARNING=10   # tasks before warning
   ANOMALY_THRESHOLD_CRITICAL=20  # tasks before critical alert
   CANCEL_ON_CRITICAL=0  # 0=alert only, 1=auto-cancel job
   ```

#### Disable Progressive Validation

```bash
# Via environment variable
ENABLE_PROGRESSIVE_VALIDATION=0 bash jobs/run_pipeline.sh

# Or in .env.validation
echo "ENABLE_PROGRESSIVE_VALIDATION=0" > .env.validation
```

### Monitoring Validation State

#### Check Daemon Status

```bash
# View SLURM queue
squeue -u $USER | grep validation

# View daemon log
tail -f logs/progressive_validation_<JOB_ID>.log
```

#### View Validation State

The daemon saves its state to `logs/validation_state_stagea_<JOB_ID>.json`:

```bash
cat logs/validation_state_stagea_12345678.json | python -m json.tool
```

Example output:
```json
{
  "stage": "a",
  "job_id": "12345678",
  "total_expected": 51,
  "analyzed_tasks": [0, 1, 2, 3, 4, 5],
  "valid_results": {
    "0": 1.4132,
    "1": 1.4098,
    "2": 1.4065
  },
  "anomalous_tasks": {
    "4": "nan_or_inf",
    "5": "parse_error"
  },
  "patterns": [],
  "progress": "6/51",
  "last_update": "2026-02-12T14:30:15"
}
```

### Auto-Cancellation (Use with Caution)

**WARNING**: Auto-cancellation will stop your Stage A/B job if a critical anomaly is detected. Use only if you're confident in the thresholds.

```bash
# Enable in .env.validation
CANCEL_ON_CRITICAL=1

# Or via environment variable
CANCEL_ON_CRITICAL=1 bash jobs/run_pipeline.sh
```

**Recommended**: Leave disabled (default) and cancel manually after reviewing alerts.

---

## Integration with Existing Workflow

### Modified Scripts

The monitoring features are integrated into existing pipeline scripts with **conditional execution** (pipeline works with or without monitoring):

1. **jobs/run_pipeline.sh**:
   - Launches progressive validation daemon for Stage A (if enabled)

2. **jobs/merge_stage_a_and_submit_b.slurm**:
   - Sends notifications for Stage A validation results (pass/fail with reason)
   - Sends notification for Stage B submission
   - Launches progressive validation daemon for Stage B (if enabled)

3. **jobs/final_merge_and_plot.slurm**:
   - Sends notification when Figure 6 generation completes

### Backwards Compatibility

All monitoring features are **opt-in** and **non-blocking**:

- Pipeline works identically if `.env.notifications` doesn't exist
- Progressive validation can be disabled with `ENABLE_PROGRESSIVE_VALIDATION=0`
- Notification failures are logged but don't crash pipeline (best-effort delivery)
- All monitoring scripts use `|| true` to prevent failures from affecting pipeline

---

## Performance Impact

### Progressive Validation Daemon

- **CPU**: <1% (polling + CSV parsing)
- **Memory**: <500MB (state + parsed results)
- **Disk I/O**: ~50 KB/min (stat calls + occasional reads)
- **Network**: <1 KB/notification
- **Partition**: Runs on `shared` partition (not `sapphire`), doesn't compete for production resources

### Notifications

- **Per-notification cost**: 1-5s (email) or 0.5-2s (Slack)
- **Total overhead**: <0.01% of 60-hour runtime (~10s per stage)

**Bottom line**: Negligible impact on pipeline performance.

---

## Troubleshooting

### Notifications Not Received

1. **Check configuration**:
   ```bash
   cat .env.notifications
   python scripts/notification.py --test
   ```

2. **Check email spam folder**: Automated emails from cluster may be filtered

3. **Check Slack webhook**:
   - URL should start with `https://hooks.slack.com/`
   - Test with: `python scripts/notification.py --test-slack "YOUR_WEBHOOK_URL"`

4. **Check logs**:
   ```bash
   grep -i "notification" logs/merge_a_submit_b_*.log
   grep -i "notification" logs/final_merge_and_plot_*.log
   ```

### Validation Daemon Not Running

1. **Check if disabled**:
   ```bash
   grep ENABLE_PROGRESSIVE_VALIDATION .env.validation
   ```

2. **Check SLURM queue**:
   ```bash
   squeue -u $USER | grep validation
   ```

3. **Check daemon logs**:
   ```bash
   ls -lht logs/progressive_validation_*.log | head -5
   tail -50 logs/progressive_validation_<JOB_ID>.log
   ```

4. **Check dependency**:
   - Daemon uses `--dependency=after:STAGE_A_JOB`
   - If Stage A fails to start, daemon won't start either
   - Use `squeue --dependency` to debug

### False Positive Anomaly Alerts

If you receive anomaly alerts but results look correct:

1. **Adjust thresholds**:
   ```bash
   # In .env.validation
   ANOMALY_THRESHOLD_WARNING=15  # Increase from 10
   ANOMALY_THRESHOLD_CRITICAL=30  # Increase from 20
   ```

2. **Check bound tolerances**:
   ```bash
   # In .env.validation
   LOWER_BOUND_TOLERANCE=0.02  # Increase from 0.01
   UPPER_BOUND_TOLERANCE=0.02  # Increase from 0.01
   ```

3. **Review validation state**:
   ```bash
   cat logs/validation_state_stagea_<JOB_ID>.json | python -m json.tool
   ```

---

## Examples

### Basic Usage (Email Only)

```bash
# Setup
bash scripts/setup_notifications.sh
# → Configure email only, skip Slack

# Run pipeline
bash jobs/run_pipeline.sh

# Monitor email for alerts
```

### Full Monitoring (Email + Slack + Progressive Validation)

```bash
# Setup notifications
bash scripts/setup_notifications.sh
# → Configure both email and Slack

# Configure validation (optional, defaults are good)
cp config/.env.validation.example .env.validation
# → Edit thresholds if desired

# Run pipeline with all monitoring
ENABLE_PROGRESSIVE_VALIDATION=1 bash jobs/run_pipeline.sh

# Check validation daemon status
squeue -u $USER | grep validation

# View validation state
tail -f logs/progressive_validation_<JOB_ID>.log
```

### Quick Test (6 tasks, ~1 hour)

```bash
# Setup notifications
bash scripts/setup_notifications.sh

# Run small test to verify monitoring works
STAGE_A_ARRAY=0-5 STAGE_B_ARRAY=0-5 ENABLE_PROGRESSIVE_VALIDATION=1 bash jobs/run_pipeline.sh

# Expected notifications:
# 1. Stage A validation passed (~30 min)
# 2. Stage B submitted (~30 min)
# 3. Figure 6 complete (~60 min)
```

### Disable Monitoring

```bash
# Disable progressive validation
ENABLE_PROGRESSIVE_VALIDATION=0 bash jobs/run_pipeline.sh

# Disable notifications (temporary)
EMAIL_ENABLED=0 bash jobs/run_pipeline.sh

# Disable both
ENABLE_PROGRESSIVE_VALIDATION=0 EMAIL_ENABLED=0 bash jobs/run_pipeline.sh
```

---

## Files Reference

### New Files

| File | Purpose |
|------|---------|
| [scripts/notification.py](../scripts/notification.py) | Core notification library (email/Slack) |
| [scripts/setup_notifications.sh](../scripts/setup_notifications.sh) | Interactive notification setup |
| [scripts/progressive_validator.py](../scripts/progressive_validator.py) | Validation daemon |
| [jobs/progressive_validation.slurm](../jobs/progressive_validation.slurm) | SLURM script for validation daemon |
| [config/.env.notifications.example](../config/.env.notifications.example) | Notification config template |
| [config/.env.validation.example](../config/.env.validation.example) | Validation config template |
| `.env.notifications` | User-specific notification config (git-ignored) |
| `.env.validation` | User-specific validation config (git-ignored) |

### Modified Files

| File | Changes |
|------|---------|
| [jobs/run_pipeline.sh](../jobs/run_pipeline.sh) | Launch Stage A validation daemon |
| [jobs/merge_stage_a_and_submit_b.slurm](../jobs/merge_stage_a_and_submit_b.slurm) | Add notifications + Stage B validation daemon |
| [jobs/final_merge_and_plot.slurm](../jobs/final_merge_and_plot.slurm) | Add completion notification |
| [.gitignore](../.gitignore) | Ignore user-specific config files |

---

## Support

For issues or questions:

1. Check this guide first
2. Review [docs/PROGRESS.md](PROGRESS.md) for pipeline status
3. Check pipeline logs in `logs/` directory
4. Review validation state JSON files
5. Test notifications with `python scripts/notification.py --test`

**Remember**: All monitoring features are optional and designed to enhance (not replace) manual monitoring. Use them to save time on 60-hour runs, but traditional monitoring (`squeue`, `tail -f logs/*.log`) still works perfectly.
