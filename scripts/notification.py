#!/usr/bin/env python3
"""
Notification system for 3D-Ising-CFT-Bootstrap pipeline.

Supports email and Slack notifications for key pipeline events.
Designed for hands-free monitoring of long-running SLURM jobs.

Usage:
    # Test email
    python scripts/notification.py --test-email user@example.com

    # Test Slack
    python scripts/notification.py --test-slack https://hooks.slack.com/...

    # Send notification (reads config from environment)
    python scripts/notification.py \\
        --title "Stage A Complete" \\
        --message "All tasks finished successfully" \\
        --severity info \\
        --context job_id=12345,valid_tasks=51
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class NotificationConfig:
    """Configuration for notification system."""
    email_enabled: bool = True
    email_recipient: str = ""
    slack_enabled: bool = False
    slack_webhook_url: str = ""

    # Notification filters
    notify_stage_a_complete: bool = True
    notify_stage_a_validation: bool = True
    notify_stage_b_submit: bool = True
    notify_stage_b_complete: bool = True
    notify_figure_complete: bool = True
    notify_anomaly_warning: bool = True
    notify_anomaly_critical: bool = True

    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Email settings
        config.email_enabled = os.getenv("EMAIL_ENABLED", "1") == "1"
        config.email_recipient = os.getenv("EMAIL_RECIPIENT", f"{os.getenv('USER', 'user')}@fas.harvard.edu")

        # Slack settings
        config.slack_enabled = os.getenv("SLACK_ENABLED", "0") == "1"
        config.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

        # Notification filters
        config.notify_stage_a_complete = os.getenv("NOTIFY_STAGE_A_COMPLETE", "1") == "1"
        config.notify_stage_a_validation = os.getenv("NOTIFY_STAGE_A_VALIDATION", "1") == "1"
        config.notify_stage_b_submit = os.getenv("NOTIFY_STAGE_B_SUBMIT", "1") == "1"
        config.notify_stage_b_complete = os.getenv("NOTIFY_STAGE_B_COMPLETE", "1") == "1"
        config.notify_figure_complete = os.getenv("NOTIFY_FIGURE_COMPLETE", "1") == "1"
        config.notify_anomaly_warning = os.getenv("NOTIFY_ANOMALY_WARNING", "1") == "1"
        config.notify_anomaly_critical = os.getenv("NOTIFY_ANOMALY_CRITICAL", "1") == "1"

        return config

    @classmethod
    def from_file(cls, path: Path) -> "NotificationConfig":
        """Load configuration from .env file."""
        config = cls()

        if not path.exists():
            return config

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                # Set attributes
                if key == "EMAIL_ENABLED":
                    config.email_enabled = value == "1"
                elif key == "EMAIL_RECIPIENT":
                    config.email_recipient = value
                elif key == "SLACK_ENABLED":
                    config.slack_enabled = value == "1"
                elif key == "SLACK_WEBHOOK_URL":
                    config.slack_webhook_url = value
                elif key.startswith("NOTIFY_"):
                    attr_name = key.lower()
                    if hasattr(config, attr_name):
                        setattr(config, attr_name, value == "1")

        return config


def send_email(subject: str, body: str, recipient: str) -> bool:
    """
    Send email using /usr/bin/mail command.

    Args:
        subject: Email subject line
        body: Email body (plain text)
        recipient: Recipient email address

    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Use mail command (available on cluster)
        proc = subprocess.run(
            ["mail", "-s", subject, recipient],
            input=body.encode(),
            capture_output=True,
            timeout=30
        )

        if proc.returncode == 0:
            print(f"✓ Email sent to {recipient}", file=sys.stderr)
            return True
        else:
            print(f"✗ Email failed (exit code {proc.returncode}): {proc.stderr.decode()}", file=sys.stderr)
            return False

    except FileNotFoundError:
        print("✗ Email failed: /usr/bin/mail not found", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("✗ Email failed: timeout after 30s", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Email failed: {e}", file=sys.stderr)
        return False


def send_slack(message: str, webhook_url: str, username: str = "Bootstrap Pipeline", icon_emoji: str = ":robot_face:") -> bool:
    """
    Send Slack notification via webhook.

    Args:
        message: Message text (markdown supported)
        webhook_url: Slack webhook URL
        username: Bot username
        icon_emoji: Bot icon emoji

    Returns:
        True if message sent successfully, False otherwise
    """
    try:
        payload = {
            "text": message,
            "username": username,
            "icon_emoji": icon_emoji
        }

        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                print("✓ Slack notification sent", file=sys.stderr)
                return True
            else:
                print(f"✗ Slack failed (HTTP {response.status})", file=sys.stderr)
                return False

    except urllib.error.URLError as e:
        print(f"✗ Slack failed: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Slack failed: {e}", file=sys.stderr)
        return False


def format_message(title: str, message: str, context: Optional[Dict] = None, severity: str = "info") -> str:
    """
    Format notification message with title, body, and context.

    Args:
        title: Message title
        message: Message body
        context: Additional context (dict of key-value pairs)
        severity: Severity level (info, warning, critical)

    Returns:
        Formatted message text
    """
    lines = []

    # Title with severity indicator
    severity_symbol = {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🚨"
    }.get(severity, "•")

    lines.append(f"{severity_symbol} {title}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(message)

    # Add context if provided
    if context:
        lines.append("")
        lines.append("Context:")
        for key, value in context.items():
            lines.append(f"  {key}: {value}")

    # Add timestamp and hostname
    lines.append("")
    lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Host: {os.getenv('HOSTNAME', 'unknown')}")

    return "\n".join(lines)


def should_notify_event(config: NotificationConfig, event_type: Optional[str], severity: str) -> bool:
    """
    Check whether an event should trigger notifications under current filters.

    Args:
        config: Notification configuration
        event_type: Logical event type (e.g., stage_b_submit, anomaly_warning)
        severity: Severity level (info, warning, critical)
    """
    if not event_type:
        return True

    event = event_type.strip().lower()
    if event == "anomaly":
        event = "anomaly_critical" if severity == "critical" else "anomaly_warning"

    filter_map = {
        "stage_a_complete": config.notify_stage_a_complete,
        "stage_a_validation": config.notify_stage_a_validation,
        "stage_b_submit": config.notify_stage_b_submit,
        "stage_b_complete": config.notify_stage_b_complete,
        "figure_complete": config.notify_figure_complete,
        "anomaly_warning": config.notify_anomaly_warning,
        "anomaly_critical": config.notify_anomaly_critical,
    }

    if event not in filter_map:
        print(f"ℹ️ Unknown event_type='{event_type}', sending notification by default", file=sys.stderr)
        return True

    return filter_map[event]


def notify(
    title: str,
    message: str,
    severity: str = "info",
    context: Optional[Dict] = None,
    config: Optional[NotificationConfig] = None,
    event_type: Optional[str] = None,
) -> Dict[str, bool]:
    """
    Send notification via all configured channels.

    Args:
        title: Notification title
        message: Notification message
        severity: Severity level (info, warning, critical)
        context: Additional context (dict)
        config: Configuration (if None, loaded from environment)
        event_type: Event type used for filter matching

    Returns:
        Dict with success status for each channel {"email": bool, "slack": bool}
    """
    if config is None:
        config = NotificationConfig.from_env()

    results = {"email": False, "slack": False}

    if not should_notify_event(config, event_type, severity):
        print(
            f"ℹ️ Notification suppressed by filter "
            f"(event_type={event_type}, severity={severity})",
            file=sys.stderr,
        )
        return results

    # Format message
    subject = f"[{severity.upper()}] {title}"
    body = format_message(title, message, context, severity)

    # Send email
    if config.email_enabled and config.email_recipient:
        results["email"] = send_email(subject, body, config.email_recipient)
    else:
        print("ℹ️ Email notifications disabled", file=sys.stderr)

    # Send Slack
    if config.slack_enabled and config.slack_webhook_url:
        # Slack message with formatting
        slack_message = f"*{subject}*\n\n{message}"
        if context:
            slack_message += "\n\n*Context:*\n" + "\n".join(f"• {k}: {v}" for k, v in context.items())

        results["slack"] = send_slack(slack_message, config.slack_webhook_url)
    else:
        print("ℹ️ Slack notifications disabled", file=sys.stderr)

    return results


def main():
    """CLI interface for testing notifications."""
    parser = argparse.ArgumentParser(description="Send pipeline notifications")
    parser.add_argument("--title", help="Notification title")
    parser.add_argument("--message", help="Notification message")
    parser.add_argument("--severity", choices=["info", "warning", "critical"], default="info")
    parser.add_argument("--context", help="Context as key=value,key2=value2")
    parser.add_argument(
        "--event-type",
        choices=[
            "stage_a_complete",
            "stage_a_validation",
            "stage_b_submit",
            "stage_b_complete",
            "figure_complete",
            "anomaly_warning",
            "anomaly_critical",
            "anomaly",
        ],
        help="Logical event type used for notification filters",
    )
    parser.add_argument("--test-email", metavar="EMAIL", help="Test email delivery")
    parser.add_argument("--test-slack", metavar="WEBHOOK", help="Test Slack delivery")
    parser.add_argument("--test", action="store_true", help="Test all configured channels")

    args = parser.parse_args()

    # Test mode
    if args.test:
        config = NotificationConfig.from_env()

        print("Testing notification configuration...")
        print(f"  Email enabled: {config.email_enabled}")
        print(f"  Email recipient: {config.email_recipient}")
        print(f"  Slack enabled: {config.slack_enabled}")
        print(f"  Slack webhook: {'configured' if config.slack_webhook_url else 'not configured'}")
        print()

        results = notify(
            title="Test Notification",
            message="This is a test from the bootstrap pipeline notification system.",
            severity="info",
            context={"test": "true", "timestamp": datetime.now().isoformat()},
            config=config
        )

        print()
        print("Results:")
        print(f"  Email: {'✓ sent' if results['email'] else '✗ failed'}")
        print(f"  Slack: {'✓ sent' if results['slack'] else '✗ failed'}")

        sys.exit(0 if any(results.values()) else 1)

    # Test email only
    if args.test_email:
        success = send_email(
            subject="Test Email from Bootstrap Pipeline",
            body="This is a test email.\n\nIf you receive this, email notifications are working correctly.",
            recipient=args.test_email
        )
        sys.exit(0 if success else 1)

    # Test Slack only
    if args.test_slack:
        success = send_slack(
            message="*Test Notification*\n\nThis is a test from the bootstrap pipeline.\n\nIf you see this, Slack notifications are working correctly.",
            webhook_url=args.test_slack
        )
        sys.exit(0 if success else 1)

    # Send notification
    if args.title and args.message:
        # Parse context
        context = None
        if args.context:
            context = {}
            for pair in args.context.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    context[key.strip()] = value.strip()

        results = notify(
            title=args.title,
            message=args.message,
            severity=args.severity,
            context=context,
            event_type=args.event_type,
        )

        sys.exit(0 if any(results.values()) else 1)

    # No valid arguments
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
