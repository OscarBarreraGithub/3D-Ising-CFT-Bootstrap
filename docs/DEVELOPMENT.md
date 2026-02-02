# Development Workflow

This document describes the development setup for working with parallel Claude Code instances using git worktrees.

## Overview

This project is configured for **parallel development** using:
- **Git worktrees**: Multiple working directories sharing the same repository
- **Claude Code agents**: Automated code review and coordination
- **Context management**: Auto-compaction to prevent context degradation

## Quick Start

### Prerequisites

- Python 3.11
- Git
- Claude Code CLI

### Initial Setup

```bash
# Clone and enter repo
cd "3D Ising CFT Bootstrap"

# Option 1: Conda (recommended)
conda env create -f environment.yml
conda activate ising_bootstrap

# Option 2: pip
pip install -e .[dev]

# Verify installation
pytest tests/ -v
ising-stage-a --help
```

## Parallel Development with Worktrees

### Why Worktrees?

Git worktrees allow multiple Claude Code instances to work simultaneously on different features without file conflicts. Each worktree:
- Has its own independent file state
- Shares git history with the main repo
- Can run its own Claude Code session
- Uses the same `.claude/` configuration

### Creating a Worktree

```bash
# From the main repo directory
make worktree NAME=blocks-implementation

# Or manually:
git worktree add ../ising-blocks -b feature/blocks
cd ../ising-blocks
pip install -e .  # or conda activate ising_bootstrap
```

### Working in a Worktree

```bash
cd ../ising-blocks
claude  # Start Claude Code session

# Work on your feature...
# When done, commit your changes:
git add -A
git commit -m "Implement conformal block computation"
```

### Merging Completed Work

1. **Signal completion** in your worktree session: "ready for review"
2. The **review-coordinator** agent will orchestrate the review process
3. After approval, merge from main:

```bash
# From main repo
git merge feature/blocks
git branch -d feature/blocks
git worktree remove ../ising-blocks
```

## Code Review Workflow

### Automated Review Process

When you signal that work is complete, the review process follows this flow:

```
Author Claude → Review Coordinator → Branch Code Reviewer
                       ↓
              Feedback presented to Author
                       ↓
              Author defends or fixes
                       ↓
              Re-review until APPROVED
```

### Review Agents

| Agent | Purpose |
|-------|---------|
| `branch-code-reviewer` | Performs rigorous code quality review |
| `review-coordinator` | Orchestrates dialogue between author and reviewer |

### Triggering Review

In your Claude session, simply say:
- "Ready for review"
- "I've finished implementing X"
- "Please review my changes"

The coordinator will automatically:
1. Analyze your branch diff against main
2. Spawn the reviewer agent
3. Present feedback for you to address
4. Iterate until approved or escalate to user

## Context Management

### Auto-Compaction

Context auto-compacts at **80%** capacity (earlier than the default 95%) to prevent context degradation during long sessions.

When compaction occurs, you'll see:
```
[CONTEXT] Auto-compacting at <timestamp>. Key context may be summarized.
```

### Best Practices

1. **Check context usage**: Run `/context` periodically
2. **Manual compaction**: Use `/compact "focus on X"` for targeted summarization
3. **Persistent instructions**: Key project info is in `CLAUDE.md` (survives compaction)
4. **Subagents for research**: Spawn agents for exploration tasks (they get fresh context)

### What Survives Compaction

- `CLAUDE.md` content (always re-read)
- Recent tool outputs and code changes
- Summarized conversation history

### What May Be Lost

- Detailed explanations from early in the session
- Specific file contents read long ago
- Nuanced decisions made early on

**Tip**: If working on a complex feature, document key decisions in comments or commit messages.

## Project Structure

```
.
├── CLAUDE.md                    # Project instructions for Claude
├── Makefile                     # Dev commands (install, test, worktree)
├── pyproject.toml               # Python package config (source of truth)
├── environment.yml              # Conda environment (references pyproject.toml)
├── requirements.lock            # Pinned dependencies for reproducibility
│
├── .claude/
│   ├── settings.json            # Shared Claude Code settings
│   ├── settings.local.json      # Personal overrides (gitignored)
│   ├── agents/
│   │   ├── branch-code-reviewer.md
│   │   └── review-coordinator.md
│   └── hooks/
│       └── notify-complete.sh   # macOS notifications
│
├── src/ising_bootstrap/         # Main package
├── tests/                       # Test suite
├── docs/                        # Documentation
├── data/                        # Generated data (gitignored)
└── figures/                     # Generated figures (gitignored)
```

## Configuration Files

### `.claude/settings.json`

Shared settings for all developers:
- **Auto-compact at 80%**: Prevents context rot
- **Pre-allowed commands**: git, pytest, pip, conda
- **Hooks**: Notifications, compaction alerts

### `.claude/settings.local.json`

Personal overrides (gitignored):
- Additional permissions for your workflow
- Custom hook configurations

### `CLAUDE.md`

Project-specific instructions that Claude reads at session start:
- Project overview and goals
- Key files and their purposes
- Code standards and conventions
- Common pitfalls to avoid

## Makefile Commands

```bash
make install      # Install package (uses lock file if available)
make install-dev  # Install with dev dependencies
make lock         # Generate requirements.lock from pyproject.toml
make test         # Run pytest
make clean        # Remove build artifacts
make worktree NAME=feature-name  # Create new worktree
```

## Dependency Management

### Single Source of Truth

All dependencies are defined in `pyproject.toml`. Other files reference it:

- `environment.yml`: `pip: -e .[dev]`
- `requirements.lock`: Auto-generated via `make lock`

### Updating Dependencies

```bash
# Edit pyproject.toml, then:
make lock  # Regenerate lock file
```

### Reproducible Installs

For cluster deployment or CI, use the lock file:
```bash
pip install -r requirements.lock
pip install -e . --no-deps
```

## Troubleshooting

### Worktree Issues

**Problem**: Can't create worktree - branch already exists
```bash
git branch -d feature/name  # Delete local branch first
```

**Problem**: Worktree shows wrong files
```bash
git worktree repair  # Fix worktree metadata
```

### Context Issues

**Problem**: Claude seems to have forgotten important context
```bash
/context  # Check usage
/compact "focus on the LP solver implementation"  # Targeted compaction
```

**Problem**: Key instructions not being followed
- Check that `CLAUDE.md` exists and contains the instructions
- Instructions in `CLAUDE.md` are re-read after compaction

### Review Issues

**Problem**: Review coordinator not responding
- Ensure you're on a feature branch (not main)
- Verify `.claude/agents/` contains the agent files
- Check that agents are properly formatted (YAML frontmatter)

## See Also

- [CLUSTER_SETUP.md](CLUSTER_SETUP.md) - FASRC cluster deployment
- [RUN.md](RUN.md) - Running scans and generating figures
- [implementation_choices.md](implementation_choices.md) - Technical design decisions
