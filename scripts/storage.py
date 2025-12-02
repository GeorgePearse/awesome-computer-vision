"""Historical data storage and month-over-month growth calculation."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import HistoryData, HistorySnapshot, RepoSnapshot, StatusData, RepoStatus

# Storage paths
DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_FILE = DATA_DIR / "history.json"
STATUS_FILE = DATA_DIR / "status.json"

# Keep approximately 60 weekly snapshots (1+ year of data)
MAX_SNAPSHOTS = 60


def load_history() -> HistoryData:
    """Load historical data from JSON file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                data = json.load(f)
                return HistoryData.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            # Corrupted file, start fresh
            return HistoryData()
    return HistoryData()


def save_history(history: HistoryData) -> None:
    """Save historical data to JSON file with atomic write."""
    DATA_DIR.mkdir(exist_ok=True)

    # Write to temp file first, then rename (atomic on POSIX)
    temp_file = HISTORY_FILE.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(history.model_dump(mode="json"), f, indent=2, default=str)

    temp_file.rename(HISTORY_FILE)


def add_snapshot(
    history: HistoryData, current_stats: dict[str, RepoSnapshot]
) -> HistoryData:
    """Add current stats as a new snapshot."""
    snapshot = HistorySnapshot(
        date=datetime.now(timezone.utc),
        repos=current_stats,
    )
    history.snapshots.append(snapshot)

    # Prune old snapshots
    if len(history.snapshots) > MAX_SNAPSHOTS:
        history.snapshots = history.snapshots[-MAX_SNAPSHOTS:]

    return history


def calculate_mom_growth(
    history: HistoryData, repo_key: str, current_stars: int
) -> float | None:
    """
    Calculate month-over-month growth rate.

    Looks for a snapshot from approximately 30 days ago and computes
    the percentage change in stars.

    Returns None if insufficient historical data.
    """
    if not history.snapshots:
        return None

    now = datetime.now(timezone.utc)
    target_date = now - timedelta(days=30)

    # Find the snapshot closest to 30 days ago (within a tolerance window)
    best_snapshot: HistorySnapshot | None = None
    best_diff = timedelta(days=999)

    for snapshot in history.snapshots:
        # Make sure snapshot date is timezone aware
        snap_date = snapshot.date
        if snap_date.tzinfo is None:
            snap_date = snap_date.replace(tzinfo=timezone.utc)

        diff = abs(snap_date - target_date)

        # Must be at least 20 days ago and within 15 days of target
        days_ago = (now - snap_date).days
        if days_ago >= 20 and diff < best_diff and diff <= timedelta(days=15):
            best_diff = diff
            best_snapshot = snapshot

    if best_snapshot is None:
        return None

    old_data = best_snapshot.repos.get(repo_key)
    if old_data is None:
        return None

    old_stars = old_data.stars
    if old_stars == 0:
        return None

    growth = ((current_stars - old_stars) / old_stars) * 100
    return round(growth, 2)


def load_status() -> StatusData:
    """Load repository status data."""
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE) as f:
                data = json.load(f)
                return StatusData.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return StatusData()
    return StatusData()


def save_status(status: StatusData) -> None:
    """Save repository status data."""
    DATA_DIR.mkdir(exist_ok=True)

    temp_file = STATUS_FILE.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(status.model_dump(mode="json"), f, indent=2, default=str)

    temp_file.rename(STATUS_FILE)


def update_repo_status(
    status: StatusData,
    repo_key: str,
    success: bool,
    error: str | None = None,
    archived: bool = False,
) -> StatusData:
    """Update status for a single repository."""
    if repo_key not in status.repos:
        status.repos[repo_key] = RepoStatus()

    repo_status = status.repos[repo_key]

    if success:
        repo_status.state = "archived" if archived else "active"
        repo_status.last_success = datetime.now(timezone.utc)
        repo_status.consecutive_failures = 0
        repo_status.last_error = None
    else:
        repo_status.consecutive_failures += 1
        repo_status.last_error = error

        # Mark as missing or error based on error type
        if "not found" in (error or "").lower():
            repo_status.state = "missing"
        else:
            repo_status.state = "error"

    return status


def get_cached_snapshot(
    history: HistoryData, repo_key: str
) -> RepoSnapshot | None:
    """Get the most recent cached snapshot for a repo (fallback on failure)."""
    for snapshot in reversed(history.snapshots):
        if repo_key in snapshot.repos:
            return snapshot.repos[repo_key]
    return None
