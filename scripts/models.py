"""Pydantic models for repository data and statistics."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RepoConfig(BaseModel):
    """Configuration for a single repository from repos.yaml."""

    owner: str
    repo: str
    description: str | None = None

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"


class CategoryConfig(BaseModel):
    """Configuration for a category of repositories."""

    name: str
    description: str | None = None
    repos: list[RepoConfig]


class ReposConfig(BaseModel):
    """Root configuration from repos.yaml."""

    categories: list[CategoryConfig]


class RepoSnapshot(BaseModel):
    """Point-in-time statistics for a repository."""

    stars: int
    last_commit: datetime
    fetched_at: datetime = Field(default_factory=lambda: datetime.now())
    archived: bool = False
    url: str | None = None


class RepoStatus(BaseModel):
    """Health status tracking for a repository."""

    state: Literal["active", "archived", "missing", "error"] = "active"
    last_success: datetime | None = None
    consecutive_failures: int = 0
    last_error: str | None = None


class HistorySnapshot(BaseModel):
    """A single historical snapshot with timestamp."""

    date: datetime
    repos: dict[str, RepoSnapshot]


class HistoryData(BaseModel):
    """Historical data storage structure."""

    snapshots: list[HistorySnapshot] = Field(default_factory=list)


class StatusData(BaseModel):
    """Status tracking for all repositories."""

    repos: dict[str, RepoStatus] = Field(default_factory=dict)
    last_run: datetime | None = None


class RepoStats(BaseModel):
    """Computed statistics for display in README."""

    owner: str
    repo: str
    description: str | None
    stars: int
    last_commit: datetime
    mom_growth: float | None = None  # Month-over-month percentage
    url: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    def format_stars(self) -> str:
        """Format star count with K suffix for thousands."""
        if self.stars >= 1000:
            return f"{self.stars / 1000:.1f}k"
        return str(self.stars)

    def format_last_commit(self) -> str:
        """Format last commit date."""
        return self.last_commit.strftime("%Y-%m-%d")

    def format_growth(self) -> str:
        """Format MoM growth with sign."""
        if self.mom_growth is None:
            return "N/A"
        if self.mom_growth > 0:
            return f"+{self.mom_growth:.1f}%"
        elif self.mom_growth < 0:
            return f"{self.mom_growth:.1f}%"
        return "0.0%"
