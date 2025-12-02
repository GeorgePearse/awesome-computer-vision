"""GitHub API client with retry logic and rate limit handling."""

import asyncio
import os
from datetime import datetime, timezone

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import RepoConfig, RepoSnapshot

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


class RateLimitExceeded(Exception):
    """Raised when GitHub API rate limit is exceeded."""

    def __init__(self, reset_time: datetime):
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded. Resets at {reset_time}")


class RepoNotFound(Exception):
    """Raised when a repository is not found."""

    pass


def get_headers() -> dict[str, str]:
    """Return headers for GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "awesome-computer-vision-stats",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def check_rate_limit(response: httpx.Response) -> None:
    """Check rate limit headers and raise if exceeded."""
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining and int(remaining) == 0:
        reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))
        reset_time = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
        raise RateLimitExceeded(reset_time)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
)
async def fetch_repo_stats(
    client: httpx.AsyncClient, owner: str, repo: str
) -> RepoSnapshot:
    """Fetch repository statistics from GitHub API."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}"

    response = await client.get(url, headers=get_headers(), timeout=30.0)

    # Handle rate limiting
    if response.status_code == 403:
        check_rate_limit(response)
        # If not rate limit, re-raise
        response.raise_for_status()

    # Handle not found
    if response.status_code == 404:
        raise RepoNotFound(f"Repository {owner}/{repo} not found")

    response.raise_for_status()
    data = response.json()

    # Parse last commit date
    pushed_at = data.get("pushed_at")
    if pushed_at:
        last_commit = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
    else:
        last_commit = datetime.now(timezone.utc)

    return RepoSnapshot(
        stars=data["stargazers_count"],
        last_commit=last_commit,
        fetched_at=datetime.now(timezone.utc),
        archived=data.get("archived", False),
        url=data["html_url"],
    )


async def fetch_all_repos(
    repos: list[RepoConfig],
    concurrency: int = 5,
) -> dict[str, RepoSnapshot | Exception]:
    """
    Fetch stats for all repositories with controlled concurrency.

    Returns a dict mapping repo full_name to either RepoSnapshot or Exception.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, RepoSnapshot | Exception] = {}

    async def fetch_one(repo: RepoConfig) -> None:
        async with semaphore:
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    snapshot = await fetch_repo_stats(client, repo.owner, repo.repo)
                    results[repo.full_name] = snapshot
            except Exception as e:
                results[repo.full_name] = e

    # Run all fetches concurrently
    tasks = [fetch_one(repo) for repo in repos]
    await asyncio.gather(*tasks)

    return results


async def check_rate_limit_status() -> dict:
    """Check current rate limit status."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GITHUB_API}/rate_limit",
            headers=get_headers(),
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        core = data["resources"]["core"]
        return {
            "limit": core["limit"],
            "remaining": core["remaining"],
            "reset": datetime.fromtimestamp(core["reset"], tz=timezone.utc),
        }
