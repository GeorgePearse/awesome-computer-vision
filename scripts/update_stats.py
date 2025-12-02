#!/usr/bin/env python3
"""
Main entry point for updating awesome-computer-vision statistics.

Fetches GitHub stats for all repositories, calculates growth rates,
and generates an updated README.md.
"""

import asyncio
import sys
from pathlib import Path

import yaml

from .fetcher import fetch_all_repos, check_rate_limit_status
from .generator import generate_readme, generate_summary_report, save_readme
from .models import CategoryConfig, RepoConfig, RepoSnapshot, ReposConfig, RepoStats
from .storage import (
    add_snapshot,
    calculate_mom_growth,
    get_cached_snapshot,
    load_history,
    load_status,
    save_history,
    save_status,
    update_repo_status,
)

REPOS_FILE = Path(__file__).parent.parent / "data" / "repos.yaml"

# Abort if more than this percentage of repos fail
FAILURE_THRESHOLD = 0.5


def load_repos_config() -> ReposConfig:
    """Load repository configuration from YAML file."""
    with open(REPOS_FILE) as f:
        data = yaml.safe_load(f)
    return ReposConfig.model_validate(data)


def get_all_repos(config: ReposConfig) -> list[RepoConfig]:
    """Extract flat list of all repos from config."""
    repos = []
    for category in config.categories:
        repos.extend(category.repos)
    return repos


async def run_update() -> tuple[int, int, list[tuple[str, str]]]:
    """
    Run the full update process.

    Returns (total, successful, failed) counts.
    """
    print("Loading configuration...")
    config = load_repos_config()
    all_repos = get_all_repos(config)
    total = len(all_repos)

    print(f"Found {total} repositories across {len(config.categories)} categories")

    # Check rate limit status
    try:
        rate_status = await check_rate_limit_status()
        print(
            f"Rate limit: {rate_status['remaining']}/{rate_status['limit']} "
            f"(resets at {rate_status['reset'].strftime('%H:%M UTC')})"
        )
        if rate_status["remaining"] < total:
            print(
                f"WARNING: Only {rate_status['remaining']} API calls remaining, "
                f"but need {total}. Some repos may use cached data."
            )
    except Exception as e:
        print(f"Could not check rate limit: {e}")

    # Load existing data
    history = load_history()
    status = load_status()

    print(f"Historical snapshots: {len(history.snapshots)}")

    # Fetch all repos
    print("\nFetching repository stats...")
    results = await fetch_all_repos(all_repos)

    # Process results
    successful_snapshots: dict[str, RepoSnapshot] = {}
    failed_repos: list[tuple[str, str]] = []
    repo_stats: dict[str, RepoStats] = {}

    for repo in all_repos:
        repo_key = repo.full_name
        result = results.get(repo_key)

        if isinstance(result, RepoSnapshot):
            # Success
            successful_snapshots[repo_key] = result
            status = update_repo_status(
                status, repo_key, success=True, archived=result.archived
            )

            # Calculate MoM growth
            mom = calculate_mom_growth(history, repo_key, result.stars)

            repo_stats[repo_key] = RepoStats(
                owner=repo.owner,
                repo=repo.repo,
                description=repo.description,
                stars=result.stars,
                last_commit=result.last_commit,
                mom_growth=mom,
                url=result.url or repo.url,
            )
            print(f"  ✓ {repo_key}: {result.stars} stars")

        elif isinstance(result, Exception):
            # Failure - try to use cached data
            error_msg = str(result)
            failed_repos.append((repo_key, error_msg))
            status = update_repo_status(status, repo_key, success=False, error=error_msg)

            cached = get_cached_snapshot(history, repo_key)
            if cached:
                print(f"  ✗ {repo_key}: {error_msg} (using cached data)")
                mom = calculate_mom_growth(history, repo_key, cached.stars)
                repo_stats[repo_key] = RepoStats(
                    owner=repo.owner,
                    repo=repo.repo,
                    description=repo.description,
                    stars=cached.stars,
                    last_commit=cached.last_commit,
                    mom_growth=mom,
                    url=cached.url or repo.url,
                )
            else:
                print(f"  ✗ {repo_key}: {error_msg} (no cache)")

    successful = len(successful_snapshots)
    print(f"\nResults: {successful}/{total} successful, {len(failed_repos)} failed")

    # Check failure threshold
    failure_rate = len(failed_repos) / total if total > 0 else 0
    if failure_rate > FAILURE_THRESHOLD:
        print(
            f"\nERROR: Too many failures ({failure_rate:.0%} > {FAILURE_THRESHOLD:.0%}). "
            "Aborting update to preserve existing README."
        )
        return total, successful, failed_repos

    # Update history with successful fetches
    if successful_snapshots:
        history = add_snapshot(history, successful_snapshots)
        save_history(history)
        print(f"Saved history snapshot ({len(history.snapshots)} total)")

    # Save status
    save_status(status)

    # Generate README
    print("\nGenerating README.md...")
    readme_content = generate_readme(
        config.categories,
        repo_stats,
        failed_repos=[r[0] for r in failed_repos] if failed_repos else None,
    )
    save_readme(readme_content)
    print("README.md updated successfully!")

    return total, successful, failed_repos


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("Awesome Computer Vision - Stats Update")
    print("=" * 60)
    print()

    try:
        total, successful, failed = asyncio.run(run_update())

        # Print summary for GitHub Actions
        summary = generate_summary_report(total, successful, failed)
        print("\n" + summary)

        # Exit with error if there were failures
        if failed:
            return 1
        return 0

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
