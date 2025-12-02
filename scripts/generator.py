"""README markdown generation from repository statistics."""

from datetime import datetime, timezone
from pathlib import Path

from .models import CategoryConfig, RepoStats

README_FILE = Path(__file__).parent.parent / "README.md"


def generate_readme(
    categories: list[CategoryConfig],
    stats: dict[str, RepoStats],
    failed_repos: list[str] | None = None,
) -> str:
    """Generate complete README.md content."""
    lines = [
        "# Awesome Computer Vision",
        "",
        "[![Update Stats](https://github.com/GeorgePearse/awesome-computer-vision/actions/workflows/update-stats.yml/badge.svg)](https://github.com/GeorgePearse/awesome-computer-vision/actions/workflows/update-stats.yml)",
        "",
        "A curated list of awesome computer vision tools, libraries, and frameworks with live GitHub statistics.",
        "",
        f"*Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "## Contents",
        "",
    ]

    # Table of contents
    for cat in categories:
        anchor = (
            cat.name.lower()
            .replace(" ", "-")
            .replace("(", "")
            .replace(")", "")
        )
        lines.append(f"- [{cat.name}](#{anchor})")

    lines.extend(["", "---", ""])

    # Category tables
    for cat in categories:
        lines.append(f"## {cat.name}")
        lines.append("")

        if cat.description:
            lines.append(f"*{cat.description}*")
            lines.append("")

        lines.append("| Repository | Stars | Last Commit | MoM Growth |")
        lines.append("|------------|------:|:-----------:|:----------:|")

        for repo_config in cat.repos:
            repo_key = repo_config.full_name
            repo_stats = stats.get(repo_key)

            if repo_stats:
                # Build the row with stats
                repo_link = f"[{repo_config.repo}]({repo_stats.url})"
                desc = repo_stats.description or ""
                if desc:
                    repo_link = f"{repo_link} - {desc}"

                row = (
                    f"| {repo_link} "
                    f"| {repo_stats.format_stars()} "
                    f"| {repo_stats.format_last_commit()} "
                    f"| {repo_stats.format_growth()} |"
                )
            else:
                # Fallback for failed fetches
                row = f"| {repo_config.repo} | - | - | - |"

            lines.append(row)

        lines.append("")

    # Footer
    lines.extend(
        [
            "---",
            "",
            "## Contributing",
            "",
            "To add a new repository:",
            "",
            "1. Edit `data/repos.yaml`",
            "2. Add your entry under the appropriate category",
            "3. Submit a pull request",
            "",
            "## Stats",
            "",
            "- Statistics are automatically updated weekly via GitHub Actions",
            "- MoM Growth = Month-over-month percentage change in stars",
            "- Last Commit = Date of most recent push to the repository",
            "",
        ]
    )

    # Add metadata comment for debugging
    if failed_repos:
        lines.append(f"<!-- Failed repos ({len(failed_repos)}): {', '.join(failed_repos)} -->")

    lines.append(
        f"<!-- Generated: {datetime.now(timezone.utc).isoformat()} -->"
    )

    return "\n".join(lines)


def save_readme(content: str) -> None:
    """Save README content to file."""
    with open(README_FILE, "w") as f:
        f.write(content)


def generate_summary_report(
    total: int,
    successful: int,
    failed: list[tuple[str, str]],
) -> str:
    """Generate summary for GitHub Actions step summary."""
    lines = [
        "## Stats Update Summary",
        "",
        f"- **Total repos:** {total}",
        f"- **Successful:** {successful}",
        f"- **Failed:** {len(failed)}",
        "",
    ]

    if failed:
        lines.append("### Failed Repositories")
        lines.append("")
        for repo, error in failed:
            lines.append(f"- `{repo}`: {error}")

    return "\n".join(lines)
