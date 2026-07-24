"""Filter and fix a browser-exported cookies.txt for yt-dlp YouTube downloads."""

from __future__ import annotations

import argparse
from http.cookiejar import LoadError, MozillaCookieJar
from pathlib import Path

YOUTUBE_DOMAINS = frozenset({"youtube.com", "accounts.google.com"})


def _domain_ok(domain: str) -> bool:
    bare = domain.lstrip(".")
    return bare in YOUTUBE_DOMAINS or bare.endswith(".youtube.com")


def _fix_line(line: str) -> str | None:
    parts = line.split("\t")
    if len(parts) != 7:
        return None
    domain, subdomains, path, secure, expiry, name, value = parts
    if not _domain_ok(domain):
        return None
    subdomains = "TRUE" if domain.startswith(".") else "FALSE"
    return "\t".join([domain, subdomains, path, secure, expiry, name, value])


def sanitize_cookies(src: Path, dst: Path) -> int:
    lines: list[str] = []
    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fixed = _fix_line(line)
        if fixed:
            lines.append(fixed)

    header = "# Netscape HTTP Cookie File\n# Filtered for YouTube\n"
    dst.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")

    jar = MozillaCookieJar(str(dst))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except LoadError as exc:
        raise SystemExit(f"Sanitized file still invalid: {exc}") from exc

    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="data/cookies.txt",
        help="Raw cookies export (default: data/cookies.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/youtube_cookies.txt",
        help="Sanitized output path (default: data/youtube_cookies.txt)",
    )
    args = parser.parse_args()
    count = sanitize_cookies(Path(args.input), Path(args.output))
    print(f"Wrote {count} cookies to {args.output}")


if __name__ == "__main__":
    main()
