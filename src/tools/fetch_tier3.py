"""Fetch Tier 3 source documents from their official sources.

55 of the 247 corpus documents (notably from FATF, MAS, and IRAS) cannot be
redistributed under their source organisations' licensing terms. The release
includes metadata, SHA-256 checksums, and source URLs for these documents in
`data/corpus/tier3_metadata_only/manifest.csv`, but not their content.

This tool retrieves each Tier 3 document from its source URL, verifies the
SHA-256 against the published checksum, and writes the document to
`data/corpus/tier3_metadata_only/<sha256_prefix>/<filename>`.

Usage:
    python src/tools/fetch_tier3.py [--output <dir>] [--retries N] [--workers N]
                                    [--source FATF|MAS|IRAS] [--dry-run]

Behaviour:
    - Skips documents already present locally with matching SHA-256.
    - Retries transient failures with exponential backoff (default: 3 attempts).
    - Verifies the SHA-256 of every fetched document; mismatches are quarantined
      under `<output>/_sha256_mismatch/` for manual inspection.
    - Logs every fetch outcome to `<output>/fetch_log.jsonl`.
    - On --dry-run, prints the planned actions without making any network calls.

Known limitations:
    - **FATF** (19 of 55 Tier 3 documents) actively blocks automated requests
      with HTTP 403 even when using browser-equivalent User-Agent headers.
      FATF documents must be retrieved manually via a browser, or via the
      Wayback Machine archive. The fetch tool will surface these as
      `http_error` entries; this is the script working correctly, not a bug.
    - **MAS** and **IRAS** (36 of 55 Tier 3 documents) are reachable with
      browser-equivalent User-Agent strings; pass `--user-agent "Mozilla/5.0"`
      if the default tool UA is blocked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "corpus" / "tier3_metadata_only" / "manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "corpus" / "tier3_metadata_only"
DEFAULT_USER_AGENT = "RegConflict-FetchTool/0.1 (academic dataset reconstruction)"


@dataclass
class FetchResult:
    sha256: str
    filename: str
    source_url: str
    issuing_body: str
    status: str  # "ok", "skipped", "sha256_mismatch", "http_error", "network_error", "no_url", "dry_run"
    bytes_written: int = 0
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path, source_filter: Optional[str]) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if source_filter:
        rows = [r for r in rows if source_filter.upper() in (r.get("issuing_body", "") + r.get("source_collection", "")).upper()]
    return rows


def _target_path(output_dir: Path, sha256: str, filename: str) -> Path:
    return output_dir / sha256[:8] / filename


def _fetch_one(
    row: dict,
    output_dir: Path,
    *,
    retries: int,
    user_agent: str,
    timeout_seconds: float,
    dry_run: bool,
) -> FetchResult:
    expected_sha = (row.get("sha256") or "").strip().lower()
    filename = (row.get("document_title") or "untitled").strip()
    if not filename.lower().endswith((".pdf", ".html", ".htm")):
        # Default extension for documents whose title field doesn't include one.
        filename = f"{filename}.pdf"
    source_url = (row.get("source_url") or "").strip()
    issuing_body = (row.get("issuing_body") or "?").strip()

    if not source_url:
        return FetchResult(expected_sha, filename, "", issuing_body, "no_url",
                           error="manifest row has no source_url; document cannot be fetched")

    target = _target_path(output_dir, expected_sha, filename)
    if target.exists():
        actual_sha = _sha256_file(target)
        if actual_sha == expected_sha:
            return FetchResult(expected_sha, filename, source_url, issuing_body, "skipped",
                               bytes_written=target.stat().st_size)
        # SHA mismatch on existing file — quarantine and refetch
        quarantine = output_dir / "_sha256_mismatch" / target.name
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        target.rename(quarantine)

    if dry_run:
        return FetchResult(expected_sha, filename, source_url, issuing_body, "dry_run")

    start = time.monotonic()
    last_err: Optional[str] = None
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=timeout_seconds, follow_redirects=True,
                              headers={"User-Agent": user_agent}) as client:
                resp = client.get(source_url)
                resp.raise_for_status()
                content = resp.content
            actual_sha = hashlib.sha256(content).hexdigest()
            if actual_sha != expected_sha:
                quarantine = output_dir / "_sha256_mismatch" / target.name
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                quarantine.write_bytes(content)
                return FetchResult(
                    expected_sha, filename, source_url, issuing_body, "sha256_mismatch",
                    bytes_written=len(content),
                    error=(f"fetched SHA-256 {actual_sha} does not match expected "
                           f"{expected_sha}; content saved to {quarantine}"),
                    elapsed_seconds=time.monotonic() - start,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            return FetchResult(expected_sha, filename, source_url, issuing_body, "ok",
                               bytes_written=len(content),
                               elapsed_seconds=time.monotonic() - start)
        except httpx.HTTPStatusError as exc:
            last_err = f"HTTP {exc.response.status_code} {exc.response.reason_phrase}"
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                # 4xx (except rate limit) are not retryable; the URL has drifted.
                return FetchResult(expected_sha, filename, source_url, issuing_body,
                                   "http_error", error=last_err,
                                   elapsed_seconds=time.monotonic() - start)
        except (httpx.RequestError, httpx.TransportError) as exc:
            last_err = f"network error: {exc}"
        if attempt < retries:
            time.sleep(backoff)
            backoff *= 2

    return FetchResult(expected_sha, filename, source_url, issuing_body, "network_error",
                       error=last_err or "exhausted retries",
                       elapsed_seconds=time.monotonic() - start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"Tier-3 manifest CSV (default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})")
    parser.add_argument("--retries", type=int, default=3, help="Retries per document (default: 3)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent fetches (default: 4)")
    parser.add_argument("--source", type=str, default=None,
                        help="Filter to a single source organisation (FATF, MAS, IRAS)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    parser.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned actions without making network calls")
    args = parser.parse_args()

    rows = _load_manifest(args.manifest, args.source)
    if not rows:
        print(f"No manifest rows matched filter source={args.source!r}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "fetch_log.jsonl"

    def _display(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    print(f"Fetching {len(rows)} Tier 3 document(s); output={_display(args.output)}; "
          f"workers={args.workers}; retries={args.retries}; dry_run={args.dry_run}")
    print()

    results: list[FetchResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_fetch_one, row, args.output,
                               retries=args.retries, user_agent=args.user_agent,
                               timeout_seconds=args.timeout, dry_run=args.dry_run)
                   for row in rows]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            sym = {"ok": "✓", "skipped": "•", "dry_run": "?", "no_url": "✗",
                   "sha256_mismatch": "✗", "http_error": "✗", "network_error": "✗"}.get(r.status, "?")
            line = f"  {sym} [{r.status:>16}] {r.issuing_body[:10]:>10}  {r.filename}"
            if r.error:
                line += f"\n        {r.error}"
            print(line)

    with log_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")

    summary: dict[str, int] = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1
    print()
    print("Summary:")
    for k in sorted(summary):
        print(f"  {k:>16}: {summary[k]}")
    print(f"Log: {_display(log_path)}")

    failure_states = {"sha256_mismatch", "http_error", "network_error", "no_url"}
    return 0 if not any(r.status in failure_states for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
