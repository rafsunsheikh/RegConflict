"""End-to-end zero-shot evaluation: prompt → LLM → parse → retry → predict.

Retry escalation per spec: temperature 0.0 → 0.3 → 0.7 (only triggered on
parsing_failure of the previous attempt). Records `n_retries_used` and
`succeeded_at_temperature` for every prediction.

Concurrency: predict_split takes a `concurrency` arg. With concurrency>1 the
records are dispatched through a ThreadPoolExecutor, all hitting the same
llama-server endpoint (which has n_slots=4). Two concurrent requests per
server is the safe default — preserves a 2-slot margin against contention.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from src.baselines.zero_shot.client import LLMClient
from src.baselines.zero_shot.parser import ParsedPrediction, parse
from src.baselines.zero_shot.prompts import render

RETRY_TEMPERATURES = (0.0, 0.3, 0.7)
MAX_TOKENS = 2048  # default; can be overridden per-call via predict_one(... max_tokens=)


def _config_hash(model_key: str, variant: str) -> str:
    """Stable identifier per (model, variant) for the harness's consistency check."""
    raw = f"zero_shot_v1|{model_key}|variant_{variant}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _build_prediction_record(
    *,
    client: LLMClient,
    variant: str,
    record: dict,
    attempts: list[dict],
    parsed: ParsedPrediction | None,
    succeeded_at_temperature: float | None,
    n_retries_used: int,
) -> dict:
    """Construct the harness-schema prediction dict from the attempt list.

    Shared by the live-inference path and the resume-from-transcript path.
    """
    first_attempt = attempts[0] if attempts else {}
    final_attempt = attempts[-1] if attempts else {}
    if parsed is not None and parsed.success:
        success_attempt = next(
            (a for a in attempts if a.get("parsed", {}).get("success")), final_attempt
        )
    else:
        success_attempt = final_attempt

    base_metadata = {
        "model_name": f"zero_shot_{client.model_key}_variant_{variant}",
        "model_version": "1.0",
        "config_hash": _config_hash(client.model_key, variant),
        "underlying_llm": client.model_key,
        "prompt_variant": variant,
    }

    common = {
        "pair_id": record["pair_id"],
        "confidence_score": None,
        "n_retries_used": n_retries_used if parsed and parsed.success else max(0, len(attempts) - 1),
        "succeeded_at_temperature": succeeded_at_temperature,
        "reasoning_content": success_attempt.get("reasoning_content") or "",
        "raw_response": success_attempt.get("content") or "",
        "reasoning_content_first_attempt": first_attempt.get("reasoning_content") or "",
        "raw_response_first_attempt": first_attempt.get("content") or "",
        "total_latency_seconds": sum(a.get("latency_seconds", 0.0) for a in attempts),
        "model_metadata": base_metadata,
    }

    if parsed is None or not parsed.success:
        last_failure = final_attempt.get("parsed", {}) or {}
        common.update({
            "predicted_conflict_present": False,
            "predicted_conflict_type": None,
            "parsing_status": "parsing_failure",
            "parsing_failure_reason": last_failure.get("failure_reason") or "all_retries_failed",
            "parsing_failure_detail": last_failure.get("failure_detail"),
            "rationale_from_llm": None,
        })
        return common

    common.update({
        "predicted_conflict_present": parsed.conflict_present,
        "predicted_conflict_type": parsed.conflict_type,
        "parsing_status": "success",
        "parsing_failure_reason": None,
        "parsing_failure_detail": None,
        "rationale_from_llm": parsed.rationale,
    })
    return common


def predict_one(
    client: LLMClient,
    variant: str,
    record: dict,
    raw_responses_dir: Path,
    max_tokens: int = MAX_TOKENS,
    resume_from_transcript: bool = True,
) -> dict:
    """Run a single conflict pair through the model, with retry escalation.

    Returns a prediction dict matching the harness JSONL schema, plus extra
    fields for error analysis (parsing_status, n_retries_used, etc.).

    Resume: if a transcript file already exists in raw_responses_dir for this
    pair_id, reconstruct the prediction from it without re-calling the LLM.
    This makes re-launches after a crash efficient (cached records reused).

    Exception handling: httpx ReadTimeout / network errors during chat() are
    caught and recorded as a synthetic "client_timeout" attempt rather than
    raised. Retry escalation continues at the next temperature; if all 3
    attempts fail (parse or timeout), the prediction is returned as
    parsing_failure. This makes the runner robust to single-call failures.
    """
    import httpx  # local import to avoid top-level dependency on httpx exceptions

    pair_id_safe = record["pair_id"].replace(":", "_").replace("/", "_")
    transcript_path = raw_responses_dir / f"{pair_id_safe}.json"

    # --- Resume from existing transcript if one is present ---
    if resume_from_transcript and transcript_path.exists():
        try:
            cached = json.loads(transcript_path.read_text())
            cached_attempts = cached.get("attempts") or []
            # Reconstruct ParsedPrediction list from cached attempts
            attempts = cached_attempts
            parsed = None
            succeeded_at_temperature = None
            n_retries_used = len(cached_attempts) - 1
            for i, a in enumerate(cached_attempts):
                pdict = a.get("parsed") or {}
                if pdict.get("success"):
                    # Rehydrate a ParsedPrediction-like object
                    parsed = ParsedPrediction(
                        success=True,
                        conflict_present=pdict.get("conflict_present"),
                        conflict_type=pdict.get("conflict_type"),
                        rationale=pdict.get("rationale"),
                    )
                    succeeded_at_temperature = a.get("temperature")
                    n_retries_used = i
                    break
            return _build_prediction_record(
                client=client, variant=variant, record=record,
                attempts=attempts, parsed=parsed,
                succeeded_at_temperature=succeeded_at_temperature,
                n_retries_used=n_retries_used,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Transcript corrupt → fall through to fresh inference
            pass

    system, user = render(variant, record)
    attempts: list[dict] = []
    parsed: ParsedPrediction | None = None
    n_retries_used = 0
    succeeded_at_temperature: float | None = None

    for i, temp in enumerate(RETRY_TEMPERATURES):
        # --- Call the LLM, capturing transient errors as synthetic failures ---
        try:
            resp = client.chat(system, user, temperature=temp, max_tokens=max_tokens)
            content = resp.content
            reasoning_content = resp.reasoning_content
            finish_reason = resp.finish_reason
            usage = resp.usage
            latency = resp.latency_seconds
            client_error = None
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError,
                httpx.RemoteProtocolError, httpx.ConnectError) as e:
            content = ""
            reasoning_content = ""
            finish_reason = "client_error"
            usage = {}
            latency = 0.0
            client_error = f"{type(e).__name__}: {e}"

        parsed_attempt = parse(content, variant) if content else ParsedPrediction(
            success=False, failure_reason="client_timeout",
            failure_detail=client_error or "no content returned",
        )

        attempts.append({
            "attempt_n": i + 1,
            "temperature": temp,
            "latency_seconds": latency,
            "finish_reason": finish_reason,
            "usage": usage,
            "content": content,
            "reasoning_content": reasoning_content,
            "client_error": client_error,
            "parsed": parsed_attempt.to_dict(),
        })
        if parsed_attempt.success:
            parsed = parsed_attempt
            succeeded_at_temperature = temp
            n_retries_used = i
            break

    # Persist the raw transcript for analysis (full request, all attempts).
    raw_responses_dir.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "pair_id": record["pair_id"],
                "model_key": client.model_key,
                "variant": variant,
                "system_prompt": system,
                "user_prompt": user,
                "attempts": attempts,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    return _build_prediction_record(
        client=client, variant=variant, record=record,
        attempts=attempts, parsed=parsed,
        succeeded_at_temperature=succeeded_at_temperature,
        n_retries_used=n_retries_used,
    )


def predict_split(
    client: LLMClient,
    variant: str,
    records: list[dict],
    output_dir: Path,
    progress_every: int = 5,
    concurrency: int = 1,
    max_tokens: int = MAX_TOKENS,
) -> list[dict]:
    """Run a full list of records through the model. Saves raw_responses to disk.

    Predictions are returned in the SAME ORDER as `records` (parallel dispatch
    plus post-hoc ordering by index — easier to reason about than depending on
    completion order).

    concurrency>1 dispatches that many records through the same server
    concurrently. llama-server defaults to n_slots=4; the runner spec uses 2
    as the safe default so a couple of slots remain for any other tooling.
    """
    raw_dir = output_dir / "raw_responses"
    preds: list[dict | None] = [None] * len(records)
    completed = 0
    completed_lock = Lock()
    t0 = time.perf_counter()
    n = len(records)

    def _run(idx: int, rec: dict):
        nonlocal completed
        try:
            return idx, predict_one(client, variant, rec, raw_dir, max_tokens=max_tokens)
        finally:
            with completed_lock:
                completed += 1
                if completed % progress_every == 0 or completed == n:
                    elapsed = time.perf_counter() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (n - completed) / rate if rate > 0 else 0
                    print(
                        f"  [{completed}/{n}] elapsed={elapsed:.0f}s "
                        f"rate={rate:.2f}/s eta={eta:.0f}s",
                        flush=True,
                    )

    if concurrency <= 1:
        for idx, rec in enumerate(records):
            _, p = _run(idx, rec)
            preds[idx] = p
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_run, idx, rec) for idx, rec in enumerate(records)]
            for fut in as_completed(futures):
                idx, p = fut.result()
                preds[idx] = p

    return [p for p in preds if p is not None]
