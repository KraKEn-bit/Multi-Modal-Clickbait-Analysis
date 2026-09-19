"""
ERA5 download — storm months only, several CDS jobs in parallel.

Two things dominate wall-clock, and both parallelise:
  1. Copernicus queue time per request (~1-2 min even for a 2 MB file)
  2. the download itself from ECMWF's object store

cdsapi waits `sleep_max` seconds between connection retries. The default
(120 s) turned a flaky link into hours of sleeping, so we lower it.

Credentials: NEW WAY/.cdsapirc, then %USERPROFILE%\\.cdsapirc, then env.
Box: (N, W, S, E) = 30, 70, 5, 102. Years 1940-2024, 3-hourly.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD_READY = ROOT.parent / "Final_Cyclone_Pred_Results_P-3" / "datasets" / "bangladesh_nextstep_dataset_research_ready.csv"
RAW = ROOT / "datasets" / "era5_raw"
PROG = ROOT / "results" / "era5_download_progress.json"
PACE = ROOT / "results" / "era5_pace.json"
STATUS = ROOT / "ERA5_STATUS.txt"
WAIT_FLAG = ROOT / "WAITING_FOR_CDS.txt"
WAIT_LICENCE = ROOT / "WAITING_FOR_LICENCE.txt"
BOX_MARK = RAW / "BOX.txt"

YEAR_START = 1940
YEAR_END = 2024
# N, W, S, E. West is 70 so BoB storms that cross south India keep ERA5.
AREA = [30, 70, 5, 102]
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]

DEFAULT_WORKERS = 2  # CDS caps queued requests per dataset; more just gets rejected
RETRY_SLEEP = 15  # cdsapi sleep_max; default 120 wasted hours on a flaky link

_lock = threading.Lock()
_local = threading.local()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_progress(**kwargs) -> None:
    PROG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": _utc(), **kwargs}
    PROG.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_pace() -> dict:
    if PACE.exists():
        try:
            return json.loads(PACE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"too_large": 0, "marks": []}


def _save_pace(pace: dict) -> None:
    PACE.parent.mkdir(parents=True, exist_ok=True)
    PACE.write_text(json.dumps(pace, indent=2), encoding="utf-8")


def _parse_rc(path: Path) -> tuple[str, str] | None:
    if not path.exists():
        return None
    url, key = "", ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("url:"):
            url = line.split(":", 1)[1].strip()
        elif line.startswith("key:"):
            key = line.split(":", 1)[1].strip()
    if url and key and "<" not in key:
        return url, key
    return None


def load_cds_credentials() -> tuple[str, str] | None:
    env_url = os.environ.get("CDSAPI_URL", "").strip()
    env_key = os.environ.get("CDSAPI_KEY", "").strip()
    if env_url and env_key:
        return env_url, env_key
    for path in (ROOT / ".cdsapirc", Path.home() / ".cdsapirc"):
        parsed = _parse_rc(path)
        if parsed:
            return parsed
    return None


class LicenceNotAccepted(Exception):
    pass


class RequestTooLarge(Exception):
    pass


class QueueLimited(Exception):
    """CDS caps how many requests one user may have queued per dataset."""


def _classify(exc: Exception) -> str:
    text = str(exc).lower()
    if "licence" in text or "license" in text:
        return "licence"
    if "too large" in text or "cost limits" in text:
        return "too_large"
    if "temporarily limited" in text or "job has been rejected" in text:
        return "queue_limited"
    return "other"


def _raise_classified(exc: Exception) -> None:
    kind = _classify(exc)
    if kind == "licence":
        raise LicenceNotAccepted(str(exc)) from exc
    if kind == "too_large":
        raise RequestTooLarge(str(exc)) from exc
    if kind == "queue_limited":
        raise QueueLimited(str(exc)) from exc
    raise exc


def get_client(url: str, key: str):
    client = getattr(_local, "client", None)
    if client is None:
        import cdsapi

        client = cdsapi.Client(
            url=url,
            key=key,
            quiet=True,
            progress=False,
            sleep_max=RETRY_SLEEP,
            timeout=120,
        )
        _local.client = client
    return client


def _retrieve(client, dataset: str, request: dict, dest: Path) -> None:
    """Submit one request, waiting out the per-dataset queue cap if we hit it."""
    tmp = Path(f"{dest}.part")
    backoff = 60
    for attempt in range(12):
        if tmp.exists():
            tmp.unlink()
        try:
            client.retrieve(dataset, request, str(tmp))
            tmp.replace(dest)
            return
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            try:
                _raise_classified(exc)
            except QueueLimited:
                wait = min(backoff * (attempt + 1), 600)
                print(f"    queue full, waiting {wait}s ({dest.name})", flush=True)
                time.sleep(wait)
                continue
    raise QueueLimited(f"CDS queue stayed full for {dest.name}")


def _stitch(files: list[Path], dest: Path) -> None:
    if dest.exists() or not files or not all(p.exists() for p in files):
        return
    import xarray as xr

    ds = xr.open_mfdataset([str(p) for p in files], combine="by_coords")
    tmp = Path(f"{dest}.part")
    ds.to_netcdf(tmp)
    ds.close()
    tmp.replace(dest)


def _base_pl(year: int, month: str, days: list[str]) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": ["200", "500", "850"],
        "year": str(year),
        "month": [month],
        "day": days,
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _base_sl(year: int, month: str, days: list[str]) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": ["mean_sea_level_pressure", "sea_surface_temperature"],
        "year": str(year),
        "month": [month],
        "day": days,
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


DATASETS = {
    "pl": "reanalysis-era5-pressure-levels",
    "sl": "reanalysis-era5-single-levels",
}


def batch_path(kind: str, year: int, months: list[str]) -> Path:
    """One month keeps the old name so existing files still count as done."""
    if len(months) == 1:
        return RAW / f"era5_{kind}_{year}_{months[0]}.nc"
    return RAW / f"era5_{kind}_{year}_b{months[0]}_{months[-1]}.nc"


def month_covered(kind: str, year: int, month: str, days: list[str]) -> bool:
    if (RAW / f"era5_{kind}_{year}.nc").exists():
        return True
    if (RAW / f"era5_{kind}_{year}_{month}.nc").exists():
        return True
    for batch in RAW.glob(f"era5_{kind}_{year}_b*.nc"):
        stem = batch.stem.split("_b", 1)[1]
        lo, hi = stem.split("_")
        if lo <= month <= hi:
            return True
    return all((RAW / f"era5_{kind}_{year}_{month}_{d}.nc").exists() for d in days)


def fetch_months(client, kind: str, year: int, months: list[str],
                 cal: dict[str, list[str]]) -> str:
    """Ask for several storm-months in ONE request; halve on a size rejection.

    CDS charges queue latency per request, not per byte, so a wide request
    that pulls some unwanted days still beats many narrow ones. A size
    rejection comes back immediately, so an over-optimistic try is cheap.
    """
    if all(month_covered(kind, year, m, cal[m]) for m in months):
        return "skip"

    dest = batch_path(kind, year, months)
    if dest.exists():
        return "skip"

    req_fn = _base_pl if kind == "pl" else _base_sl
    days = sorted(set().union(*[set(cal[m]) for m in months]))
    request = req_fn(year, months[0], days)
    request["month"] = months

    label = months[0] if len(months) == 1 else f"{months[0]}-{months[-1]}"
    try:
        print(f"  {kind} {year}-{label} ({len(months)} mo x {len(days)} days)", flush=True)
        _retrieve(client, DATASETS[kind], request, dest)
        return "batch"
    except RequestTooLarge:
        if len(months) == 1:
            print(f"  {kind} {year}-{label} too large - per day", flush=True)
            day_files = [RAW / f"era5_{kind}_{year}_{months[0]}_{d}.nc" for d in days]
            for d, day_dest in zip(days, day_files):
                if day_dest.exists():
                    continue
                _retrieve(client, DATASETS[kind], req_fn(year, months[0], [d]), day_dest)
            _stitch(day_files, dest)
            return "days"
        mid = len(months) // 2
        print(f"  {kind} {year}-{label} too large - splitting", flush=True)
        fetch_months(client, kind, year, months[:mid], cal)
        fetch_months(client, kind, year, months[mid:], cal)
        return "split"


def load_storm_days(year_start: int, year_end: int) -> dict[int, dict[str, list[str]]]:
    import pandas as pd

    t = pd.read_csv(OLD_READY, usecols=["ISO_TIME"])
    t["ISO_TIME"] = pd.to_datetime(t["ISO_TIME"], errors="coerce")
    t = t.dropna(subset=["ISO_TIME"])
    t = t[(t["ISO_TIME"].dt.year >= year_start) & (t["ISO_TIME"].dt.year <= year_end)]
    out: dict[int, dict[str, list[str]]] = {}
    for ts in t["ISO_TIME"].dt.normalize().drop_duplicates():
        y, m, d = int(ts.year), f"{ts.month:02d}", f"{ts.day:02d}"
        out.setdefault(y, {}).setdefault(m, [])
        if d not in out[y][m]:
            out[y][m].append(d)
    for y in out:
        for m in out[y]:
            out[y][m] = sorted(out[y][m])
    return out


def _kind_complete(kind: str, year: int, month: str, days: list[str]) -> bool:
    return month_covered(kind, year, month, days)


def count_months(calendar: dict) -> tuple[int, int]:
    total = sum(len(ms) for ms in calendar.values())
    done = 0
    for y, months in calendar.items():
        for m, days in months.items():
            if _kind_complete("pl", y, m, days) and _kind_complete("sl", y, m, days):
                done += 1
    return done, total


def write_status(calendar: dict, current: str = "", last_error: str | None = None,
                 workers: int = DEFAULT_WORKERS) -> None:
    done, total = count_months(calendar)
    with _lock:
        pace = _load_pace()
        marks = [m for m in pace.get("marks", []) if isinstance(m, list) and len(m) == 2]
        if not marks or marks[-1][1] != done:
            marks.append([time.time(), done])
        pace["marks"] = marks[-30:]
        _save_pace(pace)
    n_big = int(pace.get("too_large", 0))

    recent = pace["marks"][-12:]
    pace_min = None
    if len(recent) >= 2 and recent[-1][1] > recent[0][1]:
        gained = recent[-1][1] - recent[0][1]
        pace_min = (recent[-1][0] - recent[0][0]) / gained / 60.0
    remain = max(total - done, 0)
    eta_h = remain * pace_min / 60.0 if pace_min else None

    if pace_min is None:
        path = "warming up (need a few finished months)"
    elif pace_min <= 1:
        path = "FAST - wide requests are paying off"
    elif pace_min <= 3:
        path = "OK - normal CDS queue"
    else:
        path = "SLOW - CDS queue latency is high"

    lines = [
        f"updated: {_utc()}",
        f"months: {done}/{total} complete (pl+sl storm-months)",
        f"current: {current or '-'}",
        f"workers: {workers} parallel CDS jobs, retry wait {RETRY_SLEEP}s",
        f"box: N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]}  years={YEAR_START}-{YEAR_END}",
        f"CDS: wide multi-month requests, too-large fallbacks={n_big}",
        f"pace: {pace_min:.2f} min/month" if pace_min else "pace: n/a yet",
        f"ETA: {eta_h:.1f} h remaining" if eta_h is not None else "ETA: n/a yet",
        f"path: {path}",
        f"last_error: {last_error or 'none'}",
        "Only this file is trustworthy. Do not guess from timestamps.",
    ]
    STATUS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _note_month_done(too_large: bool) -> None:
    if not too_large:
        return
    with _lock:
        pace = _load_pace()
        pace["too_large"] = int(pace.get("too_large", 0)) + 1
        _save_pace(pace)


def download_year_kind(url: str, key: str, kind: str, year: int,
                       cal: dict[str, list[str]]) -> str:
    """All still-missing storm-months of one year, in as few requests as CDS allows."""
    missing = [m for m in sorted(cal) if not month_covered(kind, year, m, cal[m])]
    if not missing:
        return "skip"
    client = get_client(url, key)
    return fetch_months(client, kind, year, missing, cal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=YEAR_START)
    parser.add_argument("--end", type=int, default=YEAR_END)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    box_s = ",".join(str(x) for x in AREA)
    if BOX_MARK.exists() and BOX_MARK.read_text(encoding="utf-8").strip() != box_s:
        raise SystemExit(
            f"ERA5 box changed (disk={BOX_MARK.read_text(encoding='utf-8').strip()}, "
            f"script={box_s}). Delete datasets/era5_raw to re-download."
        )

    calendar = load_storm_days(args.start, args.end)
    write_status(calendar, current="starting", workers=args.workers)

    tasks = []
    for year in sorted(calendar):
        for kind in ("pl", "sl"):
            if any(not month_covered(kind, year, m, d) for m, d in calendar[year].items()):
                tasks.append((kind, year))

    if not tasks:
        WAIT_FLAG.unlink(missing_ok=True)
        write_progress(status="complete")
        write_status(calendar, current="complete", workers=args.workers)
        print("ERA5 download complete (everything already on disk).")
        return

    try:
        import cdsapi  # noqa: F401
    except ImportError:
        write_progress(status="waiting", last_error="cdsapi not installed")
        raise SystemExit(3)

    creds = load_cds_credentials()
    if not creds:
        WAIT_FLAG.write_text(
            "ERA5 download is paused until a Copernicus CDS key exists.\n\n"
            "Put this file at NEW WAY\\.cdsapirc or %USERPROFILE%\\.cdsapirc:\n\n"
            "url: https://cds.climate.copernicus.eu/api\n"
            "key: YOUR_TOKEN\n",
            encoding="utf-8",
        )
        write_progress(status="waiting_for_cds")
        print("No CDS credentials. Wrote WAITING_FOR_CDS.txt")
        raise SystemExit(3)

    url, key = creds
    WAIT_FLAG.unlink(missing_ok=True)
    if not BOX_MARK.exists():
        BOX_MARK.write_text(box_s, encoding="utf-8")

    print(f"{len(tasks)} year-jobs pending, {args.workers} workers", flush=True)
    failure: Exception | None = None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_year_kind, url, key, kind, year, calendar[year]): (kind, year)
            for kind, year in tasks
        }
        for future in as_completed(futures):
            kind, year = futures[future]
            try:
                outcome = future.result()
            except LicenceNotAccepted as exc:
                WAIT_LICENCE.write_text(
                    "Accept ERA5 licences, then leave the runner going.\n\n"
                    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download\n"
                    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download\n",
                    encoding="utf-8",
                )
                write_progress(status="waiting_for_licence", last_error=str(exc))
                write_status(calendar, current="licence blocked",
                             last_error=str(exc)[:200], workers=args.workers)
                print("ERA5 licences not accepted. See WAITING_FOR_LICENCE.txt")
                for pending in futures:
                    pending.cancel()
                raise SystemExit(3) from exc
            except Exception as exc:  # noqa: BLE001 - runner retries the whole pass
                failure = exc
                print(f"ERROR {kind} {year}: {exc}", flush=True)
                write_status(calendar, current=f"{year} {kind}",
                             last_error=str(exc)[:200], workers=args.workers)
                continue

            if outcome != "skip":
                _note_month_done(too_large=outcome == "days")
            write_progress(status="running", current_year=year)
            write_status(calendar, current=f"{year} {kind}", workers=args.workers)

    if failure is not None:
        write_progress(status="error", last_error=str(failure))
        raise SystemExit(2)

    write_progress(status="complete")
    write_status(calendar, current="complete", workers=args.workers)
    print("ERA5 download complete.")


if __name__ == "__main__":
    main()
