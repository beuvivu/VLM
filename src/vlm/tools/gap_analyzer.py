"""Audit from draw 1, exclusion-aware targeted repair, and a rerunnable CLI.

An observed maximum is NOT proof that the history reaches the present. Use an
official latest probe or a documented --latest-ids boundary. Repairs are bounded
and checkpointed; unresolved draws stay unresolved, never fabricated.
"""
from __future__ import annotations
import argparse
import asyncio
import bisect
import gzip
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Any, Iterable
from vlm.database.schema import DrawRecord, VN, game_code, SQLRepository, DuckRepository, _merge, _merge_metadata
from vietlott_engine.core.exceptions import SourceError

SEED_FILES={"mega645":"power645.jsonl","power655":"power655.jsonl","lotto535":"power535.jsonl",
            "max3d":"max3d.jsonl","max3dplus":"max3d.jsonl","max3dpro":"max3d_pro.jsonl",
            "keno":"keno.jsonl.gz","bingo18":"bingo18.jsonl.gz","max4d":"max4d.jsonl"}
WEEKLY={"mega645":(2,4,6),"power655":(1,3,5),"max3d":(0,2,4),"max3dplus":(0,2,4),"max3dpro":(1,3,5)}


@dataclass
class GapReport:
    game_type: str
    rows: int = 0
    valid_unique: int = 0
    first_id: int | None = None
    last_id: int | None = None
    first_date: str | None = None
    last_date: str | None = None
    checked_through_id: int = 0
    boundary_verified: bool = False
    boundary_source: str = "observed maximum only"
    missing_ids: list[int] = field(default_factory=list)
    missing_ranges: list[list[int]] = field(default_factory=list)
    duplicate_ids: list[int] = field(default_factory=list)
    conflicting_ids: list[int] = field(default_factory=list)
    excluded_ids: list[int] = field(default_factory=list)
    excluded_present_ids: list[int] = field(default_factory=list)
    invalid_rows: list[dict[str,Any]] = field(default_factory=list)
    date_order_violations: list[int] = field(default_factory=list)
    leading_count: int = 0
    internal_count: int = 0
    trailing_count: int = 0
    source_counts: dict[str,int] = field(default_factory=dict)
    date_hints: dict[str,list[str]] = field(default_factory=dict)
    finance_coverage: dict[str,int] = field(default_factory=dict)
    scheduled_updates_due: list[str] = field(default_factory=list)
    complete_in_checked_range: bool = False
    complete_through_present: bool = False


def _ranges(ids: list[int]) -> list[list[int]]:
    ranges=[]
    for i in ids:
        if ranges and i==ranges[-1][1]+1:
            ranges[-1][1]=i
        else:
            ranges.append([i,i])
    return ranges


def analyze_game(game: str, rows: Iterable[dict[str,Any] | DrawRecord], *,
                 latest_id: int | None = None, excluded_ids: set[int] | None = None,
                 boundary_source: str = "caller supplied boundary", as_of: datetime | None = None) -> GapReport:
    g=game_code(game)
    if latest_id is not None and (type(latest_id) is not int or latest_id<1):
        raise ValueError("latest_id must be a positive integer")
    excluded=set(excluded_ids or ())
    if any(type(i) is not int or i<1 for i in excluded):
        raise ValueError("excluded ids must be positive integers")
    report=GapReport(g)
    seen: dict[int,tuple[datetime,tuple[Any,...],int | None,str]]={}
    raw_max=0
    sources: Counter[str]=Counter()
    finance: Counter[str]=Counter()
    duplicates,conflicts=set(),set()
    for line_no,row in enumerate(rows,1):
        report.rows+=1
        try:
            if isinstance(row,DrawRecord):
                record=DrawRecord.model_validate(row.model_dump())
            else:
                raw_id=row.get("draw_id",row.get("id"))
                if raw_id is not None:
                    raw_max=max(raw_max,int(str(raw_id).lstrip("#")))
                record=DrawRecord.from_legacy(g,row)
            if record.game_type.value!=g:
                raise ValueError("row belongs to a different product")
            raw_max=max(raw_max,record.draw_id)
        except (ValueError,TypeError,KeyError) as exc:
            report.invalid_rows.append({"line":line_no,"draw_id":row.get("draw_id",row.get("id")) if isinstance(row,dict) else row.draw_id,
                                        "error":type(exc).__name__})
            continue
        signature=(record.draw_date,record.winning_numbers,record.bonus_number,record.time_precision)
        if record.draw_id in seen:
            duplicates.add(record.draw_id)
            if seen[record.draw_id]!=signature:
                conflicts.add(record.draw_id)
        else:
            seen[record.draw_id]=signature
            sources[record.source]+=1
            for key in ("jackpot1_value","jackpot2_value","jackpot1_winners","jackpot2_winners","sub_prizes_json"):
                finance[key]+=int(getattr(record,key) is not None)
    if latest_id is not None and raw_max>latest_id:
        raise ValueError("latest boundary is older than observed data")
    ids=sorted(seen)
    report.valid_unique=len(ids)
    report.first_id,report.last_id=(ids[0],ids[-1]) if ids else (None,None)
    dates=[seen[i][0] for i in ids]
    if dates:
        report.first_date,report.last_date=min(dates).date().isoformat(),max(dates).date().isoformat()
    report.checked_through_id=latest_id if latest_id is not None else raw_max
    report.boundary_verified=latest_id is not None
    report.boundary_source=boundary_source if latest_id is not None else "observed maximum only"
    report.excluded_ids=sorted(i for i in excluded if i<=report.checked_through_id)
    report.excluded_present_ids=sorted(excluded & set(ids))
    report.missing_ids=[i for i in range(1,report.checked_through_id+1) if i not in seen and i not in excluded]
    report.missing_ranges=_ranges(report.missing_ids)
    report.duplicate_ids,report.conflicting_ids=sorted(duplicates),sorted(conflicts)
    report.source_counts,report.finance_coverage=dict(sources),dict(finance)
    for a,b in zip(ids,ids[1:]):
        if seen[b][0]<seen[a][0]:
            report.date_order_violations.append(b)
    report.leading_count=sum(i<(report.first_id or report.checked_through_id+1) for i in report.missing_ids)
    report.trailing_count=sum(i>(report.last_id or report.checked_through_id) for i in report.missing_ids)
    report.internal_count=len(report.missing_ids)-report.leading_count-report.trailing_count
    for missing in report.missing_ids:
        index=bisect.bisect_left(ids,missing)
        hints=[]
        # No synthetic dates for absent leading history.
        if index>0: hints.append(seen[ids[index-1]][0].date().isoformat())
        if 0<index<len(ids): hints.append(seen[ids[index]][0].date().isoformat())
        if hints:
            report.date_hints[str(missing)]=list(dict.fromkeys(hints))
    if as_of and report.last_date and g in (*WEEKLY,"lotto535"):
        now=(as_of if as_of.tzinfo else as_of.replace(tzinfo=VN)).astimezone(VN)
        last_day=date.fromisoformat(report.last_date)
        day=last_day if g=="lotto535" else last_day+timedelta(days=1)
        last_time=max(dates)
        last_day_records=[seen[i] for i in ids if seen[i][0].date()==last_day]
        while day<=now.date():
            times=[time(13),time(21)] if g=="lotto535" else [time(18)] if day.weekday() in WEEKLY[g] else []
            for t in times:
                planned=datetime.combine(day,t,VN)
                if day==last_day:
                    # A precise afternoon draw leaves the evening slot outstanding.
                    # With date-only history, two results cover both daily slots;
                    # one result cannot prove which slot has been obtained.
                    if any(r[3]=="day" for r in last_day_records):
                        if len(last_day_records)>=2:
                            continue
                    elif planned<=last_time:
                        continue
                if planned+timedelta(minutes=30)<=now:
                    report.scheduled_updates_due.append(planned.isoformat())
            day+=timedelta(days=1)
    report.complete_in_checked_range=not (
        report.missing_ids or report.invalid_rows or report.duplicate_ids or report.date_order_violations
    ) and report.valid_unique>0
    report.complete_through_present=report.complete_in_checked_range and report.boundary_verified and not report.scheduled_updates_due
    return report


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,prefix=path.name+".",suffix=".tmp",delete=False) as f:
            temporary=Path(f.name)
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def write_report(path: Path | str, reports: list[GapReport], *, as_of: datetime | None = None, extra: dict | None = None) -> None:
    _atomic_json(Path(path),{"schema_version":"1.0","as_of":(as_of or datetime.now(VN)).isoformat(),
                            "games":{r.game_type:asdict(r) for r in reports},**(extra or {})})


def read_rows(path: Path) -> Iterable[dict]:
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip():
                continue
            try:
                value=json.loads(line)
                if not isinstance(value,dict):
                    raise ValueError("record must be a JSON object")
                yield value
            except (ValueError,TypeError):
                yield {"_invalid_json_line":n}


def audit_rows(game: str, path: Path, repository: Any) -> Iterable[dict | DrawRecord]:
    """Validated union of immutable seeds and persisted repairs, without duplicates.

    Conflicting seed/repair rows are both retained so the audit exposes them.
    """
    repaired={r.draw_id:r for r in repository.load(game)}
    finance_path=path.parent/f"prizes_{game_code(game)}.jsonl"
    finances={}
    bad_finances=[]
    for financial in read_rows(finance_path) if finance_path.exists() else ():
        try:
            did=int(str(financial["draw_id"]).lstrip("#"))
            if did in finances:
                raise ValueError("duplicate finance row")
            finances[did]=financial
        except (ValueError,KeyError,TypeError):
            bad_finances.append({"draw_id":financial.get("draw_id"),"_invalid_finance":True})
    for raw in read_rows(path) if path.exists() else ():
        try:
            did=int(str(raw.get("draw_id",raw.get("id"))).lstrip("#"))
        except (ValueError,TypeError):
            yield raw
            continue
        financial=finances.pop(did,None)
        if financial:
            raw_date=str(raw.get("draw_date",raw.get("date","")))[:10]
            try:
                if game_code(financial.get("game",game))!=game_code(game) or str(financial["draw_date"])[:10]!=raw_date:
                    raise ValueError("finance join has wrong game/date")
                raw=raw | {key:_merge_metadata(raw.get(key),financial[key])
                           for key in ("winners","jackpot_pots","prizes") if key in financial}
            except (ValueError,KeyError,TypeError):
                bad_finances.append({"draw_id":did,"_finance_join_mismatch":True})
        patch=repaired.pop(did,None)
        if patch is None:
            yield raw
        else:
            try:
                combined=_merge(DrawRecord.from_legacy(game,raw),patch)
            except (ValueError,KeyError,TypeError):
                yield raw
                yield patch
            else:
                yield combined
    yield from repaired.values()
    yield from bad_finances


async def repair_missing(reports: list[GapReport], fetcher: Any, repository: Any, *,
                         max_draws: int = 100, concurrency: int = 2,
                         output_dir: Path | str = Path("data/local/repair"), per_draw_timeout_s: float = 180) -> dict:
    if type(max_draws) is not int or max_draws<0 or type(concurrency) is not int or concurrency<1:
        raise ValueError("repair limits must be nonnegative/positive integers")
    target=Path(output_dir)
    target.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for r in reports:
        for did in r.missing_ids:
            if did not in r.excluded_ids:
                jobs.append((r,did))
    # Prefer short internal/trailing gaps so a missing 83k-draw prefix cannot starve other products.
    jobs.sort(key=lambda job:(job[1]<(job[0].first_id or 1),job[0].game_type,job[1]))
    remaining=len(jobs)-min(max_draws,len(jobs))
    jobs=jobs[:max_draws]
    status={"requested":len(jobs),"remaining_unattempted":remaining,"repaired":[],"failed":[]}
    queue=asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    lock=asyncio.Lock()
    journal=target/"repaired_draws.jsonl"

    async def worker() -> None:
        while not queue.empty():
            r,did=queue.get_nowait()
            try:
                async with asyncio.timeout(per_draw_timeout_s):
                    row=None
                    last=None
                    for hint in r.date_hints.get(str(did),[None]):
                        try:
                            row=await fetcher.fetch(r.game_type,did,hint)
                            break
                        except (SourceError,ValueError,OSError,RuntimeError) as exc:
                            last=exc
                    if row is None:
                        raise ValueError("draw unavailable") from last
                    row=DrawRecord.model_validate(row.model_dump())
                    if row.game_type.value!=r.game_type or row.draw_id!=did:
                        raise ValueError("repair source returned the wrong draw")
                    repository.upsert([row])
                    async with lock:
                        with journal.open("a",encoding="utf-8") as f:
                            f.write(row.model_dump_json()+"\n")
                            f.flush()
                            os.fsync(f.fileno())
                        status["repaired"].append({"game_type":r.game_type,"draw_id":did})
                        _atomic_json(target/"repair_status.json",status)
            except Exception as exc:
                async with lock:
                    status["failed"].append({"game_type":r.game_type,"draw_id":did,"error":type(exc).__name__})
                    _atomic_json(target/"repair_status.json",status)
            finally:
                queue.task_done()
    await asyncio.gather(*(worker() for _ in range(min(concurrency,max(1,len(jobs))))))
    _atomic_json(target/"repair_status.json",status)
    return status


async def _run(args: argparse.Namespace) -> int:
    from vlm.crawler.async_scraper import AsyncScraper, ScraperConfig, FlareSolverrSolver
    from vlm.crawler.sources import OfficialSource, TargetedFetcher, default_sources
    as_of=datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(VN)
    latest=json.loads(Path(args.latest_ids).read_text()) if args.latest_ids else {}
    exclusions={}
    exclusion_path=args.seed_dir/"exclusions.json"
    if exclusion_path.exists():
        for e in json.loads(exclusion_path.read_text(encoding="utf-8")):
            exclusions.setdefault(e["product"],set()).add(int(e["draw_id"]))
    games=list(SEED_FILES) if args.game=="all" else [game_code(args.game)]
    reports,probe_errors=[],{}
    repo=SQLRepository(args.database) if not args.database.startswith("duckdb:") else DuckRepository(args.database.removeprefix("duckdb:"))
    def analyze(game: str) -> GapReport:
        return analyze_game(game,audit_rows(game,args.seed_dir/SEED_FILES[game],repo),latest_id=latest.get(game),
                            excluded_ids=exclusions.get("max3d" if game=="max3dplus" else game,set()),as_of=as_of,
                            boundary_source="official latest probe" if args.probe_latest and game not in probe_errors else "caller supplied boundary")
    solver_url=os.getenv("VLM_FLARESOLVERR_URL")
    solver=FlareSolverrSolver(solver_url) if solver_url else None
    try:
        async with AsyncScraper(ScraperConfig.from_env(),solver=solver) as client:
            official=OfficialSource(client)
            for game in games:
                if args.probe_latest and game!="max4d":
                    try:
                        latest[game]=(await official.latest(game)).draw_id
                    except Exception as exc:
                        probe_errors[game]=type(exc).__name__
                reports.append(analyze(game))
            write_report(args.output,reports,as_of=as_of,extra={"latest_probe_errors":probe_errors})
            if args.repair:
                result=await repair_missing(reports,TargetedFetcher(default_sources(client)),repo,max_draws=args.max_repair,
                                            concurrency=args.concurrency,output_dir=args.repair_dir)
                reports=[analyze(game) for game in games]
                write_report(args.output,reports,as_of=as_of,extra={"latest_probe_errors":probe_errors,"repair_status":result})
                print(json.dumps({k:v for k,v in result.items() if k not in ("repaired","failed")}|{
                    "repaired":len(result["repaired"]),"failed":len(result["failed"])},ensure_ascii=False))
    finally:
        repo.close()
    for r in reports:
        print(f"{r.game_type}: valid={r.valid_unique}, missing={len(r.missing_ids)}, excluded={len(r.excluded_ids)}, boundary_verified={r.boundary_verified}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir",type=Path,default=Path("data/seed"))
    parser.add_argument("--game",default="all",choices=["all",*SEED_FILES])
    parser.add_argument("--output",type=Path,default=Path("reports/vlm/missing_draws.json"))
    parser.add_argument("--latest-ids",help="JSON object of independently confirmed game -> latest id")
    parser.add_argument("--as-of",help="ISO datetime; naive values mean Vietnam time")
    parser.add_argument("--probe-latest",action="store_true")
    parser.add_argument("--repair",action="store_true")
    parser.add_argument("--max-repair",type=int,default=100)
    parser.add_argument("--concurrency",type=int,default=2)
    parser.add_argument("--database",default="sqlite:///data/local/vlm.sqlite")
    parser.add_argument("--repair-dir",type=Path,default=Path("data/local/repair"))
    args=parser.parse_args(argv)
    if args.database.startswith("sqlite:///") and args.database!="sqlite:///:memory:":
        Path(args.database.removeprefix("sqlite:///")).parent.mkdir(parents=True,exist_ok=True)
    return asyncio.run(_run(args))


if __name__=="__main__":
    raise SystemExit(main())
