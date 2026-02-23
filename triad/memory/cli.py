from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .memory import Memory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crtx.memory", description="CRTX Memory Engine CLI")
    parser.add_argument("--memory-dir", type=Path, default=None, help="Memory storage directory")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Show memory status")

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest existing data")
    ingest_p.add_argument("--clawbucks-dir", type=Path, default=Path.cwd(), help="Clawbucks directory")

    # learn
    sub.add_parser("learn", help="Run pattern extraction")

    # patterns
    patterns_p = sub.add_parser("patterns", help="Show discovered patterns")
    patterns_p.add_argument("--niche", type=str, default=None, help="Filter by niche")

    # taxonomy
    taxonomy_p = sub.add_parser("taxonomy", help="Show taxonomy status")
    taxonomy_p.add_argument("--reset", action="store_true", help="Reset all rules")

    # decisions
    decisions_p = sub.add_parser("decisions", help="Show recent decisions")
    decisions_p.add_argument("--last", type=int, default=20, help="Number of recent decisions")
    decisions_p.add_argument("--niche", type=str, default=None, help="Filter by niche")
    decisions_p.add_argument("--type", type=str, default=None, help="Filter by content type")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    mem = Memory(memory_dir=args.memory_dir)

    if args.command == "status":
        _cmd_status(mem, args)
    elif args.command == "ingest":
        _cmd_ingest(mem, args)
    elif args.command == "learn":
        _cmd_learn(mem, args)
    elif args.command == "patterns":
        _cmd_patterns(mem, args)
    elif args.command == "taxonomy":
        _cmd_taxonomy(mem, args)
    elif args.command == "decisions":
        _cmd_decisions(mem, args)


def _cmd_status(mem: Memory, args: argparse.Namespace) -> None:
    state = mem.state
    if args.json_output:
        from dataclasses import asdict
        print(json.dumps(asdict(state), indent=2))
        return

    print("CRTX Memory Status")
    print(f"  Version:          {state.version}")
    print(f"  Created:          {state.created_at}")
    print(f"  Last updated:     {state.last_updated}")
    print(f"  Total decisions:  {state.total_decisions}")
    print(f"  Auto-ships:       {state.total_auto_ships}")
    print(f"  Human overrides:  {state.total_human_overrides}")
    print(f"  Patterns:         {len(state.patterns)}")
    print(f"  Taxonomy rules:   {len(state.taxonomy_rules)}")


def _cmd_ingest(mem: Memory, args: argparse.Namespace) -> None:
    clawbucks_dir = args.clawbucks_dir
    print(f"Ingesting from: {clawbucks_dir}")
    count = mem.ingest_history(clawbucks_dir)
    if args.json_output:
        print(json.dumps({"ingested": count}))
    else:
        print(f"Ingested {count} decisions.")


def _cmd_learn(mem: Memory, args: argparse.Namespace) -> None:
    print("Running pattern extraction...")
    summary = mem.learn()
    if args.json_output:
        print(json.dumps(summary, indent=2))
        return

    print(f"  Total patterns:  {summary['total_patterns']}")
    print(f"  Active:          {summary['active']}")
    print(f"  Weak:            {summary['weak']}")
    print(f"  Retired:         {summary['retired']}")
    if summary["pattern_types"]:
        print(f"  Types:           {', '.join(summary['pattern_types'])}")


def _cmd_patterns(mem: Memory, args: argparse.Namespace) -> None:
    patterns = mem.get_patterns(niche_id=args.niche)
    if args.json_output:
        from dataclasses import asdict
        print(json.dumps([asdict(p) for p in patterns], indent=2))
        return

    if not patterns:
        print("No active patterns found.")
        return

    print(f"Active Patterns ({len(patterns)}):")
    for p in patterns:
        print(f"\n  [{p.pattern_type}] {p.description}")
        print(f"    Confidence: {p.confidence:.0%}  Samples: {p.sample_size}  Status: {p.status}")


def _cmd_taxonomy(mem: Memory, args: argparse.Namespace) -> None:
    if args.reset:
        mem.taxonomy.reset()
        print("Taxonomy reset: all streaks cleared, all actions set to 'flag'.")
        return

    report = mem.taxonomy.get_status_report()
    if args.json_output:
        print(json.dumps(report, indent=2))
        return

    print("Taxonomy Status")
    print(f"  Total rules:      {report['total_rules']}")
    print(f"  Auto-ship:        {report['auto_ship']}")
    print(f"  Flag:             {report['flag']}")
    print(f"  Pause:            {report['pause']}")
    print(f"  Today's auto-ships: {report['daily_auto_ships']}")

    if report["rules"]:
        print("\nRules:")
        for r in report["rules"]:
            niche = r["niche_id"] or "*"
            pillar = r["pillar_id"] or "*"
            print(f"  {r['content_type']:15s} {niche:20s} {pillar:25s} -> {r['action']:10s} (streak {r['streak']}/{r['required']}, cooldown {r['cooldown']})")


def _cmd_decisions(mem: Memory, args: argparse.Namespace) -> None:
    decisions = mem.log.query(
        content_type=args.type,
        niche_id=args.niche,
        limit=args.last,
    )
    if args.json_output:
        from dataclasses import asdict
        print(json.dumps([asdict(d) for d in decisions], indent=2))
        return

    if not decisions:
        print("No decisions found.")
        return

    print(f"Recent Decisions ({len(decisions)}):")
    for d in decisions:
        preview = d.content_preview[:80] + "..." if len(d.content_preview) > 80 else d.content_preview
        print(f"\n  [{d.decision:12s}] {d.content_type:10s} | {d.niche_id}")
        print(f"    {preview}")
        print(f"    Source: {d.decision_source}  Time: {d.timestamp[:19]}")
        if d.taxonomy_action:
            print(f"    Taxonomy: {d.taxonomy_action}")


if __name__ == "__main__":
    main()
