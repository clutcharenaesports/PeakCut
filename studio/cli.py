"""Command-line interface for PeakCut and AI agents (e.g. Meta Muse).

Usage:
    python -m studio.cli cut <url_or_path> [--creator <name_or_id>] [--auto-post] [--layout <layout>] [--style <style>]
    python -m studio.cli status [<job_id>]
    python -m studio.cli accounts
    python -m studio.cli serve [--host <host>] [--port <port>]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import accounts, auth, config, creators, ctx, db, jobs


def _setup_context():
    db.init()
    user = auth.local_user()
    ctx.set_uid(user.id)
    return user


def cmd_cut(args):
    _setup_context()
    target = args.target.strip()
    p = Path(target)
    if p.is_file():
        source = {"type": "file", "path": str(p.resolve()), "name": p.name}
    else:
        source = {"type": "url", "url": target, "name": target}

    creator_id = None
    if args.creator:
        all_c = creators.all_creators()
        match = next((c for c in all_c if c["id"] == args.creator or c["name"].lower() == args.creator.lower()), None)
        if match:
            creator_id = match["id"]
        else:
            creator_id = args.creator

    opts = {
        "auto_post": bool(args.auto_post),
    }
    if args.layout:
        opts["layout"] = args.layout
    if args.style:
        opts["style"] = args.style
    if args.quality:
        opts["quality"] = int(args.quality)

    job = jobs.create(source, options=opts, creator=creator_id)
    print(f"Created job {job['id']}")
    print(f"Status: {job['status']} ({job.get('stage', '')})")
    print(f"Auto-post: {job['options'].get('auto_post', False)}")
    print(f"Storage dir: {job['dir']}")
    return 0


def cmd_status(args):
    _setup_context()
    if args.job_id:
        j = jobs.get(args.job_id)
        if not j:
            print(f"Job {args.job_id} not found.", file=sys.stderr)
            return 1
        print(f"Job: {j['id']}")
        print(f"Status: {j['status']} | Progress: {int(j.get('progress', 0) * 100)}% | Stage: {j.get('stage')}")
        print(f"Source: {j.get('title') or j.get('source', {}).get('name')}")
        print(f"Clips ({len(j.get('clips', []))}):")
        for c in j.get("clips", []):
            fc = c.get("facecam_source", "none")
            posts = list(c.get("posts", {}).keys())
            post_str = f" | Posted to: {','.join(posts)}" if posts else ""
            print(f"  - [{c['id']}] {c.get('hook', 'UNTITLED')} (score {c.get('score', 0)}) [{c.get('status')}] [facecam:{fc}]{post_str}")
            print(f"      Title: {c.get('title', '')}")
    else:
        all_j = jobs.all_jobs()
        if not all_j:
            print("No jobs found.")
            return 0
        print(f"{'JOB ID':<32} {'STATUS':<10} {'CLIPS':<6} {'STAGE'}")
        print("-" * 75)
        for j in sorted(all_j, key=lambda x: x.get("created", 0), reverse=True)[:20]:
            print(f"{j['id']:<32} {j.get('status', 'unknown'):<10} {len(j.get('clips', [])):<6} {j.get('stage', '')[:25]}")
    return 0


def cmd_accounts(args):
    _setup_context()
    accs = accounts.list_accounts()
    if not accs:
        print("No publishing accounts connected.")
        print("Connect YouTube, Instagram, or TikTok in PeakCut dashboard (Settings > Connected apps).")
        return 0
    print(f"{'PLATFORM':<12} {'LABEL':<20} {'NAME':<20} {'ID'}")
    print("-" * 65)
    for a in accs:
        print(f"{a.get('platform', ''):<12} {a.get('label', ''):<20} {a.get('name', ''):<20} {a.get('id', '')}")
    return 0


def cmd_serve(args):
    import uvicorn
    host = args.host or config.HOST
    port = int(args.port or config.PORT)
    print(f"Starting PeakCut server on http://{host}:{port} ...")
    uvicorn.run("studio.server:app", host=host, port=port, reload=False)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="peakcut", description="PeakCut CLI for automation and AI agents.")
    subparsers = parser.add_subparsers(dest="command")

    p_cut = subparsers.add_parser("cut", help="Cut a video into vertical clips")
    p_cut.add_argument("target", help="Video link (YouTube/Twitch/Kick) or local video file path")
    p_cut.add_argument("--creator", help="Creator ID or name to file under", default=None)
    p_cut.add_argument("--auto-post", action="store_true", help="Queue clips to connected YouTube accounts automatically")
    p_cut.add_argument("--layout", help="Layout style (e.g. gaming_split, blur, zoom)")
    p_cut.add_argument("--style", help="Caption style (e.g. bounce, clean, beast)")
    p_cut.add_argument("--quality", help="Target resolution: 720 or 1080", default=None)

    p_status = subparsers.add_parser("status", help="Check status of jobs")
    p_status.add_argument("job_id", nargs="?", help="Optional specific job ID to inspect")

    subparsers.add_parser("accounts", help="List connected publishing accounts")

    p_serve = subparsers.add_parser("serve", help="Run the PeakCut web server")
    p_serve.add_argument("--host", help="Host interface to bind", default=None)
    p_serve.add_argument("--port", type=int, help="Port to bind", default=None)

    parsed = parser.parse_args(argv)
    if not parsed.command:
        parser.print_help()
        return 0

    dispatch = {
        "cut": cmd_cut,
        "status": cmd_status,
        "accounts": cmd_accounts,
        "serve": cmd_serve,
    }
    return dispatch[parsed.command](parsed)


if __name__ == "__main__":
    sys.exit(main())
