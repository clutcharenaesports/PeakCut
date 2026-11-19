"""Tests for Meta Muse bridge, Smart Offline Scout, Facecam manual overrides, and Safe Auto-Post."""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

# Setup clean temporary data directory for this test
TEST_DATA = Path(tempfile.mkdtemp(prefix="pc-muse-test-"))
os.environ["PEAKCUT_DATA"] = str(TEST_DATA)
os.environ["PEAKCUT_ACCOUNTS"] = "0"
os.environ["DATABASE_URL"] = ""

import sys
sys.path.insert(0, ".")

from studio import accounts, ai, auth, brain, config, creators, ctx, db, jobs, live, posting  # noqa: E402

db.init()
user = auth.local_user()
ctx.set_uid(user.id)

fails = []


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}"))
    if not ok:
        fails.append(name)


def run_tests():
    print("\n--- 1. Fresh-install default provider is muse ---")
    check("fresh install default provider is muse", config.AI_PROVIDER, "muse")
    check("ai.ready is True for muse with no key", ai.ready(), True)
    cur = ai.current()
    check("current provider id is muse", cur["provider"], "muse")
    check("muse provider has bridge type", ai.PROVIDERS[cur["provider"]][2], "bridge")

    print("\n--- 2. Muse Bridge Request & Verdict Flow ---")
    pending_dir, done_dir = ai.bridge_dirs()
    check("pending dir exists", pending_dir.is_dir(), True)
    check("done dir exists", done_dir.is_dir(), True)

    req_id = "test_req_001"
    returned_id = ai.write_bridge_request("live_spike", {
        "request_id": req_id,
        "creator": "TestStreamer", "game": "GTA 6", "title": "Epic Stream",
        "window": {"start": 10.0, "end": 40.0},
        "words": [{"w": "bhai", "s": 12.0, "e": 12.5}, {"w": "ye", "s": 12.6, "e": 12.9}, {"w": "dekh", "s": 13.0, "e": 13.4}],
        "loudness": {"peak_z": 3.2},
        "chat": ["kya hua", "clip it"],
        "decision_needed": "Judge clip"
    })
    check("returned id matches", returned_id, req_id)
    req_file = pending_dir / f"{req_id}.json"
    check("pending request file written", req_file.is_file(), True)
    stat = ai.bridge_status()
    check("bridge status shows 1 pending", stat["pending_count"], 1)

    # Check no verdict yet
    check("check_bridge_verdict is None initially", ai.check_bridge_verdict(req_id), None)

    # Simulate Muse writing verdict to done/
    verdict_data = {
        "request_id": req_id,
        "accept": True,
        "start": 11.5,
        "end": 38.0,
        "score": 88,
        "hook": "BHAI YE DEKH",
        "title": "Unreal GTA 6 Moment",
        "description": "Streamer reaction was insane. #GTA6",
        "tags": ["GTA6", "Shorts"],
        "reason": "Top streamer reaction"
    }
    done_file = done_dir / f"{req_id}.json"
    done_file.write_text(json.dumps(verdict_data), encoding="utf-8")

    # Ingest verdict
    v_read = ai.check_bridge_verdict(req_id)
    check("check_bridge_verdict sees verdict", bool(v_read), True)
    check("verdict score is 88", v_read.get("score"), 88)

    v_consumed = ai.consume_bridge_verdict(req_id)
    check("consume_bridge_verdict returns verdict", bool(v_consumed), True)
    check("pending file deleted after consume", req_file.is_file(), False)
    check("done file deleted after consume", done_file.is_file(), False)
    check("bridge status pending is 0 now", ai.bridge_status()["pending_count"], 0)

    print("\n--- 3. Smart Offline Scout & Dynamic Spoken Hooks ---")
    words = [
        {"w": "hello", "s": 0.0, "e": 0.5},
        {"w": "bhai", "s": 20.0, "e": 20.4},
        {"w": "ye", "s": 20.5, "e": 20.8},
        {"w": "kya", "s": 20.9, "e": 21.2},
        {"w": "ho", "s": 21.3, "e": 21.6},
        {"w": "gaya", "s": 21.7, "e": 22.0},
        {"w": "unbelievable", "s": 22.5, "e": 23.2},
    ]
    loudness = [(t, 0.5 if t < 18 or t > 25 else 3.8) for t in range(40)]

    cands, _ = brain.scout_live_moment(
        words, loudness[10:35], 10.0, 35.0,
        title="Gaming stream", uploader="ProPlayer",
        min_len=10, max_len=30, game="GTA 6", chat=["omg bhai", "clip this"]
    )
    check("scout produced candidates", len(cands) > 0, True)
    if cands:
        top = cands[0]
        check("scout score is reasonable (>60)", top["score"] >= 60, True)
        check("hook is extracted from spoken words (not 'WAIT FOR IT')", top["hook"] != "WAIT FOR IT", True)
        check("hook is upper-case", top["hook"].isupper(), True)
        check("hook has <= 5 words", len(top["hook"].split()) <= 5, True)
        check("hashtags include game or shorts", any("Shorts" in t or "GTA" in t for t in top.get("hashtags", [])), True)

    print("\n--- 4. Facecam Manual Overrides & Source Transparency ---")
    c_manual = creators.create({"name": "ManualCamCreator", "facecam": [0.65, 0.05, 0.30, 0.25]})
    check("creator stores manual facecam", c_manual.get("facecam"), [0.65, 0.05, 0.30, 0.25])

    job_with_creator = {"creator": c_manual["id"], "width": 1920, "height": 1080}
    box = jobs._facecam(job_with_creator)
    check("manual facecam returns converted pixel box", bool(box), True)
    check("manual facecam X coordinate correct", box[0], int(round(0.65 * 1920)))
    check("facecam source tagged as manual", job_with_creator.get("facecam_source"), "manual")

    job_fallback = {"source_path": str(TEST_DATA / "nonexistent.mp4")}
    box_none = jobs._facecam(job_fallback)
    check("missing file gives None box", box_none, None)
    check("facecam source tagged as fallback", job_fallback.get("facecam_source"), "fallback")

    print("\n--- 5. Safe Auto-Post Gating ---")
    # Fake accounts: 1 YouTube, 1 Instagram
    acc_yt = {"id": "acc_yt_01", "platform": "youtube", "name": "GamerChannel", "added": time.time()}
    acc_ig = {"id": "acc_ig_01", "platform": "instagram", "name": "GamerInsta", "added": time.time()}
    accounts._put(acc_yt)
    accounts._put(acc_ig)

    creator_auto = creators.create({"name": "AutoCreator", "autopost": False, "post_min_score": 75, "accounts": [acc_yt["id"], acc_ig["id"]]})

    # High scoring clip but job.options auto_post is False -> should not post
    job_no_auto = {"id": "j_001", "creator": creator_auto["id"], "options": {"auto_post": False}}
    clip_high = {"id": "clip01", "score": 85, "file": "clip01.mp4", "status": "done"}
    posting.on_clip_ready(job_no_auto, clip_high)
    check("auto_post=False produces no post queue entry", "posts" not in clip_high or not clip_high["posts"], True)

    # Low scoring clip with auto_post=True -> below post_min_score -> should not post
    job_auto = {"id": "j_002", "creator": creator_auto["id"], "options": {"auto_post": True}}
    clip_low = {"id": "clip02", "score": 60, "file": "clip02.mp4", "status": "done"}
    posting.on_clip_ready(job_auto, clip_low)
    check("score below min_score produces no post queue entry", "posts" not in clip_low or not clip_low["posts"], True)

    # High scoring clip with auto_post=True and no creator accounts selected -> default ONLY to connected YouTube
    creator_no_accs = creators.create({"name": "NoAccCreator", "autopost": False, "post_min_score": 70, "accounts": []})
    job_auto_default = jobs.create(
        {"type": "file", "name": "stream.mp4"},
        {"auto_post": True},
        creator=creator_no_accs["id"]
    )
    clip_win = {"id": "clip01", "score": 82, "file": "clip01.mp4", "status": "done"}
    job_auto_default["clips"].append(clip_win)
    jobs.save(job_auto_default)

    posting.on_clip_ready(job_auto_default, clip_win)
    check("default auto_post enqueued a post", bool(clip_win.get("posts")), True)
    enqueued_targets = [p.get("account") for p in (posting._mine() or []) if p.get("job") == job_auto_default["id"]]
    check("default auto_post queues ONLY to YouTube", enqueued_targets, [acc_yt["id"]])
    check("Instagram was NOT silently posted to", acc_ig["id"] not in enqueued_targets, True)

    print("\n--- Summary ---")
    if fails:
        print(f"FAILED {len(fails)} checks:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_tests())
    finally:
        shutil.rmtree(TEST_DATA, ignore_errors=True)
