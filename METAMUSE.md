# Meta Muse Agent Bridge Specification for PeakCut

This document describes how Meta Muse (or any autonomous local AI agent) interacts with PeakCut without requiring cloud LLM API keys.

---

## 1. Overview & Architecture

PeakCut operates with three judging modes in strict priority order:

1. **DEFAULT: "Muse" Bridge (Agent on Local Machine)**  
   No API key required. When PeakCut detects a candidate moment (live stream spike or long video segment), it writes a request JSON to the filesystem and parks the candidate in `awaiting_muse` state. Muse reads the pending request, analyzes the transcript/signals, and writes a verdict JSON.
2. **OPTIONAL: API-Key Providers (Gemini / OpenAI / Grok / Claude)**  
   Enabled when configured in the dashboard (`Settings > AI`).
3. **FALLBACK: Smart Offline Scout**  
   Runs automatically if Muse does not answer within **8 minutes** or if no provider is configured. The smart offline scout uses multi-signal analysis (loudness z-scores, speech rate bursts, silence contrast, Hinglish/gaming keywords) and derives dynamic hooks directly from spoken words.

---

## 2. Directory Layout

The file-queue bridge uses the PeakCut data directory:

```
data/
└── ai_bridge/
    ├── pending/       <-- PeakCut writes requests here
    │   └── req_abc123.json
    └── done/          <-- Muse writes verdicts here
        └── req_abc123.json
```

---

## 3. Bridge JSON Schemas

### 3.1 Pending Request (`data/ai_bridge/pending/<request_id>.json`)

Written by PeakCut when evaluating a moment:

```json
{
  "request_id": "req_20261119_103000_a1b2c3",
  "kind": "live_spike",
  "created_at": 1794987000.0,
  "creator": "GamerFleet",
  "game": "GTA 6",
  "title": "GTA 6 RP Stream Day 5",
  "window": {
    "start": 419.27,
    "end": 469.87
  },
  "words": [
    {"w": "bhai", "s": 420.1, "e": 420.4},
    {"w": "ye", "s": 420.5, "e": 420.7},
    {"w": "dekh", "s": 420.8, "e": 421.1},
    {"w": "kya", "s": 421.2, "e": 421.4},
    {"w": "hua", "s": 421.5, "e": 421.8}
  ],
  "loudness": {
    "peak_z": 3.4,
    "mean_z": 1.8
  },
  "chat": [
    "LMAOOO",
    "bhai kya kiya",
    "clip it",
    "clip that"
  ],
  "decision_needed": "Decide whether this window contains a compelling clip highlight, score it 0-100, and generate hook, title, and tags."
}
```

### 3.2 Verdict Response (`data/ai_bridge/done/<request_id>.json`)

Written by Meta Muse when analysis is complete:

```json
{
  "request_id": "req_20261119_103000_a1b2c3",
  "accept": true,
  "start": 420.0,
  "end": 468.5,
  "score": 88,
  "hook": "BHAI YE KYA HUA",
  "title": "Insane Car Chase in Vice City!",
  "description": "Watch what happens when the heist escape goes completely wrong. #GTA6 #Gaming",
  "tags": ["GTA6", "Gaming", "Shorts", "ViceCity"],
  "reason": "Sudden intense car crash followed by hilarious streamer reaction."
}
```

If rejected:
```json
{
  "request_id": "req_20261119_103000_a1b2c3",
  "accept": false,
  "reason": "Just standard gameplay commentary without a distinct comedic or action peak."
}
```

---

## 4. Timeout & Fallback Contract

- **Poll Interval**: PeakCut checks `data/ai_bridge/done/` every few seconds.
- **8-Minute Deadline**: If no verdict file appears in `data/ai_bridge/done/<request_id>.json` within 8 minutes (480s) from `created_at`, PeakCut consumes the candidate using the **smart offline scout**.
- Once consumed, PeakCut removes both the pending and done files to keep the directory clean.

---

## 5. Safe Auto-Post Doctrine

When automated posting is enabled (`--auto-post` in CLI or toggle in UI):

1. **Default Destination**: If no explicit creator accounts are checked, PeakCut posts **ONLY to connected YouTube accounts**. It will NEVER post silently to Instagram or TikTok.
2. **Score Threshold**: Clips must meet or exceed the creator's `post_min_score` (default 75).
3. **Queue Discipline**: Posts enter the queue respecting `post_gap_min` (spacing between uploads) and `daily_cap`.
4. **Google Policy Notice**: In accordance with Google API policies for unverified developer apps, YouTube uploads default to `private` status.

---

## 6. Command Line Interface (CLI)

Meta Muse can operate PeakCut directly from the terminal or background scripts:

### Cut a Video:
```bash
python -m studio.cli cut "https://www.youtube.com/watch?v=..." --creator "GamerFleet" --auto-post --layout gaming_split
```

### Check Status:
```bash
python -m studio.cli status
python -m studio.cli status 20261119-103000-a1b2c3d4e5
```

### Inspect Connected Publishing Accounts:
```bash
python -m studio.cli accounts
```

### Run PeakCut Server:
```bash
python -m studio.cli serve --port 8765
```
