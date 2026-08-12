# test_repo — AI Incident Response Framework (Demo)

A self-contained sandbox for testing an **AI-driven incident triage and auto-repair pipeline**: a fault-injecting demo app exposes Prometheus metrics, Alertmanager fires on error spikes, and an [n8n](https://n8n.io) workflow picks up the alert, pulls the relevant source file from this repo, asks an LLM to diagnose and patch it, then opens a pull request (or emails a human if the fix isn't safe to apply automatically).

## How it works

```
demo app (app.py)
   │  exposes /metrics on :8000
   ▼
Prometheus  ──scrapes──▶  evaluates alertrules.yml
   │
   ▼ (HighErrorRate fires)
Alertmanager  ──webhook──▶  n8n workflow
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
            query Prometheus          fetch source file
             for current metrics         from this repo
                    └───────────┬────────────┘
                                 ▼
                     AI incident analysis (LLM)
                                 │
                                 ▼
                        patch safety check
                       ┌─────────┴─────────┐
                    safe                 unsafe
                       ▼                     ▼
          branch → commit → PR      email for manual review
                       ▼
                email: PR opened for review
```

Nothing merges automatically — every AI-generated fix lands as a pull request (or a "needs manual review" email) for a human to approve.

## Repo contents

| File | Purpose |
|---|---|
| `app.py` | Demo Flask-free Python service. Simulates normal work and randomly throws errors (~40% of cycles) without crashing, so there's always something for the pipeline to react to. Exposes Prometheus metrics on port `8000`. |
| `requirements.txt` | Python dependency (`prometheus-client`). |
| `prometheus.yml` | Prometheus config: scrapes the demo app every 5s, evaluates `alertrules.yml`, and forwards alerts to Alertmanager. |
| `alertrules.yml` | Defines the `HighErrorRate` alert, firing when `app_error_rate == 1` for 15s. |
| `alertmanager.yml` | Routes firing alerts to an email address and to the n8n webhook (`/webhook/incident-alert`). |
| `docker-compose.yml` | Runs n8n locally (port `5678`) with persistent volume storage. |
| `incident_resolution.json` | n8n workflow export — original version, sourced metrics from Datadog. |
| `incident_resolution_1_repaired.json` | n8n workflow export — current version: pulls metrics from Prometheus, dynamically resolves which repo file to patch based on the alert's `bug` label, and adds branch/commit/PR/email steps. |
| `n8n support files/test_repo/` | Supporting files referenced by the n8n workflow imports. |

## Quickstart

**1. Start the demo app**
```bash
pip install -r requirements.txt
python app.py
```
Metrics will be live at `http://localhost:8000/metrics`.

**2. Start Prometheus and Alertmanager** (pointed at `prometheus.yml` and `alertmanager.yml` respectively — not bundled in `docker-compose.yml`, which currently only runs n8n). Update `alertmanager.yml` with your own SMTP app password and recipient before starting it.

**3. Start n8n**
```bash
docker compose up -d
```
Import `incident_resolution_1_repaired.json` into n8n (Workflows → Import from File).

**4. Configure credentials in n8n**
- **GitHub API** credential — attach to `Get Repo File` and every other GitHub HTTP node.
- **Custom Auth** credential for Gemini — JSON body `{"qs": {"key": "<your-gemini-api-key>"}}`, attached to `AI Incident Analysis`.
- **SMTP** credential — attached to both email notification nodes.

**5. Trigger an incident**
Let the demo app run long enough for `HighErrorRate` to fire, or POST a synthetic alert straight to `http://localhost:5678/webhook/incident-alert` matching the Alertmanager payload shape.

## Notes

- This is a **demo/sandbox**, not production tooling — `app.py` injects faults on purpose so there's always an incident to respond to.
- The AI never commits directly to `main`; it always works on a fresh `ai-fix-<timestamp>` branch and opens a PR.
- Patches are rejected automatically (routed to the manual-review email instead) if the model's self-reported risk is anything above `medium`, or if it fails to return a patch at all.
