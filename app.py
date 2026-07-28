"""
app.py — Flask web demo for the LLM-based low-carbon concrete optimizer.
Run:  python app.py
Open: http://localhost:5000
"""

import json
import os
import queue
import threading
import traceback

from flask import Flask, jsonify, render_template, request, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv()

from optimizer_core import (
    ExperimentConfig, GWP_FACTORS, LB_YD3_TO_KG_M3, RAW_VARS,
    check_feasibility, compute_gwp, get_bounds, get_derived,
    load_df, load_surrogate, predict, run_llm, select_few_shot,
)

# ── Startup: load data + model once ──────────────────────────────────────────
DATA_PATH  = "data/Super_Cleaned_Concrete_Data.csv"
MODEL_PATH = "concrete_catboost_optimized.pkl"

print("[startup] Loading dataset …")
_df      = load_df(DATA_PATH)
_raw_b, _der_b = get_bounds(_df)
print(f"[startup] {len(_df)} rows  PC=[{_raw_b['PC']['min']:.0f},{_raw_b['PC']['max']:.0f}] kg/m3")

print("[startup] Loading surrogate model …")
_meta    = load_surrogate(MODEL_PATH)
print("[startup] Ready.")

app = Flask(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_mix(data: dict) -> dict:
    """Extract and cast raw ingredient values from a request payload."""
    return {v: float(data.get(v, 0.0)) for v in RAW_VARS}


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.route("/api/bounds")
def api_bounds():
    return jsonify({"raw": _raw_b, "der": _der_b})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        mix   = _parse_mix(request.json)
        preds = predict(_meta, mix)
        gwp   = compute_gwp(mix)
        der   = get_derived(mix)
        return jsonify({"ok": True, "preds": preds, "gwp": round(gwp, 2), "derived": der})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/optimize")
def api_optimize():
    """
    Server-Sent Events stream for the LLM optimizer.
    Query params:
      strength  float  (default 50)  — 28-day strength floor (MPa)
      knowledge bool   (default true) — inject domain knowledge
      rag       str    (default text) — "text" | "tabular" | "none"
      iters     int    (default 30)  — max feasible iterations
    """
    strength_min = float(request.args.get("strength", 50))
    use_knowledge = request.args.get("knowledge", "true").lower() == "true"
    rag_mode      = request.args.get("rag", "text")       # "none" | "tabular" | "text"
    max_iters     = min(int(request.args.get("iters", 30)), 50)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        def _err():
            yield "data: " + json.dumps({"error": "GEMINI_API_KEY not set"}) + "\n\n"
        return Response(_err(), mimetype="text/event-stream")

    q: "queue.Queue[dict | None]" = queue.Queue()

    def on_iter(record: dict):
        q.put({
            "type":       "iter",
            "feasible":   record.get("feasible", True),
            "iteration":  record["iteration"],
            "gwp":        round(record["gwp"], 2),
            "pred_28day": round(record.get("pred_28day", record.get("pred_28day", 0)), 2),
            "pred_56day": round(record.get("pred_56day", 0), 2),
            "str_margin": round(record.get("str_margin", 0), 2),
            "mode":       record["mode"],
            "reasoning":  record.get("reasoning", "")[:300],
            "PC":  round(record.get("PC", 0), 1),
            "FA":  round(record.get("FA", 0), 1),
            "SC":  round(record.get("SC", 0), 1),
            "WATER": round(record.get("WATER", 0), 1),
            "ACC": round(record.get("ACC", 0), 1),
            "WR":  round(record.get("WR", 0), 1),
            "FAGG": round(record.get("FAGG", 0), 1),
            "CAGG": round(record.get("CAGG", 0), 1),
        })

    def run():
        try:
            rag_map   = {"none": "none", "tabular": "dynamic", "text": "dynamic"}
            rag_fmt   = {"none": "tabular", "tabular": "tabular", "text": "text"}
            cfg = ExperimentConfig(
                name="web_demo",
                gemini_api_key=api_key,
                gemini_model="gemini-2.5-flash",
                strength_min=strength_min,
                max_iters=max_iters,
                use_knowledge_table=use_knowledge,
                use_situation_rules=use_knowledge,
                use_few_shot=True,
                rag_mode=rag_map.get(rag_mode, "dynamic"),
                rag_format=rag_fmt.get(rag_mode, "text"),
                output_prefix="results/web_demo",
            )
            few_shot  = select_few_shot(_df, strength_min, n=3)
            traj, summary, calls = run_llm(
                _raw_b, _der_b, _meta, ga_ref=None,
                few_shot=few_shot, cfg=cfg, df=_df, on_iter=on_iter,
            )
            best = min(traj, key=lambda r: r["gwp"]) if traj else None
            q.put({
                "type":    "done",
                "calls":   calls,
                "iters":   len(traj),
                "summary": summary,
                "best": {
                    "gwp":        round(best["gwp"], 2),
                    "pred_28day": round(best["pred_28day"], 2),
                    "pred_56day": round(best["pred_56day"], 2),
                    "PC":  round(best["PC"], 1),
                    "FA":  round(best["FA"], 1),
                    "SC":  round(best["SC"], 1),
                    "WATER": round(best["WATER"], 1),
                    "ACC": round(best.get("ACC", 0), 1),
                    "WR":  round(best.get("WR", 0), 1),
                    "FAGG": round(best.get("FAGG", 0), 1),
                    "CAGG": round(best.get("CAGG", 0), 1),
                } if best else None,
            })
        except Exception:
            q.put({"type": "error", "message": traceback.format_exc()})
        finally:
            q.put(None)  # sentinel

    t = threading.Thread(target=run, daemon=True)
    t.start()

    def generate():
        yield "data: " + json.dumps({"type": "start", "strength": strength_min,
                                     "knowledge": use_knowledge, "rag": rag_mode,
                                     "iters": max_iters}) + "\n\n"
        while True:
            try:
                msg = q.get(timeout=180)
            except queue.Empty:
                yield "data: " + json.dumps({"type": "heartbeat"}) + "\n\n"
                continue
            if msg is None:
                break
            yield "data: " + json.dumps(msg) + "\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Main page ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    bounds_json = json.dumps({"raw": _raw_b, "der": _der_b})
    gwp_json    = json.dumps(GWP_FACTORS)
    return render_template("index.html", bounds=bounds_json, gwp_factors=gwp_json)


if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5000)
