"""
Web interface for the trained complaint analyzer.

Serves a single page that sends complaints to the trained scikit-learn model and
displays the emotion probabilities, an urgency score and a triage priority.

Uses only the Python standard library for the server, so there is nothing extra
to install beyond what the models already need.

Run:  python src/app.py
Then open http://localhost:8000 in a browser.
"""
import os
import re
import sys
import json
import webbrowser
import threading
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import joblib

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C
import auth
from login_page import LOGIN_PAGE

PORT = int(os.environ.get("PORT", 8000))
ON_SERVER = "PORT" in os.environ   # set by hosting platforms such as Render
REVIEW_THRESHOLD = 0.60            # below this confidence, ask a human to check

ALLOW_SIGNUP = os.environ.get("ALLOW_SIGNUP", "1") != "0"      # set ALLOW_SIGNUP=0 to close registration
TEMP_ACCOUNTS = ON_SERVER and os.environ.get("PERSISTENT_ACCOUNTS") != "1"
MAX_BODY = 20_000                  # bytes; larger requests are refused
COOKIE = "ca_session"
STORE = None                       # the AuthStore, created in main()

# Same-origin only: nothing is loaded from, or sent to, any other site.
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
       "form-action 'self'")

# ---------------------------------------------------------------------------
# Urgency scoring. Still rule-based: the complaint corpus carries P1-P4 labels
# but modelling them is Part 4. The safety override below is deliberately a
# rule and will stay one, because a missed gas leak is not an acceptable error
# however good a classifier becomes.
# ---------------------------------------------------------------------------
URGENCY_TERMS = [
    (5, ["immediately", "right now", "emergency", "urgent", "urgently", "asap",
         "gas leak", "smell of gas", "unsafe", "hazard", "on fire", "smoke",
         "short circuit", "electric shock", "burning smell", "sparking",
         "smell burning", "wiring", "live wire", "leaking",
         "hospital", "ambulance", "injured", "stranded", "unauthorized",
         "security breach", "hacked", "theft", "turant", "abhi", "unauthorised"]),
    (4, ["today", "tonight", "tomorrow", "deadline", "flight", "exam",
         "not working", "cannot access", "blocked", "locked out", "outage",
         "server down", "no service", "no power", "no water", "debited twice",
         "double charged", "not refunded", "refund", "never received",
         "wrong medicine", "jaldi"]),
    (3, ["this week", "escalate", "overdue", "still waiting", "no update",
         "no reply", "third time", "fourth time", "fifth time", "disconnect"]),
    (2, ["soon", "pending", "delay", "delayed", "follow up", "waiting",
         "mismatch", "issue", "problem", "complaint", "error"]),
    (1, ["sometime", "suggestion", "feedback", "minor", "whenever"]),
]
# ---------------------------------------------------------------------------
# Safety rules. Each rule is a regular expression that needs context rather than
# a bare keyword, so "the delivery did not make it on time" does not look like
# fraud and "burning through my battery" does not look like a fire.
#
#   tier A  danger to life or property   urgency never below 90
#   tier B  serious harm if it waits     urgency never below 76
#
# The tier sets a floor, whatever the tone: a politely worded gas leak is still
# a gas leak. Signals that make a situation worse (it is happening now, children
# or elderly people are involved, shouting) add points on top of the floor, so
# two different emergencies do not end up with an identical score.
# ---------------------------------------------------------------------------
_MONEY = r"(transaction|payment|transfer|withdrawal|debit|charge|purchase)s?"
_NOT = r"(never|did not|didn't|do not|don't|have not|haven't)"
HAZARD_RULES = [(tier, label, re.compile(rx, re.I)) for tier, label, rx in [
    ("A", "gas leak",
     r"\bgas leak\w*|\bgas (is |was )?(leaking|escaping|leaked)\b|\bleak\w* gas\b"
     r"|\bsmell\w*( strongly| strong)?( of| like)? gas\b|\bgas smell\w*|\bsmell\w* gas\b"
     r"|\bcylinder\b[^.?!]{0,25}\bleak\w*"),
    ("A", "fire or smoke",
     r"\bon fire\b|\bcaught fire\b|\bcatch(ing)? fire\b|\bfire (broke out|started)\b|\bsmoke\b"),
    ("A", "burning smell",
     r"\bburning (smell|plastic|rubber|wires?|wiring)\b|\bsmell\w*[^.?!]{0,15}\bburn(t|ing)\b"
     r"|\bburn(t|ing) (smell|odou?r)\b"),
    ("A", "electrical danger",
     r"\bshort[- ]?circuit\w*|\belectric(al)? shock\w*|\blive wires?\b|\bsparking\b|\bsparks?\b"
     r"|\bgot (a |an )?shock"),
    ("A", "medical emergency",
     r"\bchest pain\b|\bunconscious\b|\bbleeding\b|\bambulance\b|\bfood poisoning\b"
     r"|\bnot breathing\b|\bheart attack\b|\bseizure\b|\boverdose\b"),
    ("A", "physical danger",
     r"\bbrakes? (have |has )?(failed|stopped working|not working)\b|\bstuck between floors\b"
     r"|\btrapped (in|inside) (the |a |my )?(lift|elevator|car|room|bathroom|building|house)\b"
     r"|\bbreak[- ]?in\b|\brobbery\b|\bburglar\w*"),
    ("B", "medicine shortage",
     r"\bout of (tablets|medicine|medication|insulin|oxygen|pills)\b"
     r"|\b(wrong|incorrect) (medicine|medication|dose)\b"),
    ("B", "unauthorised transaction",
     r"\b(unauthori[sz]ed|fraudulent|suspicious)\b[^.?!]{0,30}\b" + _MONEY + r"\b"
     r"|\b" + _MONEY + r"\b[^.?!]{0,40}\b(i|we)\s+" + _NOT + r"\s+(made|make|authori[sz]ed?|approved?|recogni[sz]e)\b"
     r"|\b(i|we)\s+" + _NOT + r"\s+(made|make|authori[sz]e|approve)\b[^.?!]{0,25}\b" + _MONEY + r"\b"
     r"|\b(do not|don't) recogni[sz]e\b[^.?!]{0,25}\b" + _MONEY + r"\b"
     r"|\bwithout (my )?(permission|consent|authori[sz]ation)\b[^.?!]{0,40}\b(money|account|payment|card)\b"
     r"|\b(deduct\w*|debit\w*|withdr\w*)\b[^.?!]{0,40}\bwithout (my )?(permission|consent|authori[sz]ation)"),
    ("B", "account compromised",
     r"\bhacked\b|\b(account|card|password|otp)\b[^.?!]{0,25}\b(compromised|stolen)\b"
     r"|\b(compromised|stolen)\b[^.?!]{0,15}\b(account|card|password)\b"
     r"|\bsomeone (has|is using|used|knows) my (card|otp|password|account)\b"
     r"|\bblock (my|the) (card|account)\b|\b(shared|gave) (my )?otp\b"),
]]
RESOLVED = re.compile(r"\b(resolved|handled|fixed|fixing|sorted|reversed|refunded|flagged|caught)\b", re.I)
HAZARD_FLOOR = {"A": 90, "B": 76}
IMMEDIATE = re.compile(r"\b(now|immediately|right away|asap|at once|urgent(ly)?|emergency|today|tonight)\b", re.I)
VULNERABLE = re.compile(r"\b(kids?|child(ren)?|baby|babies|infant|toddler|elderly|old (man|woman)"
                        r"|grand(mother|father|parents?)|father|mother|parents?|pregnant|daughter|son)\b", re.I)


def check_hazards(text):
    """Return (labels, tier) for every safety rule the text triggers."""
    labels, tier = [], None
    for t, label, rx in HAZARD_RULES:
        if rx.search(text):
            labels.append(label)
            if tier is None or t < tier:      # "A" sorts before "B"
                tier = t
    return labels, tier


DE_ESCALATE = ["no rush", "no hurry", "whenever you get time", "not urgent",
               "take your time", "at your convenience"]
THREATS = ["consumer court", "legal action", "lawyer", "ombudsman",
           "social media", "cancel my subscription", "close my account"]

CATEGORIES = {
    "Billing": ["bill", "invoice", "charge", "payment", "refund", "debited",
                "emi", "gst", "amount", "transaction", "deduct"],
    "Delivery": ["delivery", "deliver", "order", "courier", "parcel",
                 "package", "dispatch", "shipment"],
    "Technical": ["app", "website", "server", "login", "otp", "network",
                  "internet", "connection", "error", "crash", "broadband"],
    "Safety": ["gas", "leak", "fire", "shock", "spark", "burning", "hazard",
               "unsafe", "wiring", "cylinder", "smoke", "medicine", "lift"],
    "Staff": ["agent", "executive", "technician", "engineer", "staff", "rude",
              "behaviour", "customer care", "driver"],
    "Product": ["product", "item", "quality", "broken", "damaged", "defective",
                "replacement", "warranty", "battery"],
}
EMOTION_WEIGHT = {"anger": 20, "frustration": 16, "anxiety": 26,
                  "disappointment": 10, "neutral": 0, "satisfaction": -8}
SLA = {"P1": "Within 1 hour", "P2": "Same day",
       "P3": "Within 2 working days", "P4": "Normal queue"}


def hits(text, terms):
    found = []
    for t in terms:
        if re.search(r"\b" + re.escape(t), text):
            found.append(t)
    return found


def score_urgency(text, emotion, confidence):
    low = text.lower()
    score, matched = 8.0, []
    for weight, terms in URGENCY_TERMS:
        for t in hits(low, terms):
            score += weight * 2.6
            matched.append({"term": t, "weight": weight})

    excl = min(text.count("!"), 4)
    caps = min(len(re.findall(r"\b[A-Z]{3,}\b", text)), 3)
    score += excl * 2 + caps * 3

    threat = hits(low, THREATS)
    calm = hits(low, DE_ESCALATE)
    score += min(len(threat) * 16, 24)
    score -= min(len(calm) * 12, 20)
    score += EMOTION_WEIGHT.get(emotion, 0) * confidence

    hazard, tier = check_hazards(text)
    # A thank-you for something already dealt with is not an open emergency. Only the
    # lower tier is excused, and only when the model reads the message as satisfaction
    # AND the text uses resolved-language. A danger-to-life hazard is never excused.
    if hazard and tier == "B" and emotion == "satisfaction" and confidence >= 0.6 \
            and RESOLVED.search(text):
        hazard, tier = [], None
    overridden, floor, aggravators = False, None, []
    if hazard:
        floor = HAZARD_FLOOR[tier]
        extra = 0
        now = len({m.group(0).lower() for m in IMMEDIATE.finditer(text)})
        if now:
            pts = min(now, 2) * 3
            extra += pts
            aggravators.append(f"it is happening now +{pts}")
        if VULNERABLE.search(text):
            extra += 4
            aggravators.append("children or elderly people involved +4")
        tone = min(4, excl + caps)
        if tone:
            extra += tone
            aggravators.append(f"shouting or exclamation marks +{tone}")
        # max, not sum: where the ordinary signals already score higher, they win,
        # so the same evidence is never counted twice.
        if floor + extra > score:
            score, overridden = float(floor + extra), True

    score = max(0, min(100, round(score)))
    priority = ("P1" if score >= 70 else "P2" if score >= 52
                else "P3" if score >= 30 else "P4")
    return {"score": score, "priority": priority, "sla": SLA[priority],
            "matched": matched, "threats": threat, "de_escalators": calm,
            "hazards": hazard, "overridden": overridden,
            "hazard_tier": tier, "floor": floor, "aggravators": aggravators,
            "exclamations": excl, "caps": caps}


def detect_category(text):
    low = text.lower()
    best, best_n = "General", 0
    for cat, words in CATEGORIES.items():
        n = sum(1 for w in words if re.search(r"\b" + re.escape(w), low))
        if n > best_n:
            best, best_n = cat, n
    return best


# ---------------------------------------------------------------------------
MODEL = None


def explain(text, emotion):
    """Which words pushed the prediction, by leave-one-word-out.

    Each word is removed in turn and the drop in the predicted emotion's
    probability is measured. A large drop means that word mattered. This works
    for any model, including ones with stacked and lexicon features where
    reading coefficients directly would be misleading. All variants go through
    the model in a single batch, so it costs one extra prediction call.
    """
    tokens = re.findall(r"\S+", text)[:60]
    if len(tokens) < 2:
        return []
    idx = list(MODEL.classes_).index(emotion)
    variants = [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(len(tokens))]
    base = MODEL.predict_proba([text])[0][idx]
    probs = MODEL.predict_proba(variants)[:, idx]
    return [{"word": tok, "weight": round(float(base - p), 4)}
            for tok, p in zip(tokens, probs)]


def build_result(text, probs, with_explanation=True):
    ranked = sorted(zip(MODEL.classes_, probs), key=lambda x: -x[1])
    emotion, conf = ranked[0][0], float(ranked[0][1])
    runner_up, runner_conf = ranked[1][0], float(ranked[1][1])
    urgency = score_urgency(text, emotion, conf)

    review = None
    if conf < REVIEW_THRESHOLD:
        review = (f"The model is only {conf*100:.0f}% confident. It is torn between "
                  f"{emotion} and {runner_up} ({runner_conf*100:.0f}%). "
                  f"A person should check this one.")

    return {
        "text": text,
        "emotion": emotion,
        "confidence": round(conf, 4),
        "probabilities": {k: round(float(v), 4) for k, v in ranked},
        "category": detect_category(text),
        "urgency": urgency,
        "explanation": explain(text, emotion) if with_explanation else [],
        "needs_review": review,
    }


def analyze(text):
    probs = MODEL.predict_proba([text])[0]
    return build_result(text, probs)


# ---------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complaint Analyzer</title>
<style>
:root{--bg:#DDE2E6;--panel:#FBFBFA;--ink:#16202B;--muted:#5D6B77;--line:#C4CCD3;--soft:#E2E7EA;
--p1:#B23A2B;--p2:#C07414;--p3:#3E7C8C;--p4:#5F7A63;
--anger:#B23A2B;--frustration:#C07414;--disappointment:#75629E;
--anxiety:#3E7C8C;--satisfaction:#4F7A4A;--neutral:#6E7C87;
--sans:"Segoe UI",system-ui,-apple-system,sans-serif;--mono:Consolas,"SF Mono",monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:var(--sans);font-size:15px;line-height:1.55}
.wrap{max-width:1060px;margin:0 auto;padding:26px 18px 60px}
header{display:flex;align-items:flex-end;gap:16px;flex-wrap:wrap;
padding-bottom:14px;border-bottom:2px solid var(--ink);margin-bottom:22px}
h1{margin:0;font-size:clamp(22px,3vw,30px);letter-spacing:-.02em}
header p{margin:3px 0 0;color:var(--muted);font-size:13.5px;max-width:54ch}
.badge{font-family:var(--mono);font-size:12px;color:var(--muted);border:1px solid var(--line);
padding:3px 8px;background:var(--panel);border-radius:2px;margin-left:auto;white-space:nowrap}
.grid{display:grid;grid-template-columns:.9fr 1.1fr;gap:18px;align-items:start}
@media(max-width:840px){.grid{grid-template-columns:1fr}.badge{margin-left:0}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:18px}
h2{margin:0 0 3px;font-size:15px}.hint{margin:0 0 13px;color:var(--muted);font-size:13px}
h3{font-size:12.5px;color:var(--muted);margin:0 0 8px;font-weight:600}
textarea{width:100%;min-height:140px;font-family:var(--sans);font-size:15px;padding:11px;
border:1px solid var(--line);border-radius:2px;resize:vertical;line-height:1.6}
textarea:focus,button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
button{font-family:var(--sans);font-size:14px;font-weight:500;cursor:pointer;padding:9px 16px;
border:1px solid var(--ink);background:var(--ink);color:#fff;border-radius:2px}
button:hover{background:#28394a}button.g{background:transparent;color:var(--ink);border-color:var(--line)}
button.g:hover{background:#EDF0F2}button.sm{padding:5px 10px;font-size:12.5px}
.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-top:12px}
.chip{display:block;width:100%;text-align:left;font-size:13px;background:#fff;border:1px solid var(--line);
padding:7px 10px;margin-bottom:5px;border-radius:2px;color:var(--ink);cursor:pointer;font-family:var(--sans)}
.chip:hover{background:#EDF0F2}
.score{font-family:var(--mono);font-size:46px;line-height:1;letter-spacing:-.03em}
.of{font-family:var(--mono);font-size:13px;color:var(--muted)}
.readout{display:flex;align-items:baseline;gap:9px;margin-bottom:9px}.band{margin-left:auto;font-weight:600}
.track{position:relative;height:28px;border:1px solid var(--line);display:flex;border-radius:2px;overflow:hidden}
.seg{height:100%;position:relative}
.seg span{position:absolute;left:6px;top:50%;transform:translateY(-50%);font-family:var(--mono);font-size:10.5px;opacity:.62}
.needle{position:absolute;top:-6px;bottom:-6px;width:3px;background:var(--ink);transition:left .4s cubic-bezier(.2,.7,.2,1)}
.scale{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-top:5px}
.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin:18px 0}
.fact{background:var(--panel);padding:10px 12px}.fact dt{font-size:11.5px;color:var(--muted)}
.fact dd{margin:0;font-size:15px;font-weight:600}
.sec{margin-bottom:18px}
.bar{display:grid;grid-template-columns:104px 1fr 46px;gap:9px;align-items:center;margin-bottom:5px;font-size:13px}
.t{height:9px;background:var(--soft);border-radius:1px;overflow:hidden}
.t i{display:block;height:100%;transition:width .4s}
.v{font-family:var(--mono);font-size:12px;color:var(--muted);text-align:right}
.tag{display:inline-block;font-size:11.5px;padding:1px 6px;border:1px solid var(--line);background:#fff;
border-radius:2px;color:var(--muted);margin:0 4px 4px 0}
.flag{border-left:3px solid var(--p1);background:#F8ECEA;padding:8px 11px;font-size:13px;margin-bottom:12px}
.flag.review{border-left-color:var(--p2);background:#FBF3E6}
.why{background:#fff;border:1px solid var(--soft);border-radius:2px;padding:12px;font-size:15px;line-height:2}
.why span{padding:1px 2px;border-radius:2px}
.top{margin-top:8px;font-size:12.5px;color:var(--muted)}
pre{background:#fff;border:1px solid var(--soft);padding:11px;font-family:var(--mono);font-size:11.5px;
overflow:auto;max-height:220px;border-radius:2px;margin:0}
details summary{cursor:pointer;font-size:12.5px;color:var(--muted);font-weight:600;margin-bottom:8px}
.queue{margin-top:18px}.qhead{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.qhead h2{margin:0}.count{font-family:var(--mono);font-size:12px;color:var(--muted)}.spacer{flex:1}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--soft);vertical-align:top}
th{font-size:12px;font-weight:600;color:var(--muted);border-bottom:1px solid var(--line)}
td.num{text-align:right;font-family:var(--mono)}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 6px;border-radius:2px;color:#fff;font-family:var(--mono)}
.pill.P1{background:var(--p1)}.pill.P2{background:var(--p2)}.pill.P3{background:var(--p3)}.pill.P4{background:var(--p4)}
.done td{opacity:.42}.warn{color:var(--p2);font-weight:600}
.empty{color:var(--muted);font-size:14px;padding:6px 0}.muted{color:var(--muted)}
.cust{margin-top:14px;border-top:1px solid var(--soft);padding-top:12px}
.cgrid{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.cgrid label{font-size:12px;color:var(--muted);display:block}
.cgrid input,.cgrid select{width:100%;margin-top:3px;padding:7px 9px;font-family:var(--sans);font-size:14px;border:1px solid var(--line);border-radius:2px;background:#fff;color:var(--ink)}
.cgrid input:focus,.cgrid select:focus{outline:2px solid var(--ink);outline-offset:1px}
@media(max-width:420px){.cgrid{grid-template-columns:1fr}}
td a{color:var(--ink)}.cust a{color:var(--p3)}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:18px}
.kpi{background:var(--panel);padding:12px 14px}.kpi .n{font-family:var(--mono);font-size:28px;line-height:1.1}
.kpi .l{font-size:12px;color:var(--muted);margin-top:2px}
@media(max-width:700px){.kpis{grid-template-columns:1fr 1fr}}
.dgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}
@media(max-width:840px){.dgrid{grid-template-columns:1fr}}
.dbar{display:grid;grid-template-columns:96px 1fr 30px;gap:8px;align-items:center;margin-bottom:6px;font-size:13px}
.panel + .panel{margin-top:18px}
.who{display:flex;align-items:center;gap:8px;font-size:13px}
#whoemail{max-width:230px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);background:var(--ink);color:#fff;
font-size:13px;padding:9px 15px;border-radius:2px;opacity:0;pointer-events:none;transition:opacity .2s}
.toast.show{opacity:1}
</style></head><body><div class="wrap">
<header><div><h1>Complaint analyzer</h1>
<p>Emotion from a trained scikit-learn model with lexicon and transfer-learning features. Urgency from a weighted rule engine with a safety override.</p></div>
<div class="badge">73.8% accuracy &middot; 0.735 macro-F1</div>
<div class="who"><span id="whoemail" class="muted"></span><button class="g sm" onclick="logout()">Log out</button></div></header>

<div class="grid">
<div class="panel"><h2>Complaint</h2><p class="hint">Type anything a customer might write. Ctrl+Enter to analyze.</p>
<textarea id="t" placeholder="Type a complaint..."></textarea>
<div class="cust"><h3>Customer details <span class="muted" style="font-weight:400">(needed to add to the queue)</span></h3>
<div class="cgrid">
<label>Name<input id="c_name" type="text" maxlength="80" autocomplete="off" placeholder="Riya Sharma"></label>
<label>Phone<input id="c_phone" type="tel" maxlength="20" autocomplete="off" placeholder="98765 43210"></label>
<label>Email<input id="c_email" type="email" maxlength="120" autocomplete="off" placeholder="riya@example.com"></label>
<label>Customer or order ID<input id="c_id" type="text" maxlength="40" autocomplete="off" placeholder="ORD-88231"></label>
<label>Preferred contact<select id="c_pref"><option>Phone call</option><option>WhatsApp</option><option>Email</option></select></label></div>
<p class="hint" style="margin:8px 0 0">Kept in this browser only. The model receives just the complaint text.
<a href="#" onclick="sampleCustomer();return false">Fill a sample customer</a></p></div>
<div class="row"><button onclick="go()">Analyze</button>
<button class="g" id="addq" onclick="addToQueue()" disabled>Add to queue</button>
<button class="g" onclick="document.getElementById('t').value='';">Clear</button></div>
<div style="margin-top:16px;border-top:1px solid var(--soft);padding-top:13px">
<h3>Try one of these</h3><div id="samples"></div></div></div>

<div class="panel"><h2>Result</h2><p class="hint" id="hint">Analyze a complaint to see the output.</p>
<div id="out" style="display:none">
<div id="flags"></div>
<div class="readout"><span class="score" id="sc">0</span><span class="of">/ 100 urgency</span><span class="band" id="bd"></span></div>
<div class="track">
<div class="seg" style="width:30%;background:rgba(95,122,99,.18)"><span>P4</span></div>
<div class="seg" style="width:22%;background:rgba(62,124,140,.2)"><span>P3</span></div>
<div class="seg" style="width:18%;background:rgba(192,116,20,.22)"><span>P2</span></div>
<div class="seg" style="width:30%;background:rgba(178,58,43,.22)"><span>P1</span></div>
<div class="needle" id="nd" style="left:0%"></div></div>
<div class="scale"><span>0</span><span>30</span><span>52</span><span>70</span><span>100</span></div>
<dl class="facts">
<div class="fact"><dt>Emotion</dt><dd id="em">&mdash;</dd></div>
<div class="fact"><dt>Category</dt><dd id="ct">&mdash;</dd></div>
<div class="fact"><dt>Respond by</dt><dd id="sl">&mdash;</dd></div></dl>
<div class="sec"><h3>Why the model decided this</h3><div class="why" id="why"></div>
<div class="top" id="top"></div></div>
<div class="sec"><h3>Model probabilities</h3><div id="bars"></div></div>
<div class="sec"><h3>Urgency keywords matched</h3><div id="kw"></div></div>
<details><summary>API response</summary><pre id="js"></pre></details>
</div></div></div>

<div class="panel" id="dash" style="margin-top:18px"><h2>Dashboard</h2>
<p class="hint">Live summary of everything in the queue.</p>
<div class="kpis">
<div class="kpi"><div class="n" id="k1">0</div><div class="l">Complaints in queue</div></div>
<div class="kpi"><div class="n" id="k2" style="color:var(--p1)">0</div><div class="l">Critical (P1), still open</div></div>
<div class="kpi"><div class="n" id="k3">0</div><div class="l">Average urgency</div></div>
<div class="kpi"><div class="n" id="k4" style="color:var(--p2)">0</div><div class="l">Flagged for human review</div></div></div>
<div class="dgrid">
<div><h3>Emotions</h3><div id="d_em"></div></div>
<div><h3>Priorities</h3><div id="d_pr"></div></div>
<div><h3>Departments</h3><div id="d_ct"></div></div></div></div>

<div class="panel queue">
<div class="qhead"><h2>Triage queue</h2><span class="count" id="qc">0 complaints</span><div class="spacer"></div>
<button class="g sm" onclick="exportCsv()">Download CSV</button>
<button class="g sm" onclick="clearQueue()">Clear</button></div>
<p class="hint">Sorted by urgency, most critical first. This is the order a support team should work in.</p>
<div class="scroll"><table><thead><tr><th style="width:60px">Priority</th><th style="width:62px">Urgency</th>
<th style="width:104px">Emotion</th><th style="width:190px">Customer</th><th>Complaint</th><th style="width:90px"></th></tr></thead>
<tbody id="qb"></tbody></table></div>
<div class="empty" id="qe">Empty. Analyze a complaint, fill in the customer details, then click Add to queue.</div>
</div>
</div><div class="toast" id="toast"></div>
<script>
const ME=__ME__;   // filled in by the server for the signed-in user
const QKEY='ca_queue_'+ME.uid;   // each account keeps its own queue in this browser
const EC={anger:'--anger',frustration:'--frustration',disappointment:'--disappointment',
anxiety:'--anxiety',satisfaction:'--satisfaction',neutral:'--neutral'};
const RGB={anger:'178,58,43',frustration:'192,116,20',disappointment:'117,98,158',
anxiety:'62,124,140',satisfaction:'79,122,74',neutral:'110,124,135'};
const SAMPLES=["I can smell burning from the wiring near the meter and I am scared.",
"This is the fifth time I am calling about the same problem and nothing has changed.",
"Thank you for sorting this out so quickly, I really appreciate it.",
"My money was debited twice and the refund has not come. My EMI is due tomorrow.",
"Could you please check the gas leak in flat 4B whenever convenient? No rush.",
"Please share the invoice for order 88231 for my records.",
"I expected much better quality from a brand like yours. Very disappointed.",
"I was charged a fee I do not understand, can someone explain it?"];
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cap=s=>s.charAt(0).toUpperCase()+s.slice(1);
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');
clearTimeout(t._t);t._t=setTimeout(()=>t.classList.remove('show'),3200);}
const sd=document.getElementById('samples');
SAMPLES.forEach(s=>{const b=document.createElement('button');b.className='chip';
b.textContent=s.length>62?s.slice(0,62)+'...':s;b.onclick=()=>{document.getElementById('t').value=s;go();};sd.appendChild(b);});

let last=null;
async function go(){
 const text=document.getElementById('t').value.trim();if(!text)return;
 let d;try{const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({text})});
  if(r.status===401){toast('Your session has ended. Please sign in again.');setTimeout(()=>location.reload(),1500);return;}
  d=await r.json();if(d.error)throw new Error(d.error);}
 catch(e){toast('Could not reach the model.');return;}
 last=d;document.getElementById('addq').disabled=false;
 document.getElementById('hint').style.display='none';document.getElementById('out').style.display='block';
 const u=d.urgency;document.getElementById('sc').textContent=u.score;
 const bd=document.getElementById('bd');
 bd.textContent=u.priority+' '+({P1:'Critical',P2:'High',P3:'Medium',P4:'Low'}[u.priority]);
 bd.style.color='var(--'+u.priority.toLowerCase()+')';
 document.getElementById('nd').style.left=u.score+'%';
 const em=document.getElementById('em');em.textContent=cap(d.emotion)+' '+Math.round(d.confidence*100)+'%';
 em.style.color='var('+EC[d.emotion]+')';
 document.getElementById('ct').textContent=d.category;document.getElementById('sl').textContent=u.sla;

 let f='';
 if(u.hazards.length)f+=`<div class="flag"><b>Safety rule.</b> ${esc(u.hazards.join(', '))} detected, so urgency is held at ${u.floor} or above whatever the tone.${u.de_escalators.length?' The customer wrote &ldquo;'+u.de_escalators.map(esc).join('&rdquo;, &ldquo;')+'&rdquo;, but a safety risk is treated as urgent regardless.':''}${u.aggravators.length?' Added on top: '+esc(u.aggravators.join(', '))+'.':''}</div>`;
 if(d.needs_review)f+=`<div class="flag review"><b>Needs human review.</b> ${esc(d.needs_review)}</div>`;
 document.getElementById('flags').innerHTML=f;

 // Word highlighting: tint proportional to how much each word supported the prediction.
 const ex=d.explanation||[];const mx=Math.max(0.0001,...ex.map(e=>e.weight));const rgb=RGB[d.emotion];
 document.getElementById('why').innerHTML=ex.length?ex.map(e=>{
   const a=e.weight>0?Math.min(0.75,0.08+e.weight/mx*0.67):0;
   return `<span style="background:rgba(${rgb},${a.toFixed(2)})" title="${(e.weight*100).toFixed(1)} points">${esc(e.word)}</span>`;
 }).join(' '):esc(d.text);
 const top=ex.filter(e=>e.weight>0.005).sort((a,b)=>b.weight-a.weight).slice(0,5);
 document.getElementById('top').innerHTML=top.length?'Strongest words: '+top.map(e=>
   `<span class="tag">${esc(e.word)} +${(e.weight*100).toFixed(0)}</span>`).join(''):
   'No single word dominates. The prediction comes from the overall wording.';

 document.getElementById('bars').innerHTML=Object.entries(d.probabilities).map(([k,v])=>
  `<div class="bar"><span>${k}</span><span class="t"><i style="width:${(v*100).toFixed(1)}%;background:var(${EC[k]})"></i></span><span class="v">${(v*100).toFixed(1)}%</span></div>`).join('');
 const tags=u.matched.map(m=>`<span class="tag">${esc(m.term)} (${m.weight})</span>`).join('')
  +u.threats.map(t=>`<span class="tag">threat: ${esc(t)}</span>`).join('')
  +u.de_escalators.map(t=>`<span class="tag">can wait: ${esc(t)}</span>`).join('');
 document.getElementById('kw').innerHTML=tags||'<span class="muted">None matched.</span>';
 document.getElementById('js').textContent=JSON.stringify(d,null,2);
}

// ---- Customer details ----
const $=id=>document.getElementById(id);
const EMAIL=/^[^\s@<>"']+@[^\s@<>"']+\.[^\s@<>"']{2,}$/;
function normPhone(raw){
 const s=String(raw||'').trim(),digits=s.replace(/\D/g,'');
 if(!digits)return{ok:true,value:''};
 if(digits.length<7||digits.length>15)return{ok:false};
 return{ok:true,value:(s.startsWith('+')?'+':'')+digits};
}
function readCustomer(){
 const name=$('c_name').value.trim().slice(0,80),email=$('c_email').value.trim().slice(0,120),
  cid=$('c_id').value.trim().slice(0,40),pref=$('c_pref').value,ph=normPhone($('c_phone').value);
 if(!ph.ok)return{err:'That phone number does not look right. Use 7 to 15 digits.'};
 if(email&&!EMAIL.test(email))return{err:'That email address does not look right.'};
 if(!ph.value&&!email)return{err:'Add a phone number or an email so the customer can be contacted.'};
 if(pref==='Email'&&!email)return{err:'Preferred contact is email, but no email was entered.'};
 if(pref!=='Email'&&!ph.value)return{err:'Preferred contact is '+pref.toLowerCase()+', but no phone number was entered.'};
 return{data:{name,phone:ph.value,email,cid,pref}};
}
function sampleCustomer(){
 $('c_name').value='Riya Sharma';$('c_phone').value='98765 43210';
 $('c_email').value='riya.sharma@example.com';$('c_id').value='ORD-88231';$('c_pref').value='Phone call';
}
function clearCustomer(){['c_name','c_phone','c_email','c_id'].forEach(i=>$(i).value='');$('c_pref').value='Phone call';}

// ---- Triage queue ----
let queue=[];
try{queue=JSON.parse(localStorage.getItem(QKEY)||'[]');}catch(e){queue=[];}
function save(){try{localStorage.setItem(QKEY,JSON.stringify(queue));}catch(e){}}
function addToQueue(){
 if(!last)return;
 const c=readCustomer();
 if(c.err){toast(c.err);return;}
 const d=c.data;
 if(queue.some(q=>!q.done&&q.text===last.text&&(q.phone||'')===d.phone&&(q.email||'')===d.email)){
  toast('This complaint is already in the queue for this customer.');return;}
 queue.push({id:Date.now(),text:last.text,emotion:last.emotion,conf:last.confidence,
  score:last.urgency.score,priority:last.urgency.priority,category:last.category,
  review:!!last.needs_review,done:false,name:d.name,phone:d.phone,email:d.email,cid:d.cid,pref:d.pref});
 save();renderQueue();clearCustomer();toast('Added as '+last.urgency.priority+'.');
}
function custCell(q){
 const parts=[];
 if(q.name)parts.push('<b>'+esc(q.name)+'</b>');
 if(q.phone)parts.push('<a href="tel:'+esc(q.phone)+'">'+esc(q.phone)+'</a>');
 if(q.email)parts.push('<a href="mailto:'+esc(q.email)+'">'+esc(q.email)+'</a>');
 const meta=[];if(q.cid)meta.push('ID '+esc(q.cid));if(q.pref)meta.push('prefers '+esc(q.pref.toLowerCase()));
 if(meta.length)parts.push('<span class="muted" style="font-size:12px">'+meta.join(' &middot; ')+'</span>');
 return parts.length?parts.join('<br>'):'<span class="muted">No contact details</span>';
}
function renderQueue(){
 queue.sort((a,b)=>(a.done-b.done)||(b.score-a.score));
 renderDash();
 document.getElementById('qc').textContent=queue.length+(queue.length===1?' complaint':' complaints');
 document.getElementById('qe').style.display=queue.length?'none':'block';
 document.getElementById('qb').innerHTML=queue.map(q=>`<tr class="${q.done?'done':''}">
  <td><span class="pill ${esc(q.priority)}">${esc(q.priority)}</span></td><td class="num">${q.score}</td>
  <td style="color:var(${EC[q.emotion]||'--neutral'})">${cap(q.emotion)}${q.review?' <span class="warn" title="Needs human review">?</span>':''}</td>
  <td>${custCell(q)}</td>
  <td>${esc(q.text.length>110?q.text.slice(0,110)+'...':q.text)}</td>
  <td><button class="g sm" onclick="toggle(${q.id})">${q.done?'Reopen':'Resolve'}</button></td></tr>`).join('');
}
function toggle(id){const q=queue.find(x=>x.id===id);q.done=!q.done;save();renderQueue();}
function clearQueue(){queue=[];save();renderQueue();}
// A spreadsheet treats a cell starting with = + - @ as a formula. Prefix such cells with
// an apostrophe so a customer name like "=HYPERLINK(...)" is shown as text, not run.
function csvCell(v){
 let s=String(v==null?'':v);
 if(/^[=+\-@\t\r]/.test(s)&&!/^\+?\d+$/.test(s))s="'"+s;
 return '"'+s.replace(/"/g,'""')+'"';
}
function exportCsv(){
 if(!queue.length){toast('Queue is empty.');return;}
 const head=['priority','urgency','emotion','confidence','category','needs_review','status',
  'customer_name','phone','email','customer_id','preferred_contact','complaint'];
 const rows=[head].concat(queue.map(q=>[q.priority,q.score,q.emotion,q.conf,q.category,q.review,
  q.done?'resolved':'open',q.name||'',q.phone||'',q.email||'',q.cid||'',q.pref||'',q.text]));
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob(['\ufeff'+rows.map(r=>r.map(csvCell).join(',')).join('\n')],{type:'text/csv;charset=utf-8'}));
 a.download='triage-queue.csv';a.click();URL.revokeObjectURL(a.href);
}
// ---- Dashboard ----
function dbar(label,val,max,color){
 const w=max?Math.round(val/max*100):0;
 return `<div class="dbar"><span>${esc(label)}</span><span class="t"><i style="width:${w}%;background:${color}"></i></span><span class="v">${val}</span></div>`;
}
function renderDash(){
 const n=queue.length;
 document.getElementById('k1').textContent=n;
 document.getElementById('k2').textContent=queue.filter(q=>q.priority==='P1'&&!q.done).length;
 document.getElementById('k3').textContent=n?Math.round(queue.reduce((a,q)=>a+q.score,0)/n):0;
 document.getElementById('k4').textContent=queue.filter(q=>q.review&&!q.done).length;
 const count=k=>queue.reduce((m,q)=>(m[q[k]]=(m[q[k]]||0)+1,m),{});
 const em=count('emotion'),pr=count('priority'),ct=count('category');
 const none='<span class="muted">No data yet.</span>';
 const emKeys=Object.keys(EC);
 const mE=Math.max(1,...Object.values(em));
 document.getElementById('d_em').innerHTML=n?emKeys.map(k=>dbar(cap(k),em[k]||0,mE,'var('+EC[k]+')')).join(''):none;
 const mP=Math.max(1,...Object.values(pr));
 document.getElementById('d_pr').innerHTML=n?['P1','P2','P3','P4'].map(k=>dbar(k,pr[k]||0,mP,'var(--'+k.toLowerCase()+')')).join(''):none;
 const cats=Object.keys(ct).sort((a,b)=>ct[b]-ct[a]);const mC=Math.max(1,...Object.values(ct));
 document.getElementById('d_ct').innerHTML=n?cats.map(k=>dbar(k,ct[k],mC,'var(--ink)')).join(''):none;
}
async function logout(){
 try{await fetch('/api/logout',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});}catch(e){}
 location.reload();
}
document.getElementById('whoemail').textContent=ME.email;
document.getElementById('t').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')go();});
renderQueue();
</script></body></html>
"""


def me_json(user):
    """The signed-in user as a JSON literal that is safe to place inside a <script> tag."""
    return (json.dumps({"uid": user["uid"], "email": user["email"]})
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


class Handler(BaseHTTPRequestHandler):
    timeout = 15              # drop clients that stall in the middle of a request
    server_version = "ComplaintAnalyzer"
    sys_version = ""          # do not announce the Python version

    def log_message(self, *a):
        pass  # keep the terminal clean, and never write request details to a log

    # ---- helpers ----
    def _https(self):
        return self.headers.get("X-Forwarded-Proto", "").lower() == "https"

    def _send(self, code, body, ctype, cookie=None, extra=None):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", CSP)
        if self._https():
            self.send_header("Strict-Transport-Security", "max-age=15552000")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj, cookie=None, extra=None):
        self._send(code, json.dumps(obj), "application/json", cookie, extra)

    def _session_cookie(self, token, clear=False):
        # HttpOnly: page scripts cannot read it. SameSite=Lax: other sites cannot make the
        # browser send it with a POST. Secure: only over HTTPS, switched on when the host
        # reports that the request arrived that way.
        parts = [f"{COOKIE}={'' if clear else token}", "Path=/", "HttpOnly", "SameSite=Lax",
                 "Max-Age=0" if clear else f"Max-Age={auth.SESSION_SECONDS}"]
        if self._https():
            parts.append("Secure")
        return "; ".join(parts)

    def _token(self):
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        try:
            jar = SimpleCookie()
            jar.load(raw)
            m = jar.get(COOKIE)
            return m.value if m else None
        except Exception:
            return None

    def _addr(self):
        # Best effort behind a proxy. A forged X-Forwarded-For only lets someone dodge the
        # per-address limit; the per-account limit still applies to every guess.
        fwd = self.headers.get("X-Forwarded-For", "")
        return (fwd.split(",")[0].strip() if fwd else self.client_address[0])[:45]

    def _same_origin(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True       # not a browser cross-site request (curl, tests)
        try:
            return urlparse(origin).netloc.lower() == self.headers.get("Host", "").lower()
        except ValueError:
            return False

    def _body(self):
        """Return (payload, None) or (None, (status, message))."""
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            return None, (415, "Send JSON.")
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return None, (411, "Length required.")
        if n < 0 or n > MAX_BODY:
            return None, (413, "Request too large.")
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, UnicodeDecodeError):
            return None, (400, "Invalid JSON.")
        if not isinstance(payload, dict):
            return None, (400, "Invalid JSON.")
        return payload, None

    # ---- GET ----
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            return self._json(200, {"status": "ok"})
        if path in ("/", "/index.html"):
            user = STORE.user_for(self._token())
            if user:
                html = PAGE.replace("__ME__", me_json(user))
            else:
                html = (LOGIN_PAGE.replace("__SIGNUP__", "true" if ALLOW_SIGNUP else "false")
                        .replace("__TEMP__", "true" if TEMP_ACCOUNTS else "false"))
            return self._send(200, html, "text/html; charset=utf-8")
        self._send(404, "Not found", "text/plain")

    # ---- POST ----
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        routes = {"/api/analyze": self._analyze, "/api/login": self._login,
                  "/api/signup": self._signup, "/api/logout": self._logout}
        handler = routes.get(path)
        if handler is None:
            return self._send(404, "Not found", "text/plain")
        if not self._same_origin():
            return self._json(403, {"error": "Cross-site request refused."})
        payload, err = self._body()
        if err:
            return self._json(err[0], {"error": err[1]})
        try:
            handler(payload)
        except Exception:
            # Details stay out of the response: an error message can reveal how the server works.
            self._json(500, {"error": "Something went wrong on the server."})

    def _analyze(self, payload):
        if not STORE.user_for(self._token()):
            return self._json(401, {"error": "Please sign in again."})
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return self._json(400, {"error": "Enter a complaint to analyze."})
        self._json(200, analyze(text.strip()[:2000]))

    def _login(self, payload):
        result = STORE.login(auth.clean_email(payload.get("email")), payload.get("password"), self._addr())
        if result["ok"]:
            return self._json(200, {"ok": True}, cookie=self._session_cookie(result["token"]))
        extra = {"Retry-After": str(result["retry"])} if result.get("retry") else None
        self._json(result["status"], {"error": result["error"]}, extra=extra)

    def _signup(self, payload):
        if not ALLOW_SIGNUP:
            return self._json(403, {"error": "Sign-up is turned off on this server."})
        addr = self._addr()
        wait = STORE.signup_by_addr.wait_seconds(addr)
        if wait:
            return self._json(429, {"error": "Too many sign-ups from this connection. Try again later."},
                              extra={"Retry-After": str(wait)})
        STORE.signup_by_addr.add(addr)
        password = payload.get("password")
        if not isinstance(password, str):
            return self._json(400, {"error": "Enter a password."})
        uid, err = STORE.create_user(auth.clean_email(payload.get("email")), password)
        if err == "exists":
            # Saying so is the usual trade-off: the person needs to know to sign in instead.
            return self._json(409, {"error": "An account with this email already exists. Try signing in."})
        if err:
            return self._json(400, {"error": err})
        self._json(200, {"ok": True}, cookie=self._session_cookie(STORE.new_session(uid)))

    def _logout(self, payload):
        STORE.end_session(self._token())
        self._json(200, {"ok": True}, cookie=self._session_cookie("", clear=True))


def main():
    global MODEL, STORE
    path = C.MODELS / "emotion_final.joblib"
    if not path.exists():
        sys.exit("No trained model found. Run:  python src/domain_adapt.py")
    print("Loading the trained model...")
    MODEL = joblib.load(path)
    print(f"Classes: {', '.join(MODEL.classes_)}")

    data_dir = auth.pick_data_dir()
    STORE = auth.AuthStore(data_dir / "users.db")
    print(f"Accounts database: {data_dir / 'users.db'}")
    demo_email = auth.clean_email(os.environ.get("DEMO_EMAIL"))
    demo_password = os.environ.get("DEMO_PASSWORD", "")
    if demo_email and demo_password:
        uid, err = STORE.ensure_user(demo_email, demo_password)
        print("Demo account ready." if uid else f"Demo account NOT created: {err}")
    if TEMP_ACCOUNTS:
        print("Note: accounts live on temporary storage and are cleared when this server restarts.")
    url = f"http://localhost:{PORT}"
    print(f"\n  Complaint analyzer running at {url}")
    print("  Press Ctrl+C to stop.\n")
    if not ON_SERVER:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
        server.daemon_threads = True
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
