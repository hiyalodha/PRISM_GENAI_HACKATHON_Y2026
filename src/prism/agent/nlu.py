from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CITIES = sorted(
    {
        "new york", "los angeles", "san francisco", "las vegas", "hong kong", "rio de janeiro",
        "mexico city", "new delhi", "buenos aires", "cape town", "sao paulo", "kuala lumpur",
        "tel aviv", "paris", "rome", "london", "berlin", "madrid", "tokyo", "seoul", "sydney",
        "chicago", "boston", "seattle", "miami", "dallas", "denver", "toronto", "vancouver",
        "dubai", "singapore", "mumbai", "delhi", "bangalore", "amsterdam", "barcelona", "lisbon",
        "vienna", "prague", "dublin", "zurich", "munich", "milan", "istanbul", "bangkok",
        "beijing", "shanghai", "atlanta", "houston", "austin", "montreal", "osaka", "athens",
        "cairo", "nairobi", "oslo", "stockholm", "copenhagen", "helsinki", "warsaw", "budapest",
        "brussels", "frankfurt", "lima", "bogota", "santiago", "manila", "jakarta", "hanoi",
        "taipei", "doha", "riyadh", "auckland", "melbourne", "perth", "portland", "phoenix",
        "orlando", "detroit", "philadelphia", "washington", "honolulu", "venice", "florence",
        "edinburgh", "manchester", "geneva", "seoul", "chennai", "hyderabad", "kolkata", "pune",
    },
    key=len,
    reverse=True,
)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = [
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
]
MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4}
CORRECTION_RE = re.compile(
    r"\b(actually|instead|wait|sorry|change it|change that|make it|make that|switch|rather|"
    r"let'?s do|go with|correction|i meant|i mean|scratch that|not)\b|^\s*no\b",
    re.IGNORECASE,
)
BACKCHANNEL_RE = re.compile(r"^\s*(uh[- ]?huh|mm+[- ]?hm+|mhm|ok(ay)?|right|yeah|yep|sure|got it)[.!]?\s*$", re.IGNORECASE)
AFFIRM_RE = re.compile(r"^\s*(yes|yeah|yep|sure|correct|go ahead|please do|do it|confirm|sounds good)\b", re.IGNORECASE)
DEICTIC_RE = re.compile(r"\b(this|that|these|here|see|look|picture|photo|camera|screen|showing|shown)\b", re.IGNORECASE)
MODEL_RE = re.compile(r"\b([A-Z]{1,4}-?\d{2,5}[A-Z]?)\b")
ID_RE = re.compile(r"\b([A-Z]{2,4}-?\d{2,6})\b")
TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
NAME_RE = re.compile(
    r"(?i:\bmy name is|\bname is|\bname's|\bname should be|\bunder the name|\bpassenger name is|"
    r"\bpassenger is|\bfor passenger|\bbook it for|\bbooking for|\bthe passenger)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)
FOR_NAME_RE = re.compile(r"(?i:\bfor)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b")
STOPWORDS = {
    "a", "an", "the", "to", "for", "of", "on", "in", "at", "and", "or", "is", "are", "be",
    "me", "my", "i", "you", "it", "this", "that", "can", "could", "please", "with", "by",
    "from", "given", "specific", "about", "some", "any", "want", "would", "like", "need",
    "do", "does", "what", "which", "get", "into", "as", "up", "should", "will",
}
SYNONYMS = {
    "find": "search", "look": "search", "show": "search", "available": "search",
    "options": "search", "flights": "flight", "fly": "flight", "flying": "flight",
    "reserve": "book", "reservation": "book", "buy": "book", "purchase": "book",
    "seat": "book", "complaint": "ticket", "report": "ticket", "support": "ticket",
    "broken": "ticket", "refund": "ticket", "lookup": "manual", "manuals": "manual",
    "blinking": "light", "led": "light", "lights": "light", "flashing": "light",
    "tables": "table", "restaurant": "table", "dinner": "table", "lunch": "table",
    "people": "party", "guests": "party", "weather": "forecast",
}
ACTION_VERBS = {"book", "create", "file", "submit", "reserve", "cancel", "order", "schedule", "send", "open"}
QUERY_VERBS = {"search", "check", "what", "which", "any", "how", "why", "list", "tell"}


def title(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())


def tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower().replace("_", " "))
    out = []
    for t in raw:
        t = SYNONYMS.get(t, t)
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = SYNONYMS.get(t[:-1], t[:-1])
        out.append(t)
    return out


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS}


@dataclass
class Mention:
    value: str
    start: int
    role: str | None = None
    negated: bool = False


def _preceding_words(text: str, start: int, n: int = 3) -> list[str]:
    return re.findall(r"[a-z']+", text[:start].lower())[-n:]


def _negated(lower: str, start: int) -> bool:
    return bool(re.search(r"\b(not|no|instead of)\s+(?:the\s+)?$", lower[:start]))


def find_cities(text: str) -> list[Mention]:
    lower = text.lower()
    taken: list[tuple[int, int]] = []
    found: list[Mention] = []
    for city in CITIES:
        for m in re.finditer(rf"\b{re.escape(city)}\b", lower):
            if any(a <= m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            prev = _preceding_words(lower, m.start())
            role = None
            if prev and prev[-1] in {"to", "into", "towards"}:
                role = "destination"
            elif prev and (prev[-1] in {"from", "leaving", "departing"} or prev[-2:] == ["out", "of"]):
                role = "origin"
            elif prev and prev[-1] in {"in", "at", "near"}:
                role = "location"
            negated = _negated(lower, m.start())
            found.append(Mention(title(city), m.start(), role, negated))
    found.sort(key=lambda x: x.start)
    return found


def find_dates(text: str) -> list[Mention]:
    lower = text.lower()
    out: list[Mention] = []
    for m in re.finditer(r"\b(day after tomorrow|today|tonight|tomorrow)\b", lower):
        out.append(Mention(m.group(1), m.start()))
    for m in re.finditer(r"\b(" + "|".join(WEEKDAYS) + r")\b", lower):
        out.append(Mention(m.group(1).capitalize(), m.start()))
    for m in re.finditer(MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b", lower):
        out.append(Mention(f"{_month(m.group(1))} {int(m.group(2))}", m.start()))
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + MONTH_RE + r"\b", lower):
        out.append(Mention(f"{_month(m.group(2))} {int(m.group(1))}", m.start()))
    for m in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", lower):
        out.append(Mention(m.group(1), m.start()))
    for d in out:
        d.negated = _negated(lower, d.start)
    out.sort(key=lambda x: x.start)
    return out


def _month(prefix: str) -> str:
    for name in MONTHS:
        if name.startswith(prefix[:3]):
            return name.capitalize()
    return prefix.capitalize()


def last_positive(mentions: list[Mention], role: str | None = None) -> Mention | None:
    pool = [m for m in mentions if not m.negated and (role is None or m.role == role)]
    return pool[-1] if pool else None


def find_name(text: str) -> str | None:
    matches = NAME_RE.findall(text)
    if not matches:
        matches = [m for m in FOR_NAME_RE.findall(text) if m.lower() not in CITIES]
    if not matches:
        return None
    name = matches[-1].strip()
    words = [w for w in name.split() if w.lower() not in WEEKDAYS and w.lower() not in STOPWORDS]
    return " ".join(words) or None


def find_time(text: str) -> str | None:
    matches = list(TIME_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    minutes = m.group(2) or "00"
    return f"{int(m.group(1))}:{minutes} {m.group(3).lower()}"


def find_numbers(text: str) -> list[tuple[int, int]]:
    out = []
    for m in re.finditer(r"\b(\d+)\b", text):
        after = text[m.end(): m.end() + 3]
        if re.match(r"\s*(am|pm|:)", after, re.IGNORECASE):
            continue
        out.append((int(m.group(1)), m.start()))
    for m in re.finditer(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", text.lower()):
        out.append((NUMBER_WORDS[m.group(1)], m.start()))
    out.sort(key=lambda x: x[1])
    return out


def find_quantity(text: str, nouns: tuple[str, ...] = ("people", "guests", "persons", "passengers", "seats", "tickets", "of us")) -> int | None:
    lower = text.lower()
    for value, start in reversed(find_numbers(lower)):
        tail = lower[start: start + 20]
        if any(n in tail for n in nouns):
            return value
    nums = find_numbers(lower)
    return nums[-1][0] if nums else None


def find_model(text: str) -> str | None:
    matches = MODEL_RE.findall(text)
    return matches[-1] if matches else None


def find_selection(text: str) -> tuple[str, int] | None:
    lower = text.lower()
    m = re.search(r"\b(?:option|number|choice|flight)\s+(\d)\b", lower)
    if m:
        return ("index", int(m.group(1)) - 1)
    for word in ("cheapest", "earliest", "latest", "last"):
        if re.search(rf"\b{word}\b", lower):
            return (word, 0)
    for word, idx in ORDINALS.items():
        if re.search(rf"\b{word}\b", lower):
            return ("index", idx)
    return None


def has_correction(text: str) -> bool:
    return bool(CORRECTION_RE.search(text))


def is_backchannel(text: str) -> bool:
    return bool(BACKCHANNEL_RE.match(text))


def is_affirmation(text: str) -> bool:
    return bool(AFFIRM_RE.match(text))


def refers_to_visual(text: str) -> bool:
    return bool(DEICTIC_RE.search(text))


def humanize(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower().replace("_", " ")))


@dataclass
class ToolGuess:
    name: str
    score: float
    state_changing: bool


def rank_tools(text: str, specs: list[Any]) -> list[ToolGuess]:
    words = content_tokens(text)
    if not words:
        return []
    acting = bool(words & ACTION_VERBS)
    ranked: list[ToolGuess] = []
    for spec in specs:
        name_toks = content_tokens(spec.name)
        desc_toks = content_tokens(spec.description)
        score = 2.0 * len(words & name_toks) + 1.0 * len(words & (desc_toks - name_toks))
        if score <= 0:
            continue
        if acting and spec.state_changing:
            score += 1.5
        if not acting and not spec.state_changing:
            score += 0.5
        ranked.append(ToolGuess(spec.name, score, spec.state_changing))
    ranked.sort(key=lambda g: (-g.score, g.name))
    return ranked


def guess_tool(text: str, specs: list[Any], min_score: float = 2.0) -> ToolGuess | None:
    ranked = rank_tools(text, specs)
    if ranked and ranked[0].score >= min_score:
        return ranked[0]
    return None


@dataclass
class SlotHints:
    values: dict[str, Any] = field(default_factory=dict)
    corrected: set[str] = field(default_factory=set)


PARTY_WORDS = ("party", "size", "people", "guests", "passengers", "count", "number", "quantity", "qty", "seats", "tickets")
FREE_TEXT_WORDS = ("issue", "description", "summary", "query", "question", "problem", "message", "text", "details", "subject", "symptom")


def param_kind(name: str, schema: dict[str, Any]) -> str:
    n = name.lower()
    typ = schema.get("type")
    if "enum" in schema:
        return "enum"
    if n == "id" or n.endswith("_id"):
        return "identifier"
    if n in {"origin", "source", "from", "from_city", "departure_city"} or n.startswith("origin"):
        return "origin"
    if n in {"destination", "to", "to_city", "arrival_city"} or n.startswith("dest"):
        return "destination"
    if "city" in n or "location" in n or "place" == n:
        return "location"
    if "date" in n or n in {"day", "when"}:
        return "date"
    if "time" in n:
        return "time"
    if typ in {"integer", "number"} or any(w in n for w in PARTY_WORDS):
        return "number"
    if typ == "boolean":
        return "boolean"
    if "passenger" in n or n in {"name", "full_name", "customer_name", "traveler", "traveller"} or n.endswith("_name") and "restaurant" not in n:
        return "person"
    if "model" in n or "device" in n or "product" in n or "serial" in n:
        return "model"
    if any(w in n for w in FREE_TEXT_WORDS):
        return "free_text"
    return "proper_noun"


def _proper_noun(text: str, exclude: set[str]) -> str | None:
    for m in re.finditer(r"\b(?:at|called|named|for|with)\s+((?:[A-Z][\w'&-]*)(?:\s+[A-Z][\w'&-]*)*)", text):
        cand = m.group(1).strip()
        if cand.lower() in exclude or cand.lower() in WEEKDAYS or cand.lower() in CITIES:
            continue
        return cand
    return None


def extract_param(name: str, schema: dict[str, Any], text: str, extra: dict[str, Any] | None = None) -> Any:
    kind = param_kind(name, schema)
    extra = extra or {}
    if kind == "enum":
        lower = text.lower()
        for option in schema.get("enum", []):
            if isinstance(option, str) and re.search(rf"\b{re.escape(option.lower())}\b", lower):
                return option
        return None
    if kind in {"origin", "destination"}:
        cities = find_cities(text)
        hit = last_positive(cities, kind)
        return hit.value if hit else None
    if kind == "location":
        hit = last_positive(find_cities(text))
        return hit.value if hit else None
    if kind == "date":
        hit = last_positive(find_dates(text))
        return hit.value if hit else None
    if kind == "time":
        return find_time(text)
    if kind == "number":
        return find_quantity(text)
    if kind == "boolean":
        return True if is_affirmation(text) else None
    if kind == "person":
        return find_name(text)
    if kind == "model":
        model = find_model(text)
        if model:
            return model
        for key, value in extra.items():
            if key == name or "model" in key or "device" in key:
                return value
        return None
    if kind == "identifier":
        ids = ID_RE.findall(text)
        return ids[-1] if ids else None
    if kind == "free_text":
        cleaned = text.strip()
        return cleaned or None
    return _proper_noun(text, set())
