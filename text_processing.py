"""Text processing helpers shared across phases."""

from __future__ import annotations

import json
import re
from collections import deque

STOPWORDS = {
    'a','an','and','or','but','if','then','else','when','while','for','to','from','by','with','without',
    'of','in','on','at','as','is','are','was','were','be','been','being','it','its','this','that','these','those',
    'you','your','yours','we','our','us','ours','they','them','their','theirs','he','she','his','her','hers','i','me','my',
    'do','does','did','done','doing','can','could','should','would','may','might','must','will','shall',
    'not','no','yes','up','down','over','under','again','more','most','some','such','only','own','same','so','than','too','very',
    'into','out','about','above','below','between','through','during','before','after','off','under','against',
    'new','now','enter','unlock','unlocked','available','tap','click','open','close','back','next','previous','ok','okay','cancel',
    'please','error','success','failed','confirm','retry','skip','start','stop','continue','loading','load','save','saved',
    'press','hold','release','enable','disable','enabled','disabled','have','has','alright','hey','hi','hello','bye','goodbye','thanks',
    'thank','sorry','okey','yah','yeah','yep','nope','uh','uhh','hmm','huh','ah','oh','oops','briefly',
    'who','whom','whose','what','which','where','when','why','how',
    "it's","isn't","aren't","wasn't","weren't","hasn't","haven't","hadn't",
    "won't","wouldn't","can't","couldn't","shouldn't","don't","doesn't","didn't",
    "i'll","you'll","he'll","she'll","we'll","they'll",
    "i'd","you'd","he'd","she'd","we'd","they'd",
    "i'm","you're","we're","they're","he's","she's","that's","there's","here's",
    "what's","who's","how's","where's","let's","ain't",
    "i've","you've","we've","they've","gonna","wanna","gotta","kinda","sorta","lotta","lemme","gimme","y’all","c’mon",
    "'em","ma’am","’cause","cos","’til","’bout","’round","should’ve","would’ve","could’ve","might’ve","must’ve",
    'just','really','quite','even','ever','always','maybe','perhaps','still','also','yet','already','almost',
    'though','although','however','therefore','hence','thus','either','neither','each','both','every',
    'anyone','anything','everyone','everything','someone','something','none','nothing','somebody','nobody','everybody',
    'whichever','whenever','wherever','whatever',
    'i','ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii','xiii','xiv','xv'
}


def looks_like_title_variant(s: str) -> bool:
    toks = s.split()
    return any(t and (t[0].isupper() or t.isupper()) for t in toks)


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\b(Mr|Mrs|Ms|Dr|St)\.", r"\1<dot>", s)
    s = re.sub(r"(\[.*?\]|\{.*?\}|\<.*?\>)", " | ", s)
    s = re.sub(r"[()]", " | ", s)
    s = re.sub(r"\s-\s", " | ", s)
    s = re.sub(r"[,:;!?]", " | ", s)
    s = s.replace("<dot>", ".")
    s = re.sub(r"[\r\n\t]", " ", s)
    s = re.sub(r'[“”"#%&*_+=<>/\\^~\|]', " | ", s)
    s = re.sub(r"\|+", "|", s)
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize_to_segments(s: str):
    if not s:
        return []
    segments = [seg.strip() for seg in s.split("|") if seg.strip()]
    result_segments = []
    for seg in segments:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", seg)
        cur = []
        for tok in tokens:
            low = tok.lower()
            if re.fullmatch(r"\d+", tok):
                continue
            if len(tok) == 1 and re.fullmatch(r"[A-Za-z0-9]", tok):
                continue
            if low in STOPWORDS:
                if cur:
                    result_segments.append(cur)
                    cur = []
                continue
            cur.append(tok)
        if cur:
            result_segments.append(cur)
    cleaned = []
    for seg in result_segments:
        if len(seg) == 1:
            t = seg[0]
            if re.fullmatch(r"\d+", t):
                continue
            if len(t) == 1 and re.fullmatch(r"[A-Za-z0-9]", t):
                continue
        cleaned.append(seg)
    return cleaned


def starts_with_capital_first_token(s: str) -> bool:
    if not isinstance(s, str):
        return False
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", s)
    if not toks:
        return False
    first = toks[0]
    return first[0].isalpha() and first[0].isupper()


def generate_ngrams_from_segment(tokens, max_n=3):
    n_max = min(max_n, len(tokens))
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            yield tokens[i:i + n]


def _term_regex(term: str):
    esc = re.escape(str(term))
    return re.compile(rf"(?i)(?<!\w){esc}(?!\w)")


def split_segments_strict(s: str) -> list[str]:
    if not s:
        return []
    segs = re.split(r"\|\||\|", s)
    return [seg.strip() for seg in segs if seg and seg.strip()]


def find_term_contexts(term: str, texts: list[str]) -> list[str]:
    term = str(term).strip()
    if not term:
        return []
    rx = _term_regex(term)
    contexts = []
    single_word = len(term.split()) == 1
    for s in texts:
        if not s:
            continue
        segs = split_segments_strict(s)
        for seg in segs:
            seg_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", seg)
            seg_tokens_lower = [t.lower() for t in seg_tokens]
            if single_word:
                if term.lower() in seg_tokens_lower:
                    contexts.append(seg.strip())
            else:
                if rx.search(seg):
                    contexts.append(seg.strip())
    return contexts


def _diverse_order(n: int) -> list[int]:
    if n <= 0:
        return []
    order, used = [], set()
    left, right = 0, n - 1
    mid = n // 2
    for a in (0, right, mid):
        if 0 <= a < n and a not in used:
            order.append(a)
            used.add(a)
    l, r, lm, rm = 1, n - 2, mid - 1, mid + 1
    while len(order) < n:
        for cand in (l, r, lm, rm):
            if 0 <= cand < n and cand not in used:
                order.append(cand)
                used.add(cand)
        l += 1
        r -= 1
        lm -= 1
        rm += 1
        if l > r and lm < 0 and rm >= n:
            order.extend([i for i in range(n) if i not in used])
            break
    return order


def select_diverse_contexts(contexts: list[str], max_lines: int = 200, char_cap: int = 3000) -> list[str]:
    if not contexts:
        return []
    seen, uniq = set(), []
    for s in contexts:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    with_idx = list(enumerate(uniq))
    with_idx.sort(key=lambda x: len(x[1]))
    order_idx = _diverse_order(len(with_idx))
    picked, total = [], 0
    for k in order_idx:
        if len(picked) >= max_lines:
            break
        s = with_idx[k][1]
        sep = " || " if picked else ""
        if total + len(sep) + len(s) > char_cap:
            break
        picked.append(s)
        total += len(sep) + len(s)
    return picked


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)```$", re.IGNORECASE)


def _strip_code_fences(s: str) -> str:
    m = _CODE_FENCE_RE.match(s.strip())
    return m.group(1).strip() if m else s


def extract_json_array(s: str):
    if not s or not s.strip():
        raise ValueError("Empty content")
    s = _strip_code_fences(s).strip()
    if s.startswith("[") and s.endswith("]"):
        return json.loads(s)
    first, last = s.find("["), s.rfind("]")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON array bounds found")
    candidate = s[first:last + 1].strip()
    candidate = re.sub(r",\s*([\]\}])", r"\1", candidate).replace("\ufeff", "").strip()
    return json.loads(candidate)


def extract_json_array_of_objects(s: str):
    arr = extract_json_array(s)
    if not isinstance(arr, list) or (arr and not isinstance(arr[0], dict)):
        raise ValueError("Expected list of objects")
    return arr


def has_exact_segment(term: str, texts: list[str]) -> bool:
    t = (term or "").strip().lower()
    if not t:
        return False
    for raw in texts:
        if not raw:
            continue
        for seg in split_segments_strict(str(raw)):
            if seg.strip().lower() == t:
                return True
    return False


def build_context_string(term: str, texts: list[str], max_lines: int = 30, char_cap: int = 1200) -> str:
    ctxs = find_term_contexts(term, texts)
    picked = select_diverse_contexts(ctxs, max_lines=max_lines, char_cap=char_cap)
    return " || ".join(picked)


def strip_segment_noise(seg: str) -> str:
    if not seg:
        return ""
    s = str(seg)
    s = re.sub(r"\[(?:\/?C|-)\]", " ", s)
    s = re.sub(r"\[\#?[0-9A-Fa-f]{3,8}\]", " ", s)
    s = re.sub(r"\[[0-9A-Za-z]{3,10}\]", " ", s)
    s = re.sub(r"[\+\-\*\=_~^]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def has_segment_or_noisy_equivalent(term: str, texts: list[str]) -> bool:
    t = (term or "").strip().lower()
    if not t:
        return False
    for raw in texts or []:
        if not raw:
            continue
        for seg in split_segments_strict(str(raw)):
            s = seg.strip()
            if s.lower() == t:
                return True
            if strip_segment_noise(s).lower() == t:
                return True
    return False


def build_phase1_segment_set(texts: list[str]) -> set[str]:
    segset = set()
    for raw in texts or []:
        s = clean_text(raw)
        for seg in s.split("|"):
            seg = seg.strip()
            if seg:
                segset.add(seg.lower())
    return segset


def has_phase1_clean_segment(term: str, segset: set[str]) -> bool:
    t = (term or "").strip().lower()
    return bool(t) and (t in segset)

