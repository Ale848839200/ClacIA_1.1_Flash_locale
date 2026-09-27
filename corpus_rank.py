"""Ranking multi-corpus per ClacIA 1.1 Flash."""
from __future__ import annotations
import re, os, math
from collections import Counter, defaultdict

_STOP = set(
    "che cos cosa come una un uno il la lo i gli le di da in su per con a e o si se non "
    "del della dei delle mi ti fa sono serve esempio quale quando".split()
)
_SIN = {
    "liste": "lista", "tuple": "tupla", "dict": "dizionario",
    "aggiungo": "append", "aggiungere": "append", "tolgo": "strip", "spazi": "strip",
    "immagini": "img", "link": "a",
}
_W = {
    "append": 4, "strip": 4.5, "split": 4, "join": 4, "return": 4, "lambda": 4,
    "nameerror": 4.5, "typeerror": 4.5, "lista": 2.5, "tupla": 2.5, "dizionario": 2.5,
    "html": 3.5, "href": 4.5, "alt": 4.5, "img": 3.5, "form": 3.5, "ul": 4, "li": 3.5,
    "python": 1.5, "def": 3.5, "range": 3, "import": 3, "math": 3, "print": 2.5,
    "clacia": 4, "chatgpt": 4, "mutabile": 3, "immutabile": 3, "while": 3, "open": 3,
}

def norm(t: str) -> str:
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^\w\s=<>/.*+-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def tok(t: str) -> set:
    out = set()
    for w in norm(t).split():
        if len(w) > 1 and w not in _STOP:
            out.add(w)
            if w in _SIN:
                out.add(_SIN[w])
    return out

class RankedCorpus:
    def __init__(self, directory: str):
        self.entries = []
        self.inv = defaultdict(list)
        self.idf = {}
        for fn, dom in [
            ("corpus_python_it.txt", "python"),
            ("corpus_html_it.txt", "html"),
            ("corpus_italiano_espositivo.txt", "italiano"),
        ]:
            path = os.path.join(directory, fn)
            if not os.path.exists(path):
                print(f"  [rank] MANCANTE: {fn}")
                continue
            n = 0
            text = open(path, encoding="utf-8").read()
            for b in re.split(r"\n\s*\n", text.strip()):
                mi = re.search(r"Istruzione:\s*(.+?)(?=\nRisposta:|$)", b, re.S | re.I)
                mr = re.search(r"Risposta:\s*(.+)$", b, re.S | re.I)
                if not mi or not mr:
                    continue
                i, r = mi.group(1).strip(), mr.group(1).strip()
                ti, tr = tok(i), tok(r)
                for m in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", r):
                    tr.add(m.lower())
                for m in re.findall(r"</?([a-z0-9]+)", r, re.I):
                    tr.add(m.lower())
                tset = ti | tr
                idx = len(self.entries)
                self.entries.append(
                    {"i": i, "r": r, "ti": ti, "t": tset, "dom": dom, "i_norm": norm(i)}
                )
                for term in tset:
                    self.inv[term].append(idx)
                n += 1
            print(f"  [rank] {fn}: {n} ({dom})")
        print(f"  [rank] totale: {len(self.entries)}")
        df = Counter()
        for e in self.entries:
            for t in e["t"]:
                df[t] += 1
        N = max(1, len(self.entries))
        self.idf = {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}

    def search(self, q: str, soglia: float = 0.35):
        qt = tok(q)
        for t in list(qt):
            if t in _SIN:
                qt.add(_SIN[t])
        if not qt:
            return None, 0.0
        cand = set()
        for t in qt:
            for idx in self.inv.get(t, [])[:1000]:
                cand.add(idx)
        if not cand:
            return None, 0.0
        qn = norm(q)
        ranked = []
        for idx in cand:
            e = self.entries[idx]
            must = {t for t in qt if t in _W and _W[t] >= 3.5}
            if must and not any(m in e["t"] or m in e["i_norm"] for m in must):
                continue
            wq = sum(_W.get(t, 1) * self.idf.get(t, 1) for t in qt) or 1
            cov = sum(_W.get(t, 1) * self.idf.get(t, 1) for t in qt if t in e["t"]) / wq
            covi = sum(_W.get(t, 1) * self.idf.get(t, 1) for t in (qt & e["ti"])) / wq
            ph = 0.5 if qn == e["i_norm"] else (0.35 if qn in e["i_norm"] or e["i_norm"] in qn else 0)
            s = 0.35 * cov + 0.4 * covi + ph
            if e["dom"] == "python" and any(x in qt for x in ("lista", "tupla", "append", "python")):
                s += 0.05
            ranked.append((s, e))
        if not ranked:
            return None, 0.0
        ranked.sort(key=lambda x: x[0], reverse=True)
        if ranked[0][0] < soglia:
            return None, ranked[0][0]
        return ranked[0][1]["r"], ranked[0][0]

    def num(self) -> int:
        return len(self.entries)
