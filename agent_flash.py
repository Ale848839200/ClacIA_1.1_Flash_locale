"""
ClacIA 1.1 Flash — uso in locale
Avvio: python3 agent_flash.py
"""
from __future__ import annotations
import re, ast, operator, os, json, urllib.request, urllib.parse
import torch
from tokenizers import ByteLevelBPETokenizer
from model import MiniGPT
from corpus_rank import RankedCorpus

VERSIONE = "1.1 Flash"
BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE, "clacia_data")

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}

def _val(n):
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.BinOp):
        return _OPS[type(n.op)](_val(n.left), _val(n.right))
    if isinstance(n, ast.UnaryOp):
        return _OPS[type(n.op)](_val(n.operand))
    raise ValueError("espressione non supportata")

def calcola(expr: str):
    try:
        r = _val(ast.parse(expr, mode="eval").body)
        return int(r) if isinstance(r, float) and r.is_integer() else r
    except Exception as e:
        return f"Errore: {e}"

def estrai_math(d: str):
    low = d.lower()
    n = re.findall(r"\d+(?:\.\d+)?", d)
    if ("percento" in low or "per cento" in low) and len(n) >= 2:
        return f"{n[1]} * {n[0]} / 100"
    if "quadrato" in low and n:
        return f"{n[0]} ** 2"
    if "radice" in low and n:
        return f"{n[0]} ** 0.5"
    if "elevat" in low and len(n) >= 2:
        return f"{n[0]} ** {n[1]}"
    # somma multipla tipo 33+66+1
    if re.fullmatch(r"[\d\s\+\-\*/\(\)\.]+", d.strip()) and any(c in d for c in "+-*/"):
        try:
            ast.parse(d.strip(), mode="eval")
            return d.strip()
        except Exception:
            pass
    m = re.search(r"[\d\s\+\-\*/\(\)\.]{5,}", d)
    if m and any(c in m.group() for c in "+-*/"):
        c = m.group().strip()
        try:
            ast.parse(c, mode="eval")
            if re.search(r"\d", c):
                return c
        except Exception:
            pass
    if len(n) < 2:
        return None
    a, b = n[0], n[1]
    if any(p in low for p in ["diviso", "dividi"]):
        return f"{a} / {b}"
    if any(p in low for p in ["meno", "sottra"]):
        return f"{a} - {b}"
    if any(p in low for p in ["piu", "più", "somma"]):
        return f"{a} + {b}"
    if "per" in low or "moltiplic" in low:
        return f"{a} * {b}"
    if "+" in d:
        return f"{a} + {b}"
    return f"{a} + {b}"

def is_calc(d: str) -> bool:
    low = d.lower()
    nums = re.findall(r"\d+", d)
    if len(nums) >= 1 and any(w in low for w in ["quadrato", "radice", "percento", "elevat"]):
        return True
    if len(nums) >= 2 and any(
        w in low
        for w in ["meno", "piu", "più", "per", "diviso", "dividi", "somma", "+", "-", "*", "/", "quanto", "calcola"]
    ):
        return True
    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", d):
        return True
    return False

def is_pratica(d: str) -> bool:
    low = d.lower()
    keys = [
        "programmare", "programma", "esercizio", "codice", "aiutami", "media", "spesa",
        "pari", "while", "compito", "pratica", "scrivi un", "come faccio", "lista della",
        "finche", "finché", "voglio fare", "mi aiuti", "implementa",
    ]
    return any(k in low for k in keys)

def genera_codice(d: str) -> str:
    low = d.lower()
    if "media" in low:
        return (
            "numeri = []\n"
            "while True:\n"
            "    s = input('Numero (invio per fine): ').strip()\n"
            "    if s == '':\n"
            "        break\n"
            "    numeri.append(float(s))\n"
            "if not numeri:\n"
            "    print('Nessun numero')\n"
            "else:\n"
            "    print('Media:', sum(numeri) / len(numeri))"
        )
    if "spesa" in low:
        return (
            "spesa = []\n"
            "while True:\n"
            "    voce = input(\"Aggiungi (o 'fine'): \").strip()\n"
            "    if voce.lower() == 'fine':\n"
            "        break\n"
            "    if voce:\n"
            "        spesa.append(voce)\n"
            "for i, v in enumerate(spesa, 1):\n"
            "    print(i, v)"
        )
    if "pari" in low:
        return "for n in range(1, 21):\n    if n % 2 == 0:\n        print(n)"
    if "while" in low:
        return "n = 3\nwhile n > 0:\n    print(n)\n    n = n - 1\nprint('fine')"
    if any(w in low for w in ["esci", "finche", "finché"]):
        return (
            "while True:\n"
            "    msg = input('Scrivi (esci per uscire): ').strip()\n"
            "    if msg.lower() == 'esci':\n"
            "        break\n"
            "    print(msg)"
        )
    return "n = int(input('Numero: '))\nprint('Doppio:', n * 2)"

def ripasso_e_codice(d: str) -> str:
    return (
        "**1) Ripasso teoria**\n"
        "- Usa solo variabili/if/cicli/liste che servono al risultato.\n"
        "- input() restituisce sempre una stringa.\n\n"
        "**2) Codice di esempio**\n```python\n"
        + genera_codice(d)
        + "\n```\n\n"
        "**3) Prossimo passo**\nCopia, esegui, modifica un valore e riprova."
    )

def cerca_wikipedia(soggetto: str) -> str:
    try:
        url = f"https://it.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(soggetto)}"
        req = urllib.request.Request(url, headers={"User-Agent": "ClacIA-Flash/1.1"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read().decode()).get("extract") or f"Nessun risultato per '{soggetto}'."
    except Exception as e:
        return f"Errore Wikipedia: {e}"

def estrai_soggetto(d: str) -> str:
    p = d.strip().rstrip("?").strip()
    low = p.lower()
    if "capitale" in low and "francia" in low:
        return "Parigi"
    for pat in [r"^chi (ha|è|era|fu)\s+", r"^cos'?è\s+(un|una|il|la)\s+", r"^cos'?è\s+", r"^che cos'?è\s+"]:
        n = re.sub(pat, "", p, flags=re.I)
        if n != p:
            return re.sub(r"^(l'|il|la|lo|un|una)\s*", "", n, flags=re.I).strip() or p
    return p

_TR = [
    "python", "lista", "tupla", "append", "strip", "return", "lambda", "nameerror", "typeerror",
    "html", "href", "alt", "form", "def", "import", "math", "mutabile", "range", "print",
    "true", "false", "variabile", "funzione", "ul", "li", "len", "get", "break", "join", "split",
    "open", "file", "none", "clacia", "chatgpt", "head", "body", "doctype", "nav", "footer",
]

class AgenteFlash:
    def __init__(self, search_fn=None):
        vocab = os.path.join(BASE, "wiki_bpe_it14-vocab.json")
        merges = os.path.join(BASE, "wiki_bpe_it14-merges.txt")
        ckpt = os.path.join(BASE, "minigpt_it14_best.pt")
        self.tok = ByteLevelBPETokenizer(vocab, merges)
        ck = torch.load(ckpt, map_location="cpu")
        self.model = MiniGPT(**ck["config"])
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()
        self.search_fn = search_fn or cerca_wikipedia
        self.corpus = RankedCorpus(CORPUS_DIR)

    def _cls(self, d: str) -> str:
        low = d.lower().strip()
        if any(x in low for x in ["gestionale completo", "sito completo", "app completa"]):
            return "refuse"
        if is_pratica(d):
            return "pratica"
        if any(low == s or low.startswith(s) for s in ["ciao", "salve", "buongiorno", "grazie", "come stai"]):
            return "chat"
        if is_calc(d):
            return "calc"
        if any(k in low for k in _TR) or ("differenza" in low and ("lista" in low or "tupla" in low)):
            return "corpus"
        if any(x in low for x in ["cos'è", "che cos'è", "chi era", "chi ha", "capitale", "scoperto"]):
            return "search"
        return "direct"

    def rispondi(self, d: str, verbose: bool = False) -> str:
        it = self._cls(d)
        if it == "refuse":
            return "Non genero progetti enormi interi. Posso teoria + esempi Python a pezzi."
        if it == "pratica":
            return ripasso_e_codice(d)
        if it == "chat":
            if "grazie" in d.lower():
                return "Di nulla!"
            if "come stai" in d.lower():
                return "Sto bene! Sono ClacIA 1.1 Flash."
            return "Ciao! Sono ClacIA 1.1 Flash."
        if it == "calc":
            e = estrai_math(d)
            return "Calcolo non riconosciuto." if not e else f"{e.replace('**', '^')} = {calcola(e)}"
        if it == "corpus":
            r, s = self.corpus.search(d)
            return r if r else "Non trovo questa nozione nei corpus. Prova a riformulare."
        if it == "search":
            return self.search_fn(estrai_soggetto(d))
        r, s = self.corpus.search(d)
        if r and s >= 0.45:
            return r
        return "Chiedimi Python, HTML, pratica di programmazione, un calcolo o un fatto."

if __name__ == "__main__":
    print("Caricamento ClacIA 1.1 Flash...")
    agent = AgenteFlash()
    print(f"=== ClacIA {VERSIONE} | corpus: {agent.corpus.num()} conoscenze ===")
    print("Scrivi una domanda (o 'esci').\n")
    while True:
        try:
            q = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("esci", "exit", "quit"):
            break
        if q:
            print(f"ClacIA: {agent.rispondi(q)}\n")
