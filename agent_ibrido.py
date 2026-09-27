"""
Agente ibrido: usa il MiniGPT per CLASSIFICARE l'intento della domanda
(calcolo / ricerca / risposta diretta), ma estrae i dettagli esatti
(numeri, operazione, soggetto) direttamente dalla domanda originale con
espressioni regolari, invece di fidarsi di quello che il modello genera
dentro la chiamata.

Questo aggira il limite dimostrato sperimentalmente: un modello piccolo
riconosce bene IL TIPO di richiesta, ma non è affidabile nel COPIARE con
precisione numeri o nomi lunghi. Estrarli con codice deterministico
risolve il problema alla radice.
"""
import re
import ast
import operator
import urllib.request
import urllib.parse
import json
import torch
from tokenizers import ByteLevelBPETokenizer
from model import MiniGPT

# ---- Calcolatrice sicura ----
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _valuta_nodo(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_valuta_nodo(node.left), _valuta_nodo(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_valuta_nodo(node.operand))
    raise ValueError("Espressione non supportata")


def calcola_sicuro(espressione: str):
    try:
        node = ast.parse(espressione, mode="eval").body
        risultato = _valuta_nodo(node)
        if isinstance(risultato, float) and risultato.is_integer():
            risultato = int(risultato)
        return risultato
    except Exception as e:
        return f"Errore: {e}"


# ---- Conversioni di unità (deterministiche, formule esatte) ----
_CONVERSIONI = {
    ("km", "miglia"): lambda x: x * 0.621371,
    ("miglia", "km"): lambda x: x / 0.621371,
    ("kg", "libbre"): lambda x: x * 2.20462,
    ("libbre", "kg"): lambda x: x / 2.20462,
    ("kg", "lb"): lambda x: x * 2.20462,
    ("lb", "kg"): lambda x: x / 2.20462,
    ("m", "piedi"): lambda x: x * 3.28084,
    ("piedi", "m"): lambda x: x / 3.28084,
    ("cm", "pollici"): lambda x: x / 2.54,
    ("pollici", "cm"): lambda x: x * 2.54,
    ("litri", "galloni"): lambda x: x * 0.264172,
    ("galloni", "litri"): lambda x: x / 0.264172,
}


def celsius_a_fahrenheit(c):
    return c * 9 / 5 + 32


def fahrenheit_a_celsius(f):
    return (f - 32) * 5 / 9


def estrai_conversione(domanda: str):
    """
    Riconosce richieste di conversione di unità e restituisce
    (valore, unità_da, unità_a) oppure None se non è una conversione.
    """
    domanda_norm = domanda.lower()
    numeri = re.findall(r"\d+(?:\.\d+)?", domanda)
    if not numeri:
        return None
    valore = float(numeri[0])

    # Temperature: caso speciale, formula additiva non moltiplicativa
    if ("celsius" in domanda_norm or "gradi c" in domanda_norm or "°c" in domanda_norm) and \
       ("fahrenheit" in domanda_norm or "gradi f" in domanda_norm or "°f" in domanda_norm):
        if domanda_norm.index("celsius" if "celsius" in domanda_norm else "°c" if "°c" in domanda_norm else "gradi c") < \
           domanda_norm.index("fahrenheit" if "fahrenheit" in domanda_norm else "°f" if "°f" in domanda_norm else "gradi f"):
            return ("temp_c_to_f", valore)
        else:
            return ("temp_f_to_c", valore)

    unita_map = {
        "km": ["km", "chilometri", "chilometro"],
        "miglia": ["miglia", "miglio"],
        "kg": ["kg", "chili", "chilogrammi", "chilo"],
        "libbre": ["libbre", "libbra", "lb"],
        "m": ["metri", "metro"],
        "piedi": ["piedi", "piede", "feet"],
        "cm": ["cm", "centimetri", "centimetro"],
        "pollici": ["pollici", "pollice", "inch"],
        "litri": ["litri", "litro"],
        "galloni": ["galloni", "gallone"],
    }

    # Troviamo la posizione di ciascuna unità menzionata nel testo, per sapere
    # quale unità è effettivamente adiacente al numero (quella di origine)
    posizioni_unita = []
    for chiave, varianti in unita_map.items():
        for v in varianti:
            pos = domanda_norm.find(v)
            if pos != -1:
                posizioni_unita.append((pos, chiave))
                break
    posizioni_unita.sort()
    unita_trovate = [chiave for _, chiave in posizioni_unita]
    # rimuoviamo eventuali duplicati mantenendo la prima occorrenza
    visti = set()
    unita_uniche = []
    for u in unita_trovate:
        if u not in visti:
            visti.add(u)
            unita_uniche.append(u)
    unita_trovate = unita_uniche

    if len(unita_trovate) < 2:
        return None

    # Troviamo la posizione del numero nel testo, per capire a quale delle
    # due unità è effettivamente adiacente (quella è l'unità di origine)
    match_numero = re.search(r"\d+(?:\.\d+)?", domanda)
    pos_numero = match_numero.start() if match_numero else -1

    pos_unita_0 = domanda_norm.find(next(v for v in unita_map[unita_trovate[0]] if v in domanda_norm))
    pos_unita_1 = domanda_norm.find(next(v for v in unita_map[unita_trovate[1]] if v in domanda_norm))

    # L'unità di origine è quella più vicina al numero (di solito appena dopo)
    if abs(pos_unita_1 - pos_numero) < abs(pos_unita_0 - pos_numero):
        unita_da, unita_a = unita_trovate[1], unita_trovate[0]
    else:
        unita_da, unita_a = unita_trovate[0], unita_trovate[1]

    return (unita_da, valore, unita_a)


# ---- Estrazione precisa dei dettagli dalla domanda originale (non dal modello) ----

def estrai_espressione_matematica(domanda: str):
    """
    Cerca numeri e operatori nella domanda originale e costruisce
    un'espressione Python valida, con precisione al 100% (regex, non AI).
    """
    domanda_norm = domanda.lower()

    if "radice quadrata" in domanda_norm or "radice di" in domanda_norm:
        numeri = re.findall(r"\d+(?:\.\d+)?", domanda)
        if numeri:
            return f"{numeri[0]} ** 0.5"

    if "percento di" in domanda_norm or "per cento di" in domanda_norm:
        numeri = re.findall(r"\d+(?:\.\d+)?", domanda)
        if len(numeri) >= 2:
            percentuale, valore = numeri[0], numeri[1]
            return f"{valore} * {percentuale} / 100"

    if "cubo" in domanda_norm:
        numeri = re.findall(r"\d+", domanda)
        if numeri:
            return f"{numeri[0]} ** 3"

    if any(p in domanda_norm for p in ["quadrato", "al quadrato"]):
        numeri = re.findall(r"\d+", domanda)
        if numeri:
            return f"{numeri[0]} ** 2"

    # "X alla decima/quinta/ennesima" ecc. -> potenza
    parole_numeri_ordinali = {
        "seconda": 2, "terza": 3, "quarta": 4, "quinta": 5, "sesta": 6,
        "settima": 7, "ottava": 8, "nona": 9, "decima": 10,
    }
    if "alla" in domanda_norm or "elevat" in domanda_norm:
        numeri = re.findall(r"\d+", domanda)
        for parola, valore in parole_numeri_ordinali.items():
            if parola in domanda_norm and numeri:
                return f"{numeri[0]} ** {valore}"
        if len(numeri) >= 2:
            return f"{numeri[0]} ** {numeri[1]}"

    # Espressione con PIÙ operatori (es. "(5+3)*2", "100-20+5"): se la domanda
    # contiene già simboli matematici scritti esplicitamente, li usiamo così
    # come sono, rispettando l'ordine reale invece di prendere solo i primi due numeri
    espressione_esplicita = re.search(r"[\d\s\+\-\*/\(\)\.]{5,}", domanda)
    if espressione_esplicita and any(op in espressione_esplicita.group() for op in "+-*/"):
        candidato = espressione_esplicita.group().strip()
        # verifichiamo che sia un'espressione valida prima di usarla
        try:
            ast.parse(candidato, mode="eval")
            # deve contenere almeno un numero, altrimenti è probabilmente rumore
            if re.search(r"\d", candidato):
                return candidato
        except Exception:
            pass

    numeri = re.findall(r"\d+(?:\.\d+)?", domanda)
    if len(numeri) < 2:
        return None

    # Gestione di PIÙ numeri con operatori a parole (es. "100 meno 20 più 5")
    if len(numeri) >= 3:
        # ricostruiamo l'espressione seguendo l'ordine di apparizione di numeri e operatori
        token_pattern = r"(\d+(?:\.\d+)?|più|piu|meno|per|volte|diviso)"
        pezzi = re.findall(token_pattern, domanda_norm)
        mappa_operatori = {
            "più": "+", "piu": "+", "meno": "-",
            "per": "*", "volte": "*", "diviso": "/",
        }
        espressione_parti = []
        for pezzo in pezzi:
            if pezzo in mappa_operatori:
                espressione_parti.append(mappa_operatori[pezzo])
            elif re.match(r"^\d+(\.\d+)?$", pezzo):
                espressione_parti.append(pezzo)
        if len(espressione_parti) >= 3:
            candidato = " ".join(espressione_parti)
            try:
                ast.parse(candidato, mode="eval")
                return candidato
            except Exception:
                pass

    a, b = numeri[0], numeri[1]

    if any(p in domanda_norm for p in ["più", "piu", "somma", "sommare", "aggiung"]):
        return f"{a} + {b}"
    if any(p in domanda_norm for p in ["meno", "sottra", "differenza", "togli"]):
        return f"{a} - {b}"
    if any(p in domanda_norm for p in ["per", "moltiplic", "prodotto", "volte"]) and "*" not in domanda:
        return f"{a} * {b}"
    if any(p in domanda_norm for p in ["diviso", "dividi", "divisione"]):
        return f"{a} / {b}"
    if "+" in domanda:
        return f"{a} + {b}"
    if "-" in domanda:
        return f"{a} - {b}"
    if "*" in domanda or "x" in domanda_norm:
        return f"{a} * {b}"
    if "/" in domanda:
        return f"{a} / {b}"

    # Default ragionevole: se non capiamo l'operatore, proviamo la somma
    return f"{a} + {b}"


def estrai_soggetto_ricerca(domanda: str):
    """
    Rimuove le parole interrogative comuni per isolare il soggetto
    reale della domanda, da passare alla ricerca vera.
    """
    domanda_pulita = domanda.strip().rstrip("?").strip()

    prefissi_da_rimuovere = [
        r"^chi (era|fu|è)\s+",
        r"^chi (era|fu|è) stato\s+",
        r"^cosa (sai|conosci) di\s+",
        r"^parlami di\s+",
        r"^raccontami di\s+",
        r"^quando (visse|nacque|morì|mori)\s+",
        r"^quando (avvenne|iniziò|inizio|finì|fini)\s+",
        r"^come (iniziò|inizio|finì|fini|terminò|termino)\s+",
        r"^perché (iniziò|inizio|finì|fini)\s+",
        r"^cosa fu\s+",
        r"^perché è importante\s+",
        r"^perche e importante\s+",
        r"^cos'?è\s+(un|una|uno|il|la|lo|gli|le)\s+",
        r"^cos'?è\s+",
        r"^che cos'?è\s+(un|una|uno|il|la|lo|gli|le)\s+",
        r"^che cos'?è\s+",
        r"^a cosa serve\s+(un|una|uno|il|la|lo|gli|le)\s+",
        r"^a cosa serve\s+",
        r"^come funziona\s+(un|una|uno|il|la|lo|gli|le)\s+",
        r"^come funziona\s+",
        r"^cosa significa (la parola|il termine|il vocabolo)\s+",
        r"^cosa significa parola\s+",
        r"^cosa significa\s+",
        r"^che cosa significa (la parola|il termine)\s+",
        r"^che cosa significa\s+",
        r"^cosa vuol dire\s+",
        r"^che cos'?è il significato di\s+",
        r"^qual è il significato di\s+",
        r"^qual è la capitale (di|del|della|dell')\s+",
        r"^quanti abitanti ha\s+",
        r"^quanto è alto\s+",
        r"^quanto è grande\s+",
        r"^dove si trova\s+",
        r"^quando è nato\s+",
        r"^quando è morto\s+",
        r"^chi ha scritto\s+",
        r"^chi ha dipinto\s+",
        r"^chi ha composto\s+",
        r"^in che anno è nato\s+",
        r"^in che anno è morto\s+",
        r"^quanti anni visse\s+",
        r"^quanti anni ha vissuto\s+",
        r"^quanti anni ha\s+",
        r"^quanti anni aveva\s+",
        r"^chi vinse\s+",
        r"^chi ha vinto\s+",
        r"^come si dichiara (una|un)\s+",
        r"^come si (crea|scrive|usa)\s+(una|un)\s+",
    ]
    for pattern in prefissi_da_rimuovere:
        nuova = re.sub(pattern, "", domanda_pulita, flags=re.IGNORECASE)
        if nuova != domanda_pulita:
            return nuova.strip()

    # Se nessun prefisso corrisponde, usiamo l'intera domanda ripulita
    return domanda_pulita


def cerca_wikipedia(soggetto: str) -> str:
    """
    Ricerca vera e gratuita usando l'API pubblica di Wikipedia in italiano.
    Non richiede alcuna chiave API. Restituisce il riassunto introduttivo
    della pagina corrispondente al soggetto cercato.
    """
    try:
        query = urllib.parse.quote(soggetto)
        url = f"https://it.wikipedia.org/api/rest_v1/page/summary/{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "MiniGPT-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            dati = json.loads(response.read().decode("utf-8"))
        estratto = dati.get("extract", "")
        if estratto:
            return estratto
        return f"Nessun risultato trovato su Wikipedia per '{soggetto}'."
    except Exception as e:
        return f"Errore nella ricerca su Wikipedia per '{soggetto}': {e}"


class AgenteIbrido:
    def __init__(self, vocab_path, merges_path, checkpoint_path, search_fn=None,
                 vocab_prog_path=None, merges_prog_path=None, checkpoint_prog_path=None):
        self.tok = ByteLevelBPETokenizer(vocab_path, merges_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model = MiniGPT(**ckpt["config"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.search_fn = search_fn

        # Modello secondario, specializzato in risposte dirette di programmazione
        # (il modello principale è ottimizzato per tool-use, non per queste risposte)
        self.tok_prog = None
        self.model_prog = None
        if checkpoint_prog_path:
            self.tok_prog = ByteLevelBPETokenizer(vocab_prog_path, merges_prog_path)
            ckpt_prog = torch.load(checkpoint_prog_path, map_location="cpu")
            self.model_prog = MiniGPT(**ckpt_prog["config"])
            self.model_prog.load_state_dict(ckpt_prog["model_state"])
            self.model_prog.eval()

    def _classifica_intento(self, domanda: str, verbose=False) -> str:
        """
        Usa il modello SOLO per classificare l'intento (calcolo / cerca / diretta),
        non per generare i dettagli specifici. Alcune regole di buon senso
        anticipano il modello per i casi più chiari e frequenti.
        """
        domanda_norm = domanda.lower().strip()

        # Richieste di GENERARE codice/programmi (non spiegare concetti): il
        # modello di programmazione sa solo rispondere a domande descrittive,
        # non generare codice funzionante. Meglio dirlo onestamente.
        verbi_generazione_codice = ["crea un sito", "crea una pagina", "scrivi un programma",
                                      "scrivi uno script", "scrivi il codice",
                                      "genera un sito", "fammi un sito", "crea un'app",
                                      "sviluppa un", "programma un", "codifica un",
                                      "crea un programma", "scrivi una funzione",
                                      "scrivi un codice", "crea un codice",
                                      "genera un programma", "genera un codice"]
        if any(v in domanda_norm for v in verbi_generazione_codice):
            return "generazione_codice"

        # Saluti e conversazione casuale: gestiti a parte, senza modelli inaffidabili
        saluti = ["ciao", "come stai", "buongiorno", "buonasera", "grazie",
                  "raccontami una barzelletta", "dimmi una barzelletta"]
        if any(domanda_norm.startswith(s) or s in domanda_norm for s in saluti):
            return "conversazione"

        ha_numeri = bool(re.search(r"\d", domanda))

        # Conversioni di unità: priorità alta se ci sono numeri e parole chiave
        parole_conversione = ["km", "miglia", "miglio", "kg", "chili", "libbre",
                                "metri", "piedi", "centimetri", "pollici", "litri",
                                "galloni", "celsius", "fahrenheit", "gradi c", "gradi f",
                                "converti", "conversione"]
        if ha_numeri and any(p in domanda_norm for p in parole_conversione):
            return "conversione"

        # Calcoli: priorità alta se ci sono numeri e parole chiave matematiche
        parole_matematiche = ["radice quadrata", "percento", "per cento",
                               "quanto fa", "calcola", "somma", "sottrai",
                               "moltiplica", "dividi", "al quadrato", "elevat"]
        if ha_numeri and any(p in domanda_norm for p in parole_matematiche):
            return "calcola"

        # Scorciatoia di buon senso: termini di programmazione -> modello dedicato
        # Scorciatoia di buon senso: SOLO i concetti Python verificati funzionare
        # bene con questo modello (testati sistematicamente) vanno al modello
        # dedicato; tutto il resto, anche se di programmazione, va cercato,
        # perché il modello confabula su argomenti meno rappresentati nel corpus
        termini_python_verificati = ["lista in python", "differenza tra una lista",
                                       "lista e una tupla", "cos'è una lista",
                                       "cos'è una tupla", "tupla in python"]
        if any(t in domanda_norm for t in termini_python_verificati):
            return "programmazione"

        # Altri linguaggi o concetti di programmazione: meglio cercare sempre,
        # incluse funzione/ciclo/variabile/classe che il modello non padroneggia
        termini_prog_generali = ["puntatore", "javascript", "java ", " c++",
                                   "linguaggio c", "orientata agli oggetti",
                                   "programmazione a oggetti", "git", "sql",
                                   "html", "css", "algoritmo", "compilatore",
                                   "funzione", "ciclo for", "ciclo while",
                                   "variabile", "dizionario", "classe", "stringa",
                                   "python"]
        if any(t in domanda_norm for t in termini_prog_generali):
            return "cerca"

        # Domande di definizione lessicale ("cosa significa X") vanno sempre
        # cercate (il modello non ha un dizionario, meglio Wikipedia/Wikizionario)
        if any(p in domanda_norm for p in ["cosa significa", "che cosa significa",
                                             "cosa vuol dire", "significato di"]):
            return "cerca"

        # Domande fattuali molto comuni: il piccolo modello non ha un'enciclopedia
        # affidabile, meglio sempre cercare
        pattern_fattuali = [
            r"^qual è la capitale", r"^quanti abitanti", r"^chi ha scritto",
            r"^chi ha dipinto", r"^chi ha composto", r"^quanto è alto",
            r"^quanto è grande", r"^dove si trova", r"^quando è nato",
            r"^quando è morto", r"^quanti anni ha", r"^quanti anni aveva",
            r"^quanti anni visse",
        ]
        if any(re.search(p, domanda_norm) for p in pattern_fattuali):
            return "cerca"

        prompt = f"Istruzione: {domanda}\nRisposta:"
        ids = self.tok.encode(prompt).ids
        ctx = torch.tensor([ids])
        out = self.model.generate(ctx, max_new_tokens=15, temperature=0.1, top_k=3)[0].tolist()
        testo = self.tok.decode(out)

        if verbose:
            print(f"  [modello, grezzo] {testo!r}")

        if "calcola" in testo.lower():
            return "calcola"
        if "cerca" in testo.lower():
            return "cerca"
        return "diretta"

    def rispondi(self, domanda: str, verbose=True) -> str:
        intento = self._classifica_intento(domanda, verbose=verbose)
        if verbose:
            print(f"  [agente] intento riconosciuto: {intento}")

        if intento == "conversione":
            risultato_conv = estrai_conversione(domanda)
            if risultato_conv is None:
                return ("Non sono riuscito a capire quale conversione fare. "
                        "Posso convertire km/miglia, kg/libbre, metri/piedi, "
                        "cm/pollici, litri/galloni, Celsius/Fahrenheit.")
            if risultato_conv[0] == "temp_c_to_f":
                valore = risultato_conv[1]
                risultato = round(celsius_a_fahrenheit(valore), 2)
                return f"{valore}°C = {risultato}°F"
            if risultato_conv[0] == "temp_f_to_c":
                valore = risultato_conv[1]
                risultato = round(fahrenheit_a_celsius(valore), 2)
                return f"{valore}°F = {risultato}°C"

            unita_da, valore, unita_a = risultato_conv
            funzione = _CONVERSIONI.get((unita_da, unita_a))
            if funzione is None:
                return f"Non so convertire da {unita_da} a {unita_a}."
            risultato = round(funzione(valore), 4)
            return f"{valore} {unita_da} = {risultato} {unita_a}"

        if intento == "generazione_codice":
            return ("Non sono in grado di generare codice funzionante (siti web, "
                    "programmi, script): questo modello sa solo spiegare concetti "
                    "di programmazione a parole, non scrivere codice affidabile. "
                    "Dirti il contrario o inventare del codice sarebbe una risposta "
                    "sbagliata spacciata per corretta.")

        if intento == "conversazione":
            domanda_norm = domanda.lower()
            if "come stai" in domanda_norm:
                return "Sto bene, grazie! Come posso aiutarti?"
            if "grazie" in domanda_norm:
                return "Di nulla, sono qui per aiutarti."
            if "barzelletta" in domanda_norm:
                return ("Non ho un repertorio di barzellette affidabile: rischierei "
                        "di inventarne una che non ha senso. Posso aiutarti con "
                        "calcoli, fatti storici o domande di programmazione.")
            return "Ciao! Come posso aiutarti?"

        if intento == "calcola":
            espressione = estrai_espressione_matematica(domanda)
            if espressione is None:
                return ("Questa non sembra una domanda di calcolo che so gestire "
                        "(magari riguarda meteo, orari o altro): posso aiutarti con "
                        "calcoli matematici, fatti storici/geografici o concetti di "
                        "programmazione in Python.")
            risultato = calcola_sicuro(espressione)
            return f"{espressione.replace('**', '^')} = {risultato}"

        elif intento == "cerca":
            soggetto = estrai_soggetto_ricerca(domanda)
            if self.search_fn is None:
                return f"[Cercherei informazioni su: '{soggetto}', ma nessuna funzione di ricerca è collegata]"
            return self.search_fn(soggetto)

        elif intento == "programmazione" and self.model_prog is not None:
            # Usa il modello dedicato alla programmazione, con il formato
            # di prompt su cui è stato addestrato
            prompt = f"Istruzione: Spiegalo con parole semplici: {domanda}\nRisposta:"
            ids = self.tok_prog.encode(prompt).ids
            ctx = torch.tensor([ids])
            out = self.model_prog.generate(ctx, max_new_tokens=60, temperature=0.5, top_k=20)[0].tolist()
            testo = self.tok_prog.decode(out)
            if "Risposta:" in testo:
                return testo.split("Risposta:", 1)[1].split("Istruzione:")[0].strip()
            return testo.strip()

        else:
            # Risposta diretta generata dal modello (per domande di programmazione ecc.)
            prompt = f"Istruzione: {domanda}\nRisposta:"
            ids = self.tok.encode(prompt).ids
            ctx = torch.tensor([ids])
            out = self.model.generate(ctx, max_new_tokens=60, temperature=0.5, top_k=20)[0].tolist()
            testo = self.tok.decode(out)
            if "Risposta:" in testo:
                risposta = testo.split("Risposta:", 1)[1].split("Istruzione:")[0].strip()
            else:
                risposta = testo.strip()

            # Se il modello ha comunque generato una chiamata TOOL invece di una
            # risposta diretta, meglio dirlo onestamente che mostrare testo grezzo
            if risposta.startswith("[TOOL:"):
                return ("Non sono riuscito a generare una risposta diretta affidabile "
                        "per questa domanda con il modello attuale.")
            return risposta


if __name__ == "__main__":
    agent = AgenteIbrido(
        "wiki_bpe_it14-vocab.json",
        "wiki_bpe_it14-merges.txt",
        "minigpt_it14_best.pt",
        search_fn=cerca_wikipedia,  # ricerca vera, gratuita, tramite Wikipedia
        vocab_prog_path="wiki_bpe_prog-vocab.json",
        merges_prog_path="wiki_bpe_prog-merges.txt",
        checkpoint_prog_path="minigpt_prog_best.pt",
    )

    print("=== ClacIA pronta. Scrivi una domanda (o 'esci' per terminare) ===\n")
    while True:
        domanda = input("Tu: ").strip()
        if domanda.lower() in ("esci", "exit", "quit"):
            break
        if not domanda:
            continue
        print(f"ClacIA: {agent.rispondi(domanda, verbose=False)}\n")
