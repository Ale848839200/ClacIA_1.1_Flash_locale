# ClacIA 1.1 Flash — avvio in locale

## Cosa fa
- Calcoli esatti (codice, non LLM)
- Tutor **Python** e **HTML** da corpus (~17.000 schede)
- **Pratica**: ripasso teoria + esempio di codice
- Fatti via Wikipedia (serve internet)
- Chat base

## Requisiti
- Python 3.10+
- ~500 MB di RAM per i checkpoint
- (opzionale) connessione per Wikipedia

## Installazione
```bash
cd cartella_clacia
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Avvio
```bash
python3 agent_flash.py
```

## File obbligatori (stessa cartella)
```
agent_flash.py
corpus_rank.py
model.py
minigpt_it14_best.pt
minigpt_prog_best.pt          # opzionale per fallback
wiki_bpe_it14-vocab.json
wiki_bpe_it14-merges.txt
wiki_bpe_prog-vocab.json      # se usi prog
wiki_bpe_prog-merges.txt
clacia_data/
  corpus_python_it.txt
  corpus_html_it.txt
  corpus_italiano_espositivo.txt
requirements.txt
```

## Esempi di domande
- `Quanto fa 15 meno 7?`
- `a cosa serve append?`
- `Che cos'è HTML?`
- `devo fare la media dei numeri dall'utente`
- `chi era Leonardo da Vinci?`

## Limiti
- Non è ChatGPT: specializzato su calcoli + Python/HTML didattici
- Non genera applicazioni complete
- Wikipedia può rispondere 429 se troppe richieste
- Corpus grandi: primo caricamento può richiedere qualche secondo

## Versione
ClacIA **1.1 Flash** — ibrido (MiniGPT + corpus + tool)
