# ClacIA — Calcola, Cerca, Intelligenza Artificiale

Un sistema ibrido che combina un piccolo modello linguistico addestrato
da zero con strumenti esterni reali (calcolatrice, ricerca Wikipedia),
per aggirare i limiti di precisione dei modelli piccoli.

## Come funziona

ClacIA usa **due modelli specializzati** più del codice deterministico:

1. **Modello di classificazione** (894.840 parametri): riconosce se una
   domanda è un calcolo, una ricerca di fatti, o una domanda di
   programmazione
2. **Estrazione precisa via codice**: per calcoli e ricerche, i numeri e
   i soggetti vengono presi dalla domanda originale con espressioni
   regolari — mai generati dal modello, quindi sempre esatti
3. **Modello di programmazione** (3.999.000 parametri): risponde
   direttamente a domande su Python, liste, funzioni, Git, ecc., con un
   modello addestrato specificamente su questi contenuti
4. **Esecuzione reale**: i calcoli sono fatti da Python (sempre esatti),
   le ricerche usano l'API pubblica e gratuita di Wikipedia in italiano

## Esempi

```
Tu: Quanto fa 234 + 789?
ClacIA: 234 + 789 = 1023

Tu: Chi era Napoleone Bonaparte?
ClacIA: (riassunto vero da Wikipedia su Napoleone Bonaparte)

Tu: Cos'è una lista in Python?
ClacIA: Una lista è mutabile: gli elementi possono essere aggiunti,
rimossi o sostituiti. Una tupla è invece immutabile dopo la creazione...
```

## Come usarlo

```bash
pip install torch tokenizers
python3 agent_ibrido.py
```

Si apre la modalità interattiva di ClacIA. La ricerca Wikipedia
richiede una connessione Internet normale (non funziona nella sandbox
di sviluppo, che ha rete ristretta, ma funziona su qualsiasi computer).

## Perché è fatto così (il percorso che ci ha portato qui)

Abbiamo scoperto sperimentalmente che un piccolo modello linguistico
**non riesce a copiare con precisione** numeri o nomi specifici da una
domanda alla risposta — anche isolando il compito al massimo (solo
copiare due numeri), il modello sbagliava sempre almeno una cifra.

La soluzione: **non chiedere al modello di fare quello che non sa fare
bene**. Il modello classifica (compito alla sua portata), il codice
estrae con precisione (compito impossibile da sbagliare per un
programma deterministico), e per la programmazione — dove un modello
dedicato risponde bene sui concetti che ha visto abbastanza — lasciamo
rispondere direttamente il modello specializzato.

## Limiti onesti

- La classificazione dell'intento non è perfetta al 100%
- Il modello di programmazione risponde bene sui temi ben coperti dal
  training (liste, tuple, funzioni base) ma può confabulare su
  argomenti meno rappresentati
- La ricerca Wikipedia funziona bene per argomenti enciclopedici noti

## File inclusi

- `model.py` — architettura transformer
- `agent_ibrido.py` — ClacIA: classificazione, estrazione precisa,
  calcolatrice, ricerca Wikipedia, modello di programmazione dedicato
- `minigpt_it14_best.pt` — modello di classificazione/tool-use
- `minigpt_prog_best.pt` — modello specializzato in programmazione
- file di tokenizer per entrambi i modelli
