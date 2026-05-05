# Email World Agent

Agent d'email avec:
- environnement simulé,
- world model PyTorch,
- mémoire vectorielle FAISS (`IndexFlatL2(6)`).

## Prerequis

- Python 3.11 recommande
- macOS (Apple Silicon compatible)

## Installation

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
```

Sortie attendue (exemple):

```text
Similar memories:
- action: répondre maintenant | reward: 0.62
- action: validation humaine | reward: 0.44
Decision: répondre maintenant
```

## Notes

- Les dependances sont listees dans `requirements.txt`.
- La branche de travail actuelle est `cursor/faiss-memory-agent`.
