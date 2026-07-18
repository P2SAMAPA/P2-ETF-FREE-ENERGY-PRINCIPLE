# Free Energy Principle – Market Beliefs for ETFs

Implements Karl Friston's Free Energy Principle (active inference) for markets. Markets are treated as systems minimising variational free energy through collective belief updating. Price dynamics emerge from agents maintaining internal generative models. The per‑ETF score measures market belief coherence.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Agent-based ensemble with generative models
- Variational free energy minimisation
- Belief coherence as market stability signal
- Score = belief coherence (higher = more stable)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-free-energy-principle-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High belief coherence → stable market regime → potential alpha.
- Low belief coherence → fragmented beliefs → regime change.

## Requirements

See `requirements.txt`.
