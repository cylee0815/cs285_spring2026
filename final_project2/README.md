# Portfolio IQL

Research-grade offline reinforcement learning framework for 8-asset portfolio
allocation using **Implicit Q-Learning (IQL)**. Built for reproducibility and
extensibility; designed to match the quality bar of ML conference repositories.

## Status

**Milestone 0 — Scaffolding (complete).** Configuration system, deterministic
seeding, logging, and smoke tests are in place. Training, environment, and
evaluation modules are scaffolded as empty packages to be filled in subsequent
milestones.

## Layout

```
final_project2/
├── data/             # price download + offline dataset builder
├── env/              # PortfolioEnv simulator
├── features/         # feature engineering + normalization
├── models/           # Q / V / policy networks + shared MLP
├── algorithms/       # IQL agent
├── training/         # training loop
├── evaluation/       # backtest, metrics, baselines
├── experiments/
│   └── configs/      # YAML experiment configs
├── utils/            # config loader, seeding, logging
├── tests/            # unit + smoke tests
└── results/          # experiment outputs (gitignored)
```

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
cd final_project2
uv sync
```

## Running the smoke tests (Milestone 0)

```bash
uv run pytest
```

Expected: all tests green. The smoke tests validate that

1. every module is importable,
2. the default YAML config loads into typed dataclasses,
3. CLI-style overrides (e.g. `iql.expectile_tau=0.85`) are applied correctly,
4. seeding produces deterministic numpy / torch / python-random streams,
5. the `Logger` can be instantiated in disabled mode without network side effects.

## Asset universe

The default experiment uses an 8-asset orthogonal ETF basket:

| Ticker | Exposure            |
| :----: | ------------------- |
|  SPY   | US equities         |
|  EEM   | Emerging markets    |
|  TLT   | Long duration bonds |
|  HYG   | High-yield credit   |
|  DBC   | Broad commodities   |
|  GLD   | Gold                |
|  UUP   | US dollar           |
|  SHY   | Cash proxy          |

Data begins **2008-01-01** (constrained by UUP/DBC launch dates).

## Config system

All experiments are driven by YAML files under `experiments/configs/`. Configs
are loaded into typed dataclasses via `utils.config.load_config`, which also
supports dotted CLI overrides:

```bash
uv run python -m training.train_iql \
    --config experiments/configs/default.yaml \
    iql.expectile_tau=0.85 \
    train.seed=7
```

## Roadmap

- [x] **M0** — Scaffolding, config, seeding, logging
- [x] **M1** — Data ingestion + feature pipeline
- [ ] **M2** — Portfolio environment (tests first)
- [ ] **M3** — Offline dataset generation (tests first)
- [ ] **M4** — Q / V / policy network architectures
- [ ] **M5** — IQL training loop
- [ ] **M6** — Walk-forward backtest + baselines + tear sheet

## License

MIT.
