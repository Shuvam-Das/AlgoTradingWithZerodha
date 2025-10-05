# Architecture and Roadmap — Super Advanced Zerodha Trading Bot

This document outlines the modular architecture, data flows, security controls, and roadmap for the advanced trading bot integrating with Zerodha/Kite.

High-level components
- src/api.py: FastAPI-based HTTP API (health, predict, train, backtest, webhooks). Token-auth middleware.
- src/data_fetcher.py: Historical and intraday data ingestion (yfinance, broker, NSE/BSE). Caching & normalization.
- src/storage.py: Persistent storage (SQLite by default, PostgreSQL option). Stores orders, trades, models metadata.
- src/kite_live.py / src/kite_client.py: Kite Connect wrapper with simulated fallback. Order placement, positions, margins.
- src/kite_ws.py / src/webhook_receiver.py: Real-time connections and webhook order updates.
- src/strategy.py: Strategy interface and implementations.
- src/executor.py / src/order_manager.py: Execution engine and lifecycle tracking.
- src/ml/: Machine-learning modules (training, inference, model registry).
- src/security_utils.py: Encryption helpers, token management, secret storage guidance.
- frontend/: TypeScript React app for TradingView-like charts and interactions.

Data flows
- Ingestion: Historical & realtime data from Kite + supplemental sources -> normalizer -> storage.
- Analysis: Strategies & ML models read historical data from storage and produce signals.
- Execution: Signals go to Executor -> OrderManager -> Kite. Webhooks update status -> storage.
- Feedback loop: Market fills and performance results feed ML training / strategy optimization.

Security & secrets
- Use environment variables for secrets in development and OS keyring / HashiCorp Vault in production.
- Encrypt stored secrets with Fernet (symmetric key stored in KMS or Vault).
- Token-based auth for API endpoints; rotate tokens periodically.

Deployment
- Docker + docker-compose for local dev; Kubernetes for production.
- GitHub Actions: lint, test, bandit, mypy, container build, and optional deploy steps.

Roadmap (phases)
1) Core: data ingestion, strategy engine, simulated execution, tests.
2) Live integration: Kite Connect auth, websocket, webhook, margin-aware sizing.
3) ML: basic supervised models (sklearn), model registry, scheduled retraining.
4) Advanced: reinforcement learning/genetic optimization, real-time dashboard, streaming pipelines.

Notes
- This project emphasizes modularity to allow safe testing in simulated mode before enabling live trading.
