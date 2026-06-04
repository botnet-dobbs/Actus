# Actus

**FastAPI-based multi-agent platform with LLM routing, RAG-powered context retrieval, and operational automation.**

Actus is an internal automation platform. Deploy it, register agents for your operations, and let them work autonomously on your data. A self-hosted infrastructure layer your team owns and controls.

Define agents in YAML for whatever you need to automate: analyse customer data, monitor servers, digest logs, process documents, generate reports. Each agent runs on a schedule or on demand, calls your internal tools, and reads your domain knowledge through RAG. Multiple agents run independently.

Actus provides the platform, you bring the tools. A tool is a Python function decorated with `@tool` that connects an agent to your systems: your database, your APIs, your filesystem. The platform handles the agent loop, retries, timeouts, PII scrubbing, and observability. You write the functions that do the actual work.

---

## Stack

```
Python 3.13      FastAPI + Uvicorn       SQLModel + SQLite/PostgreSQL
LiteLLM          Ollama                  Presidio (PII)
APScheduler      structlog               bcrypt + python-jose
pytest + httpx
```

---

## Quick Start

**Prerequisites:** Docker and Docker Compose.

```bash
git clone https://github.com/you/actus && cd actus

cat > .env << 'EOF'
SECRET_KEY=your-secret-key-here
POSTGRES_PASSWORD=your-postgres-password-here
GRAFANA_PASSWORD=your-grafana-password-here
DEBUG=false
EOF

make docker-up-d       # start all services in background
make ollama-pull       # pull the model into the Ollama container (first run only)
```

The first build takes a few minutes. The spaCy NLP model is downloaded into the image. Subsequent starts are fast. Actus starts immediately; Ollama initialises in the background (typically 1-3 minutes on first run).

To check when Ollama is ready:

```bash
curl http://localhost:8000/healthz
# Not ready yet:  {"status":"degraded","checks":{"database":"ok","ollama":"unreachable"}}
# Ready:          {"status":"ok","checks":{"database":"ok","ollama":"ok"}}
```

**Services:**

| Service | URL | Notes |
|---|---|---|
| Actus API | `http://localhost:8000` | API docs at `/docs` |
| Prometheus | `http://localhost:9090` | Metrics storage |
| Grafana | `http://localhost:3000` | Dashboards — login: `admin` / `GRAFANA_PASSWORD` |

**Commands:**

| Command | What it does |
|---|---|
| `make docker-up` | Build (if needed) and start in foreground |
| `make docker-up-d` | Start in background |
| `make docker-logs` | Tail all service logs |
| `make docker-restart` | Restart Actus without rebuilding |
| `make docker-rebuild` | Rebuild image and restart all services |
| `make docker-down` | Stop and remove all containers |
| `make ollama-pull` | Pull a model into the Ollama container |

Agent YAML files in `config/agents/` are volume-mounted — add or edit agents and `make docker-restart`, no rebuild needed. Database data persists in a Docker volume across restarts.

---

## Documentation

## License

MIT.
