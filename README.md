# Finsights AI Service

An AI service for financial research using agentic retrieval-augmented generation
(RAG), structured market data, and document understanding. The planned architecture
combines FastAPI, LangChain, LangGraph, Voyage AI embeddings, and optional Jev
decision services.

**Status:** Repository scaffold. Directories contain `.gitkeep` placeholders.
Application code, dependencies, API endpoints, and infrastructure are not implemented yet.

## Architecture

The service covers three workflows:

- **Offline processing:** Read collected documents and news, normalize metadata,
  extract text and tables, create chunks, generate embeddings, and update indexes.
- **Online assistance:** Route requests to a fast SQL/RAG path or a single agent
  loop, verify results, and stream answers with source citations.
- **Evaluation and training:** Compare retrieval and decision strategies,
  benchmark document extraction, and prepare data for optional VLM fine-tuning.

The existing frontend, authentication gateway, and collection services are external
dependencies. See [Architecture](docs/architecture.md) for system boundaries and data flows.

## Planned Technology Stack

| Component | Technology | Location |
| --- | --- | --- |
| HTTP API and streaming | FastAPI, Pydantic, server-sent events (SSE) | `app/api/`, `app/schemas/` |
| Workflow orchestration | LangGraph state, nodes, conditional edges, and checkpoints | `app/graph/` |
| Models and tools | LangChain and provider integrations | `app/tools/`, `app/integrations/` |
| Embeddings | Voyage AI API | `app/integrations/voyage/` |
| Hybrid retrieval | PostgreSQL, pgvector, lexical search, RRF, and reranking | `app/rag/` |
| Decision services | Jev adapter with interchangeable baselines | `app/integrations/jev/`, `app/decisions/` |
| Offline orchestration | Dagster assets, jobs, schedules, and sensors | `orchestration/` |
| Document understanding | Docling, OCR, VLM APIs, and optional LoRA adapters | `app/vision/` |
| Business data and chat history | Existing MySQL databases; SQLAlchemy when implemented | `app/storage/mysql/` |
| Source documents | OCI Object Storage | `app/integrations/oci/` |
| Evaluation and fine-tuning | Evaluation harness; Transformers and PEFT for LoRA | `evals/`, `finetuning/` |

LangGraph will control routing, retries, and state transitions. LangChain will
provide model and tool abstractions. Calculations, numeric verification, and
parameterized SQL queries will use deterministic application logic.

### Embedding Configuration

Voyage AI is the planned embedding provider. The initial configuration uses
`voyage-4` with 1,024 dimensions as a multilingual retrieval baseline.
`voyage-finance-2` is a candidate for financial-domain evaluation. These capabilities
and dimensions are documented in the [Voyage AI embedding guide](https://docs.voyageai.com/docs/embeddings).
The final choice should be validated against the project's Vietnamese financial documents and queries.

```dotenv
EMBEDDING_PROVIDER=voyage
VOYAGE_API_KEY=
EMBEDDING_MODEL=voyage-4
EMBEDDING_DIMENSIONS=1024
```

During implementation, use `input_type="document"` for indexed content and
`input_type="query"` for search queries, as described in the Voyage AI guide.
Record the model and dimension for each index. Switching between model families
requires re-embedding the indexed corpus; matching dimensions alone do not establish compatibility.

PostgreSQL's built-in full-text search does not implement BM25. Select a BM25
extension or a separately identified lexical baseline before implementing hybrid retrieval.

## Repository Structure

```text
app/
  api/
    routes/                  # Chat, upload, and health endpoints
    streaming/               # SSE event handling
  core/                      # Settings, logging, and execution limits
  schemas/                   # API, document, tool result, and citation schemas
  graph/
    state/                   # Shared LangGraph state
    nodes/                   # Router, fast path, agent, verifier, and guardrails
    checkpoints/             # Execution state persistence adapters
  decisions/
    routing/                 # Route selection and confidence handling
    relevance/               # Relevance and context sufficiency decisions
    classification/          # Document type, page kind, and scan detection
    baselines/               # Rule, LLM, and classifier alternatives to Jev
  tools/
    sql/                     # Read-only tools with parameterized queries
    rag/                     # Disclosure and news search tools
    calculator/              # Calculations referencing tool results
    uploads/                 # User-upload processing through the vision pipeline
  rag/
    chunking/                # Structural, table, and parent-child chunking
    indexing/                # Embedding generation, upserts, and metadata
    retrieval/               # Dense and lexical search, filters, and RRF
    reranking/               # Jev and baseline rerankers
    citations/               # Source, asset, page, and block references
  ingestion/
    sources/                 # Readers for existing asset and news sources
    normalization/           # Metadata, identifiers, and reporting periods
    pipelines/               # Classification, extraction, chunking, and indexing
  vision/
    rendering/               # PDF-to-image conversion
    parsers/                 # Text extraction, OCR, Docling, and VLM adapters
    tables/                  # Structured table extraction
    metrics/                 # Financial metric, period, and unit mapping
  integrations/
    llm/                     # Language model provider integrations
    jev/                     # Choice, score, and Noul API adapter
    voyage/                  # Voyage AI embedding integration
    oci/                     # Object Storage integration
  storage/
    mysql/                   # Business data, chat, and vision persistence
    postgres/                # Document pages, chunks, metadata, and vector indexes
  observability/             # Traces, latency, token usage, and cost tracking
  prompts/                   # Versioned prompt templates
orchestration/
  assets/                    # Dagster asset definitions
  jobs/                      # Offline jobs
  schedules/                 # Scheduled execution
  sensors/                   # Ingestion triggers after collection updates
configs/
  models/                    # Provider and model profiles
  retrieval/                 # Filters, top-k, fusion, and thresholds
  experiments/               # Baseline comparisons and ablation settings
evals/
  datasets/                  # Development/test sets and routing labels
  snapshots/                 # Snapshot manifests; database dumps excluded
  runners/                   # LLM-only, RAG-only, agent, and full-pipeline runs
  metrics/                   # Retrieval, citation, numeric, routing, and vision metrics
  reports/                   # Generated benchmark reports; contents ignored
finetuning/
  datasets/                  # Weakly labeled data, split by company
  configs/                   # Model and LoRA training settings
  scripts/                   # Data preparation, training, and export
  notebooks/                 # GPU and Colab experiments
migrations/
  postgres/                  # New RAG schema migrations
  mysql/                     # Chat and vision changes aligned with existing schemas
deploy/
  docker/                    # Future Dockerfiles and Compose configuration
  k8s/                       # Future Kubernetes manifests
scripts/                     # Administration, ingestion, and evaluation commands
tests/
  unit/
  integration/
  fixtures/
docs/
  architecture.md            # System boundaries and data flows
  adr/                       # Architecture decision records
```

## Configuration and Development Status

[`.env.example`](.env.example) lists proposed settings for model providers,
Voyage AI, databases, Object Storage, Jev, and execution limits. It contains no
credentials. A configuration loader has not been implemented.

There are currently no installation, startup, migration, or test commands.
Dependency versions and lockfiles will be introduced with the first implementation.

Keep credentials, source PDFs, raw financial data, database dumps, and model weights
outside version control. Store large artifacts externally and track their identifiers or manifests.

## Implementation Roadmap

1. Define gateway contracts, document metadata, citations, and SSE event schemas.
2. Implement ingestion and hybrid retrieval, then a RAG endpoint with citations.
3. Build the LangGraph workflow: router, fast path or agent/tools, and verifier.
4. Integrate Jev through an adapter and compare it with baseline strategies.
5. Benchmark vision extraction before introducing VLM fine-tuning.

## References

- [BOA](https://github.com/kutin2005gl/BOA): inspiration for AI service boundaries,
  gateway integration, provider configuration, and retrieval organization.
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview): workflow orchestration and state management.
- [LangChain](https://docs.langchain.com/oss/python/langchain/overview): model and tool integrations.
- [Voyage AI](https://docs.voyageai.com/docs/embeddings): embedding models and API usage.
