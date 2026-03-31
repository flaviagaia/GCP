# GCP

## Português

Este repositório reúne três projetos de **ML clássico** pensados como portfólio técnico: `credit default prediction`, `customer churn prediction` e `fraud detection baseline`.

### Projetos incluídos

1. `credit_default_prediction`
   Problema supervisionado de classificação binária para estimar risco de inadimplência.
2. `customer_churn_prediction`
   Problema supervisionado de classificação binária para prever cancelamento de clientes.
3. `fraud_detection_baseline`
   Benchmark híbrido entre abordagem supervisionada e detecção de anomalias para fraude.

### Objetivo técnico

O foco deste repositório é mostrar fundamentos de modelagem clássica com:

- engenharia de atributos tabulares;
- separação treino/teste estratificada;
- comparação entre modelos tradicionais;
- métricas adequadas para classificação desbalanceada;
- persistência de artefatos analíticos em `JSON`.

### Stack

- `pandas`
- `numpy`
- `scikit-learn`
- `unittest`

### Modelos utilizados

#### Credit Default Prediction
- `Logistic Regression`
- `RandomForestClassifier`

#### Customer Churn Prediction
- `Logistic Regression`
- `RandomForestClassifier`

#### Fraud Detection Baseline
- `Logistic Regression` com `class_weight="balanced"`
- `IsolationForest`

### Estrutura dos dados

Os três projetos usam datasets sintéticos reproduzíveis gerados via `scikit-learn`, com variáveis derivadas para simular cenários de negócio:

- crédito: renda, razão dívida/renda, utilização, histórico e pressão de parcelas;
- churn: engajamento, fricção de suporte, profundidade de uso, tenure e saúde da conta;
- fraude: velocidade transacional, risco do merchant, conflito de identidade, atividade noturna e sinais de card testing.

### Artefato gerado

O pipeline consolidado salva:

- `data/processed/classic_ml_portfolio_report.json`

Esse arquivo é gerado em runtime e não é versionado.

### Execução

```bash
python3 main.py
python3 -m unittest discover -s tests -v
python3 -m py_compile main.py src/data_factory.py src/projects.py
```

### Leitura técnica

Este repositório foi desenhado para enfatizar que **ML clássico ainda é extremamente relevante** quando o problema é tabular, o conjunto de dados é relativamente estruturado e a explicabilidade operacional importa mais do que complexidade de arquitetura.

No caso de fraude, a presença simultânea de um baseline supervisionado e de um detector de anomalias ajuda a mostrar duas famílias de abordagem:

- classificação com rótulo explícito;
- identificação de comportamento raro sem depender integralmente de labels.

### Roadmap de projetos com GCP

Além do bundle atual de ML clássico, este repositório também serve como base conceitual para evoluções orientadas a Google Cloud Platform. A ideia é mapear uma ferramenta principal do GCP para um caso de uso de portfólio com valor arquitetural claro.

| Ferramenta GCP | Projeto sugerido | Papel técnico |
| --- | --- | --- |
| `Vertex AI` | `credit-risk-scoring-platform` | treino, serving e explicabilidade de modelos de risco |
| `BigQuery` | `customer-churn-analytics` | feature mart, analytics e scoring em larga escala |
| `Cloud Run` | `ticket-triage-api` | serving stateless de inferência ou regras |
| `Cloud Storage` | `document-intelligence-lake` | armazenamento de datasets, imagens e artefatos |
| `Pub/Sub` | `fraud-stream-monitoring` | ingestão orientada a eventos |
| `Dataflow` | `fraud-stream-monitoring` | processamento stream/batch e enriquecimento |
| `Document AI` | `document-intake-automation` | extração estruturada de documentos |
| `Looker Studio` | `risk-executive-dashboard` | visualização executiva de KPIs e SLAs |
| `Cloud Monitoring + Logging` | `ml-observability-gcp` | observabilidade de pipelines e serviços |
| `Apigee` | `decisioning-api-gateway` | governança, segurança e exposição de APIs |

### Como este repositório se conecta com GCP

Os três projetos atuais são propositalmente independentes de cloud para manter:

- reprodutibilidade local;
- baixo custo de execução;
- validação rápida em ambiente de portfólio.

Mas eles podem evoluir diretamente para GCP:

- `credit_default_prediction` -> `Vertex AI + BigQuery + Cloud Run`
- `customer_churn_prediction` -> `BigQuery + Vertex AI + Looker Studio`
- `fraud_detection_baseline` -> `Pub/Sub + Dataflow + BigQuery + Vertex AI`

Essa transição é natural porque o núcleo analítico já está isolado em pipelines versionáveis e testáveis.

### Referência de serviços

As sugestões acima seguem o catálogo oficial de produtos do Google Cloud, com destaque para `BigQuery`, `Cloud Run`, `Vertex AI`, `Cloud Storage`, `Looker`, `Apigee`, `Document AI` e demais serviços da plataforma:

- [Google Cloud products](https://cloud.google.com/products/)

---

## English

This repository bundles three **classic ML** portfolio projects: `credit default prediction`, `customer churn prediction`, and a `fraud detection baseline`.

### Included projects

1. `credit_default_prediction`
   Binary classification for default risk estimation.
2. `customer_churn_prediction`
   Binary classification for churn prediction.
3. `fraud_detection_baseline`
   Hybrid benchmark combining supervised classification and anomaly detection.

### Technical focus

- tabular feature engineering
- stratified train/test split
- classical model comparison
- metrics suitable for imbalanced classification
- JSON analytical artifact generation

### Runtime artifact

- `data/processed/classic_ml_portfolio_report.json`

This artifact is generated at runtime and is not versioned.

### GCP project ideas mapped to core services

Besides the current classic ML bundle, this repository also acts as a planning base for GCP-oriented portfolio projects, mapping core Google Cloud services to concrete implementation ideas.

| GCP service | Suggested project | Technical role |
| --- | --- | --- |
| `Vertex AI` | `credit-risk-scoring-platform` | model training, deployment, and explainability |
| `BigQuery` | `customer-churn-analytics` | analytical warehouse and feature store |
| `Cloud Run` | `ticket-triage-api` | stateless inference serving |
| `Cloud Storage` | `document-intelligence-lake` | artifact and raw data storage |
| `Pub/Sub` | `fraud-stream-monitoring` | event ingestion |
| `Dataflow` | `fraud-stream-monitoring` | stream and batch processing |
| `Document AI` | `document-intake-automation` | structured extraction from documents |
| `Looker Studio` | `risk-executive-dashboard` | BI layer for operational KPIs |
| `Cloud Monitoring + Logging` | `ml-observability-gcp` | production observability |
| `Apigee` | `decisioning-api-gateway` | API governance and exposure |

### How the current bundle can evolve to GCP

- `credit_default_prediction` -> `Vertex AI + BigQuery + Cloud Run`
- `customer_churn_prediction` -> `BigQuery + Vertex AI + Looker Studio`
- `fraud_detection_baseline` -> `Pub/Sub + Dataflow + BigQuery + Vertex AI`

Official services reference:

- [Google Cloud products](https://cloud.google.com/products/)
