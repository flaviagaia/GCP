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
