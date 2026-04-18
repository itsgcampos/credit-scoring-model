# Credit Scoring Model

## 📌 Objetivo

Desenvolver um modelo de credit scoring para prever a probabilidade de inadimplência de clientes, auxiliando na tomada de decisão de concessão de crédito.

## 🧠 Problema de Negócio

Instituições financeiras precisam decidir quem aprovar ou não crédito minimizando risco e maximizando retorno.

Este projeto responde:

- Qual a probabilidade de um cliente se tornar inadimplente?
- Qual o melhor threshold de aprovação?

## 📊 Dataset

- Fonte: Dataset de crédito (arquivo credit_score.csv)
- Target: default (1 = inadimplente, 0 = adimplente)

## ⚙️ Pipeline

- Análise exploratória (EDA)
- Limpeza e tratamento de dados
- Feature engineering
- Treinamento de modelos
- Avaliação e comparação
- Geração de score

## 🤖 Modelos Utilizados

- Regressão Logística (baseline interpretável)
- Gradient Boosting (modelo mais robusto)

## 📈 Métricas

- AUC-ROC
- KS (Kolmogorov-Smirnov)
- Precision / Recall

## 💡 Insights de Negócio

- Clientes com maior risco apresentam [exemplo: alta razão dívida/renda]
- O modelo permite ajustar o threshold para balancear risco vs aprovação

## 🚀 Próximos Passos

- Deploy do modelo
- Monitoramento (drift e performance)
- Integração com pipeline de dados

## 🛠️ Tecnologias

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
