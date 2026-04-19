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

### 📌 Descrição das Variáveis

- ID:
  Representa a identificação única de cada registro.
- Customer_ID:
  Identificação única de cada cliente.
- Month:
  Representa o mês do ano referente ao registro.
- Name:
  Nome da pessoa.
- Age:
  Idade do cliente.
- SSN:
  Número de identificação (equivalente ao CPF/SSN).
- Occupation:
  Ocupação/profissão do cliente.
- Annual_Income:
  Renda anual do cliente.
- Monthly_Inhand_Salary:
  Salário mensal líquido do cliente.
- Num_Bank_Accounts:
  Número de contas bancárias que o cliente possui.
- Num_Credit_Card:
  Número de cartões de crédito que o cliente possui.
- Interest_Rate:
  Taxa de juros aplicada ao crédito.
- Num_of_Loan:
  Quantidade de empréstimos contratados pelo cliente.
- Type_of_Loan:
  Tipos de empréstimos contratados.
- Delay_from_due_date:
  Média de dias de atraso nos pagamentos.
- Num_of_Delayed_Payment:
  Número médio de pagamentos em atraso.
- Changed_Credit_Limit:
  Percentual de mudança no limite de crédito.
- Num_Credit_Inquiries:
  Número de consultas de crédito realizadas.
- Credit_Mix:
  Classificação da composição de crédito do cliente.
- Outstanding_Debt:
  Dívida pendente total (em USD).
- Credit_Utilization_Ratio:
  Percentual de utilização do limite de crédito.
- Credit_History_Age:
  Tempo de histórico de crédito do cliente.
- Payment_of_Min_Amount:
  Indica se o cliente pagou apenas o valor mínimo da fatura.
- Total_EMI_per_month:
  Valor total pago mensalmente em parcelas (EMI).
- Amount_invested_monthly:
  Valor mensal investido pelo cliente.
- Payment_Behaviour:
  Comportamento de pagamento do cliente.
- Monthly_Balance:
  Saldo mensal disponível do cliente.
- Credit_Score:
  Classficação de crédito do cliente. Variável Target

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
- Precision / Recall

## 💡 Insights de Negócio

- Clientes com maior risco apresentam [exemplo: alta razão dívida/renda]
- O modelo permite ajustar o threshold para balancear risco vs aprovação

## 🛠️ Tecnologias

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP

## 🚀 Pipeline do modelo em produção

1. Executar main.py, responsável por ler a base de dados crua, fazer o pré-processamento e gerar a base tratada que treinará o modelo;
2. Executar src/models/train_model.py que gera o arquivo .pkl que pode ser utilizado em produção pra gerar as predições;
3. O arquivo src/models/predict.py é responsável por realizar as predições em produção;
4. Para testes isolados, executar o arquivo src/models/test_predict.py. O csv dos clientes para realizar a predição, deve ser salvo como o path: data/raw/clients_to_predict.csv.
   Os resultados estarão presentes em data/predictions/clients_to_predict.csv
