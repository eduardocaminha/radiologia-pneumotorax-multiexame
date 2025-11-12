# Detecção de Pneumotórax em Múltiplos Tipos de Exames

Pipeline completo para identificar casos de pneumotórax em exames variados (TC, US, RM, procedimentos, etc), validar com LLM e buscar RX de tórax relacionados para auditoria.

## 📋 Contexto

Muitos laudos de RX de tórax não possuem laudo médico registrado no sistema. Para criar um dataset de possíveis RX de tórax com pneumotórax, este projeto implementa uma abordagem indireta:

1. Busca menções a "pneumotórax" em **qualquer tipo de exame** do paciente
2. Valida com LLM se é pneumotórax real (não apenas menção/negação)
3. Para casos confirmados, busca **todos os RX de tórax** daqueles atendimentos

**Objetivo:** Criar dataset de RX de tórax com alta probabilidade de pneumotórax, incluindo exames sem laudo que podem ser auditados manualmente.

## 🏗️ Arquitetura

### Pipeline em 3 Camadas (Bronze → Silver → Gold)

```
┌────────────────────────────────────────────────────────────────┐
│ BRONZE: Busca Inicial                                          │
│ - Query mês a mês (2000-2025) em HSP e PSC                    │
│ - 48 tipos de procedimentos (TC, US, RM, drenagens, etc)     │
│ - Filtra laudos com termo "pneumot"                          │
│ - Extrai trechos contextuais (±30 chars)                     │
│ → Tabela: innovation_dev.bronze.radiologia_pneumotorax_...   │
└────────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│ SILVER: Validação com LLM                                      │
│ - Llama 3.1 8B (Databricks Foundation Models)                │
│ - Prompt: "SIM se pneumotórax confirmado (>50% confiança)"   │
│ - Classifica cada trecho como SIM/NAO                        │
│ → Tabela: innovation_dev.silver.radiologia_pneumotorax_...   │
└────────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│ GOLD: RX de Tórax dos Casos Positivos                         │
│ - Filtra apenas casos com LLM = "SIM"                        │
│ - Busca 19 tipos de RX de tórax daqueles atendimentos       │
│ - Traz laudos (quando existem) ou marca "SEM LAUDO"         │
│ → Tabela: innovation_dev.gold.radiologia_pneumotorax_...    │
└────────────────────────────────────────────────────────────────┘
```

## 📁 Estrutura do Projeto

```
radiologia_pneumotorax_multiexame/
├── README.md                           # Este arquivo
├── notebooks/
│   └── 01_processar_multiexame.py     # Pipeline completo (Bronze + Silver + Gold)
├── config/
│   ├── procedimentos_busca.csv        # 48 códigos de procedimentos para busca inicial
│   └── procedimentos_rx_torax.csv     # 19 códigos de RX de tórax
└── outputs/
    └── (vazio - dados salvos em Delta Lake)
```

## 🔍 Procedimentos Monitorados

### Busca Inicial (48 códigos)

Procedimentos que podem documentar pneumotórax:

| Categoria | Exemplos |
|-----------|----------|
| **Tomografias** | TC de Tórax, AngioTC Arterial/Venosa |
| **Ultrassonografias** | US Tórax Extra-cardíaco, USG com Doppler |
| **Ressonâncias** | RM de Tórax, Angio-RM |
| **Procedimentos** | Drenagem de Pneumotórax, Pneumotórax Artificial |
| **Broncoscopias** | Broncografia por Hemitórax |
| **Punções** | Punção Biópsia Aspirativa de Estruturas Profundas |

Ver lista completa em [`config/procedimentos_busca.csv`](config/procedimentos_busca.csv)

### RX de Tórax para Dataset Final (19 códigos)

| Código | Nome |
|--------|------|
| 32050038 | RAIOX TORAX P.A |
| 32050054 | RAIOX TORAX: P.A - LAT |
| 40805026 | TORAX - 2 INCIDENCIAS |
| 40805034 | RX TORAX - 3 INCIDENCIAS |
| ... | (15 códigos adicionais) |

Ver lista completa em [`config/procedimentos_rx_torax.csv`](config/procedimentos_rx_torax.csv)

## 🚀 Como Executar

### Pré-requisitos

- Acesso ao Databricks Workspace da Hapvida
- Permissões para:
  - RAWZN (Lake HSP e PSC)
  - Catalog `innovation_dev` (bronze/silver/gold)
  - Databricks Foundation Models

### Execução

1. **Abrir notebook no Databricks:**
   ```
   /Workspace/Innovation/t_eduardo.caminha/radiologia_pneumotorax_multiexame/notebooks/01_processar_multiexame.py
   ```

2. **Executar células sequencialmente:**
   - Seções 1-3: BRONZE (busca inicial)
   - Seção 4: SILVER (validação LLM)
   - Seção 5: GOLD (RX de tórax)
   - Seção 6: Estatísticas

3. **Monitoramento:**
   - Progresso visual com `tqdm`
   - Logs detalhados por mês/fonte
   - Pode deixar rodando e voltar depois

### Tempo Estimado

- **Bronze:** ~2-4 horas (300 meses × HSP/PSC)
- **Silver:** ~0.2s por registro (depende do volume)
- **Gold:** ~30 minutos

**Total:** 3-5 horas para 25 anos de dados

## 📊 Tabelas Delta Lake

### Bronze: `innovation_dev.bronze.radiologia_pneumotorax_multiexame_laudos`

Laudos com menção a "pneumotórax" (antes da validação).

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| CD_PACIENTE | Long | Código do paciente |
| CD_ATENDIMENTO | Long | Código do atendimento |
| CD_OCORRENCIA | Long | Código da ocorrência |
| CD_ORDEM | Long | Código da ordem |
| CD_PROCEDIMENTO | Long | Código do procedimento realizado |
| NM_PROCEDIMENTO | String | Nome do procedimento |
| DS_LAUDO_MEDICO | String | Laudo completo (convertido de CLOB) |
| TRECHO_PNEUMOT | String | Trechos com "pneumot" (múltiplos separados por `;`) |
| DT_PROCEDIMENTO_REALIZADO | Date | Data do procedimento |
| FONTE | String | HSP ou PSC |
| DT_PROCESSAMENTO | Timestamp | Data/hora do processamento |

### Silver: `innovation_dev.silver.radiologia_pneumotorax_multiexame_validado`

Bronze + validação LLM.

| Colunas Adicionais | Tipo | Descrição |
|-------------------|------|-----------|
| INF_LLM | String | SIM (confirmado), NAO (negado/inconclusivo), ERRO |
| TEMPO_LLM_S | Double | Tempo de resposta do LLM em segundos |

### Gold: `innovation_dev.gold.radiologia_pneumotorax_multiexame_rx_torax`

RX de tórax dos casos validados positivamente.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| CD_PACIENTE | Long | Código do paciente |
| CD_ATENDIMENTO | Long | Código do atendimento |
| CD_OCORRENCIA | Long | Código da ocorrência |
| CD_ORDEM | Long | Código da ordem |
| ACC_NUM | String | Accession Number (concatenação sem separadores) |
| CD_PROCEDIMENTO | Long | Código do RX de tórax |
| NM_PROCEDIMENTO | String | Nome do RX |
| DS_LAUDO_MEDICO | String | Laudo do RX (ou "SEM LAUDO") |
| CD_MOTIVO_ATENDIMENTO | Long | 1 (Urgência) ou 2 (Eletivo) |
| TIPO_ATENDIMENTO | String | URGENCIA ou ELETIVO |
| FONTE | String | HSP ou PSC |
| DT_PROCESSAMENTO | Timestamp | Data/hora do processamento |

## 🤖 Validação com LLM

### Configuração

- **Modelo:** Llama 3.1 8B (Databricks Foundation Models)
- **Endpoint:** `databricks-meta-llama-3-1-8b-instruct`
- **Temperature:** 0.1 (determinístico)
- **Max Tokens:** 10 (resposta curta: "SIM" ou "NAO")

### Prompt

```
Você é um assistante médico. Analise o trecho abaixo de um laudo médico.
Responda APENAS "SIM" se o trecho indica presença de pneumotórax (com >50% de confiança).
Responda "NAO" se nega pneumotórax, é inconclusivo, ou menciona apenas risco/possibilidade.

Trecho: [TRECHO_EXTRAÍDO]

Resposta (SIM ou NAO):
```

### Lógica de Classificação

- **SIM:** Pneumotórax confirmado presente
- **NAO:** 
  - Nega pneumotórax ("ausência de pneumotórax")
  - Inconclusivo
  - Menciona apenas possibilidade/risco
  - Contexto não claro

## 🔧 Detalhes Técnicos

### Tratamento de CLOB

`DS_LAUDO_MEDICO` pode ser CLOB/BLOB no Oracle. Usamos:

```sql
CAST(DS_LAUDO_MEDICO AS VARCHAR(32000)) AS DS_LAUDO_MEDICO
```

### Detecção de "Pneumotórax"

Padrões normalizados (tolerância a erros):
- `PNEUMOT`
- `PNEUMO`
- `PENUMOT`
- `PNEMOT`

**Nota:** Não inclui `PNEUMATO` (evita "pneumatocele")

### Extração de Trechos

Para cada match:
- 30 caracteres antes do termo
- Termo encontrado
- 30 caracteres depois do termo
- Múltiplos matches concatenados com `;`

### Batch Processing

- **Query de procedimentos:** Por mês (evita timeout)
- **Query de laudos:** Blocos de 500 registros
- **Query de CD_PACIENTE:** Blocos de 500 atendimentos
- **Validação LLM:** Registro a registro (API síncrona)

### Fontes de Dados

Pipeline processa **HSP** e **PSC** separadamente:
- `RAWZN.RAW_HSP_TB_PROCEDIMENTO_REALIZADO`
- `RAWZN.RAW_PSC_TB_PROCEDIMENTO_REALIZADO`
- `RAWZN.RAW_HSP_TB_LAUDO_PACIENTE`
- `RAWZN.RAW_PSC_TB_LAUDO_PACIENTE`
- `RAWZN.RAW_HSP_TM_ATENDIMENTO`
- `RAWZN.RAW_PSC_TM_ATENDIMENTO`

Coluna `FONTE` identifica origem de cada registro.

## 📈 Casos de Uso

### 1. Auditoria de RX sem Laudo (Urgência)

```sql
SELECT *
FROM innovation_dev.gold.radiologia_pneumotorax_multiexame_rx_torax
WHERE DS_LAUDO_MEDICO = 'SEM LAUDO'
  AND TIPO_ATENDIMENTO = 'URGENCIA'
ORDER BY CD_ATENDIMENTO DESC
```

→ Lista de RX de tórax de urgência com alta probabilidade de pneumotórax que precisam ser auditados.

### 2. Validação de Acurácia do LLM

```sql
SELECT 
    INF_LLM,
    COUNT(*) as TOTAL,
    AVG(TEMPO_LLM_S) as TEMPO_MEDIO_S
FROM innovation_dev.silver.radiologia_pneumotorax_multiexame_validado
GROUP BY INF_LLM
```

### 3. Análise Temporal

```sql
SELECT 
    YEAR(DT_PROCEDIMENTO_REALIZADO) as ANO,
    COUNT(*) as TOTAL_CASOS
FROM innovation_dev.bronze.radiologia_pneumotorax_multiexame_laudos
GROUP BY ANO
ORDER BY ANO
```

### 4. Tipos de Exames Mais Relevantes

```sql
SELECT 
    NM_PROCEDIMENTO,
    COUNT(*) as TOTAL,
    SUM(CASE WHEN INF_LLM = 'SIM' THEN 1 ELSE 0 END) as CONFIRMADOS
FROM innovation_dev.silver.radiologia_pneumotorax_multiexame_validado
GROUP BY NM_PROCEDIMENTO
ORDER BY CONFIRMADOS DESC
```

### 5. Distribuição por Tipo de Atendimento

```sql
SELECT 
    TIPO_ATENDIMENTO,
    COUNT(*) as TOTAL_RX,
    SUM(CASE WHEN DS_LAUDO_MEDICO = 'SEM LAUDO' THEN 1 ELSE 0 END) as SEM_LAUDO,
    SUM(CASE WHEN DS_LAUDO_MEDICO != 'SEM LAUDO' THEN 1 ELSE 0 END) as COM_LAUDO
FROM innovation_dev.gold.radiologia_pneumotorax_multiexame_rx_torax
GROUP BY TIPO_ATENDIMENTO
```

→ Analisa distribuição entre urgência/eletivo e quantos têm laudo.

## ⚠️ Considerações Importantes

### Limitações

1. **Dependência de Laudo Textual:** Apenas exames com `DS_LAUDO_MEDICO` preenchido
2. **Sensibilidade LLM:** Configurado para >50% confiança (pode ter falsos negativos)
3. **Período Fixo:** 2000-2025 (ajustar variáveis `ano_inicio`/`ano_fim` se necessário)
4. **Performance:** ~0.2s por registro no LLM (pode acumular em volumes grandes)

### Boas Práticas

- **Primeira Execução:** Testar com período menor (ex: 1 ano) para validar
- **Monitoramento:** Acompanhar logs e `tqdm` durante execução
- **Re-execução:** Bronze usa `mode("append")` - considerar limpeza antes
- **Silver/Gold:** Usam `mode("overwrite")` - podem ser re-processados

### Manutenção

Para adicionar novos procedimentos:

1. Editar [`config/procedimentos_busca.csv`](config/procedimentos_busca.csv) ou [`config/procedimentos_rx_torax.csv`](config/procedimentos_rx_torax.csv)
2. Re-executar pipeline (Bronze usa append, então pode duplicar se não limpar antes)

## 📞 Contato

**Projeto:** Radiologia - Detecção de Pneumotórax  
**Owner:** Eduardo Caminha  
**Workspace:** `/Workspace/Innovation/t_eduardo.caminha/radiologia_pneumotorax_multiexame/`

## 📝 Changelog

### v1.0.0 (2025-01-12)
- Pipeline inicial completo (Bronze → Silver → Gold)
- Validação com Llama 3.1 8B
- Suporte a HSP e PSC
- 48 procedimentos de busca + 19 RX de tórax
- Período: 2000-2025

