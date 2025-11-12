# Databricks notebook source
# MAGIC %md
# MAGIC # Detecção de Pneumotórax em Múltiplos Tipos de Exames
# MAGIC 
# MAGIC Pipeline completo para detectar pneumotórax em exames variados (TC, US, RM, etc)
# MAGIC entre 2000-2025, validar com LLM e buscar RX de tórax relacionados.
# MAGIC 
# MAGIC **Etapas:**
# MAGIC 1. Bronze: Buscar laudos com "pneumot" em 48 tipos de procedimentos
# MAGIC 2. Silver: Validar com LLM (Llama 3.1 8B) se é pneumotórax real
# MAGIC 3. Gold: Buscar RX de tórax dos casos positivos

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Conexão com Datalake

# COMMAND ----------

# MAGIC %run /Workspace/Libraries/Lake

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import re
from tqdm import tqdm

# Conectar ao datalake
connect_to_datalake(
    username="USR_PROD_INFORMATICA_SAUDE",
    password=dbutils.secrets.get(scope="INNOVATION_RAW", key="USR_PROD_INFORMATICA_SAUDE"),
    layer="RAWZN",
    level="LOW",
    dbx_secret_scope="INNOVATION_RAW" 
)

print("✅ Conexão estabelecida!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuração e Tabelas de Procedimentos

# COMMAND ----------

# Carregar procedimentos de busca (48 códigos)
df_proc_busca = pd.read_csv("/Workspace/Innovation/t_eduardo.caminha/radiologia-pneumotorax-multiexame/config/procedimentos_busca.csv")
codigos_busca = df_proc_busca['CD_PROCEDIMENTO'].tolist()
dict_proc_busca = dict(zip(df_proc_busca['CD_PROCEDIMENTO'], df_proc_busca['NM_PROCEDIMENTO']))

print(f"📋 Procedimentos de busca: {len(codigos_busca)} códigos")
print(f"   Exemplos: {codigos_busca[:5]}")

# Carregar procedimentos de RX tórax (19 códigos)
df_proc_rx = pd.read_csv("/Workspace/Innovation/t_eduardo.caminha/radiologia_pneumotorax_multiexame/config/procedimentos_rx_torax.csv")
codigos_rx_torax = df_proc_rx['CD_PROCEDIMENTO'].tolist()
dict_proc_rx = dict(zip(df_proc_rx['CD_PROCEDIMENTO'], df_proc_rx['NM_PROCEDIMENTO']))

print(f"📋 Procedimentos RX tórax: {len(codigos_rx_torax)} códigos")
print(f"   Exemplos: {codigos_rx_torax[:5]}")

# Configuração das tabelas Delta
TABELA_BRONZE = "innovation_dev.bronze.radiologia_pneumotorax_multiexame_laudos"
TABELA_SILVER = "innovation_dev.silver.radiologia_pneumotorax_multiexame_validado"
TABELA_GOLD = "innovation_dev.gold.radiologia_pneumotorax_multiexame_rx_torax"

print(f"\n📊 Tabelas Delta:")
print(f"   Bronze: {TABELA_BRONZE}")
print(f"   Silver: {TABELA_SILVER}")
print(f"   Gold: {TABELA_GOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ETAPA BRONZE - Buscar Laudos com Pneumotórax

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Função para Detectar Pneumotórax

# COMMAND ----------

def detectar_pneumot(texto):
    """Detecta pneumotórax com tolerância a erros e retorna trechos"""
    if pd.isna(texto) or texto is None or str(texto).strip() == '':
        return False, None
    
    texto_str = str(texto)
    texto_upper = texto_str.upper()
    
    # Padrões variantes de pneumotórax
    padroes = ['PNEUMOT', 'PNEUMO', 'PENUMOT', 'PNEMOT']
    
    trechos_encontrados = []
    
    for padrao in padroes:
        idx = 0
        while True:
            idx = texto_upper.find(padrao, idx)
            if idx == -1:
                break
            
            # Extrair 30 chars antes e depois (usar if/else ao invés de max/min)
            inicio = 0 if idx - 30 < 0 else idx - 30
            fim_calc = idx + len(padrao) + 30
            fim = len(texto_str) if fim_calc > len(texto_str) else fim_calc
            trecho = texto_str[inicio:fim].strip()
            
            # Limpar espaços múltiplos
            trecho = ' '.join(trecho.split())
            
            if trecho not in trechos_encontrados:
                trechos_encontrados.append(trecho)
            
            idx += len(padrao)
    
    if trechos_encontrados:
        return True, '; '.join(trechos_encontrados)
    
    return False, None

print("✅ Função detectar_pneumot() criada")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Processar Meses (2000-2025)

# COMMAND ----------

# Gerar lista de meses
ano_inicio = 2000
ano_fim = 2025
mes_fim_ano = 12

meses = []
data_atual = datetime(ano_inicio, 1, 1)
data_fim = datetime(ano_fim, mes_fim_ano, 1)

while data_atual <= data_fim:
    proximo_mes = data_atual + relativedelta(months=1)
    meses.append((
        data_atual.strftime('%Y-%m-%d'),
        proximo_mes.strftime('%Y-%m-%d'),
        data_atual.strftime('%Y-%m')
    ))
    data_atual = proximo_mes

print(f"📅 Total de meses para processar: {len(meses)}")
print(f"   Período: {meses[0][2]} até {meses[-1][2]}")

# COMMAND ----------

# Lista para acumular resultados antes de salvar no Delta
dados_acumulados = []
total_procedimentos = 0
total_com_pneumot = 0

# Processar cada mês
for i, (mes_inicio, mes_fim, mes_label) in enumerate(tqdm(meses, desc="Processando meses")):
    
    # Processar HSP
    for fonte in ['HSP', 'PSC']:
        try:
            # Query simples: buscar procedimentos do mês
            query_proc = f"""
            SELECT 
                CD_ATENDIMENTO,
                CD_OCORRENCIA,
                CD_ORDEM,
                CD_PROCEDIMENTO,
                DT_PROCEDIMENTO_REALIZADO
            FROM RAWZN.RAW_{fonte}_TB_PROCEDIMENTO_REALIZADO
            WHERE CD_PROCEDIMENTO IN ({','.join(map(str, codigos_busca))})
              AND DT_PROCEDIMENTO_REALIZADO >= DATE '{mes_inicio}'
              AND DT_PROCEDIMENTO_REALIZADO < DATE '{mes_fim}'
            """
            
            df_proc = run_sql(query_proc)
            
            if len(df_proc) == 0:
                continue
            
            df_proc = pd.DataFrame(df_proc)
            total_procedimentos += len(df_proc)
            
            # Buscar laudos em batch de 500
            lista_chaves = list(zip(
                df_proc['CD_ATENDIMENTO'],
                df_proc['CD_OCORRENCIA'],
                df_proc['CD_ORDEM']
            ))
            
            laudos_dict = {}
            
            for inicio in range(0, len(lista_chaves), 500):
                chunk = lista_chaves[inicio:inicio + 500]
                
                # Montar condições WHERE
                condicoes = ' OR '.join([
                    f"(CD_ATENDIMENTO = {atend} AND CD_OCORRENCIA = {ocorr} AND CD_ORDEM = {ordem})"
                    for atend, ocorr, ordem in chunk
                ])
                
                query_laudos = f"""
                SELECT 
                    CD_ATENDIMENTO,
                    CD_OCORRENCIA,
                    CD_ORDEM,
                    DS_LAUDO_MEDICO
                FROM RAWZN.RAW_{fonte}_TB_LAUDO_PACIENTE
                WHERE {condicoes}
                """
                
                df_laudos_chunk = run_sql(query_laudos)
                
                if len(df_laudos_chunk) > 0:
                    df_laudos_chunk = pd.DataFrame(df_laudos_chunk)
                    for _, row in df_laudos_chunk.iterrows():
                        chave = (row['CD_ATENDIMENTO'], row['CD_OCORRENCIA'], row['CD_ORDEM'])
                        # Converter LONG para string em Python
                        laudo = row['DS_LAUDO_MEDICO']
                        if laudo is not None:
                            laudo = str(laudo)
                        laudos_dict[chave] = laudo
            
            # Juntar laudos com procedimentos
            df_proc['DS_LAUDO_MEDICO'] = df_proc.apply(
                lambda row: laudos_dict.get((row['CD_ATENDIMENTO'], row['CD_OCORRENCIA'], row['CD_ORDEM']), None),
                axis=1
            )
            
            # Filtrar apenas com laudo
            df_proc = df_proc[df_proc['DS_LAUDO_MEDICO'].notna()].copy()
            
            if len(df_proc) == 0:
                continue
            
            # Detectar pneumotórax
            resultado_deteccao = df_proc['DS_LAUDO_MEDICO'].apply(detectar_pneumot)
            df_proc['TEM_PNEUMOT'] = resultado_deteccao.apply(lambda x: x[0])
            df_proc['TRECHO_PNEUMOT'] = resultado_deteccao.apply(lambda x: x[1])
            
            # Filtrar apenas com pneumot
            df_pneumot = df_proc[df_proc['TEM_PNEUMOT'] == True].copy()
            
            if len(df_pneumot) == 0:
                continue
            
            total_com_pneumot += len(df_pneumot)
            
            # Buscar CD_PACIENTE em batch de 500
            cd_atendimentos = df_pneumot['CD_ATENDIMENTO'].dropna().astype(int).unique().tolist()
            
            resultado_pacientes = []
            for inicio in range(0, len(cd_atendimentos), 500):
                chunk = cd_atendimentos[inicio:inicio + 500]
                valores_in = ', '.join(str(x) for x in chunk)
                
                query_pac = f"""
                SELECT CD_ATENDIMENTO, CD_PACIENTE
                FROM RAWZN.RAW_{fonte}_TM_ATENDIMENTO
                WHERE CD_ATENDIMENTO IN ({valores_in})
                """
                
                resultado = run_sql(query_pac)
                resultado_pacientes.append(pd.DataFrame(resultado))
            
            if resultado_pacientes:
                df_pacientes = pd.concat(resultado_pacientes, ignore_index=True)
                df_pacientes.columns = [str(col).strip().upper() for col in df_pacientes.columns]
                
                # Garantir que tem as colunas certas
                if len(df_pacientes.columns) >= 2:
                    df_pacientes.columns = ['CD_ATENDIMENTO', 'CD_PACIENTE']
                
                df_pacientes['CD_ATENDIMENTO'] = pd.to_numeric(df_pacientes['CD_ATENDIMENTO'], errors='coerce')
                df_pacientes['CD_PACIENTE'] = pd.to_numeric(df_pacientes['CD_PACIENTE'], errors='coerce')
                
                # Merge
                df_pneumot['CD_ATENDIMENTO'] = pd.to_numeric(df_pneumot['CD_ATENDIMENTO'], errors='coerce')
                df_pneumot = df_pneumot.merge(df_pacientes, on='CD_ATENDIMENTO', how='left')
            else:
                df_pneumot['CD_PACIENTE'] = None
            
            # Adicionar NM_PROCEDIMENTO
            df_pneumot['NM_PROCEDIMENTO'] = df_pneumot['CD_PROCEDIMENTO'].map(dict_proc_busca)
            
            # Adicionar FONTE e timestamp
            df_pneumot['FONTE'] = fonte
            df_pneumot['DT_PROCESSAMENTO'] = datetime.now()
            
            # Remover coluna auxiliar
            df_pneumot = df_pneumot.drop(columns=['TEM_PNEUMOT'])
            
            # Acumular
            dados_acumulados.append(df_pneumot)
            
        except Exception as e:
            print(f"\n⚠️ Erro ao processar {fonte} {mes_label}: {e}")
            continue

print(f"\n{'='*60}")
print(f"📊 RESUMO PROCESSAMENTO BRONZE")
print(f"{'='*60}")
print(f"Total de procedimentos encontrados: {total_procedimentos}")
print(f"Total com pneumotórax detectado: {total_com_pneumot}")
print(f"Chunks acumulados: {len(dados_acumulados)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3. Verificar Tabela Bronze Existente

# COMMAND ----------

# Verificar se tabela já existe e quantos registros tem
try:
    df_bronze_existente = spark.table(TABELA_BRONZE)
    total_existente = df_bronze_existente.count()
    print(f"⚠️  Tabela Bronze já existe com {total_existente} registros!")
    print(f"   Se continuar, serão adicionados mais {len(dados_acumulados)} chunks (mode='append')")
    print(f"   Para evitar duplicação, considere DROP TABLE ou use mode='overwrite'")
    
    # Mostrar últimos registros processados
    from pyspark.sql.functions import max as spark_max, count as spark_count
    print(f"\n📅 Últimas datas processadas na tabela existente:")
    display(df_bronze_existente.groupBy('FONTE').agg(
        spark_max('DT_PROCESSAMENTO').alias('ULTIMO_PROCESSAMENTO'),
        spark_count('*').alias('TOTAL_REGISTROS')
    ))
except Exception as e:
    print(f"✅ Tabela Bronze não existe ainda. Será criada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4. Salvar Tabela Bronze no Delta Lake

# COMMAND ----------

if dados_acumulados:
    # Concatenar todos os DataFrames
    df_bronze_pd = pd.concat(dados_acumulados, ignore_index=True)
    
    # Reordenar e selecionar colunas (remover NM_PROCEDIMENTO que está causando problema)
    colunas_bronze = [
        'CD_PACIENTE', 'CD_ATENDIMENTO', 'CD_OCORRENCIA', 'CD_ORDEM',
        'CD_PROCEDIMENTO', 'DS_LAUDO_MEDICO',
        'TRECHO_PNEUMOT', 'DT_PROCEDIMENTO_REALIZADO', 'FONTE', 'DT_PROCESSAMENTO'
    ]
    df_bronze_pd = df_bronze_pd[colunas_bronze]
    
    print(f"📊 Total de registros para salvar: {len(df_bronze_pd)}")
    print(f"📋 Colunas: {list(df_bronze_pd.columns)}")
    
    # Converter para Spark DataFrame
    df_bronze_spark = spark.createDataFrame(df_bronze_pd)
    
    # Salvar no Delta Lake (append para acumular se rodar novamente)
    df_bronze_spark.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable(TABELA_BRONZE)
    
    print(f"✅ Tabela Bronze salva: {TABELA_BRONZE}")
    print(f"   Registros salvos: {len(df_bronze_pd)}")
    
    # Mostrar amostra
    display(spark.table(TABELA_BRONZE).limit(10))
else:
    print("⚠️ Nenhum registro para salvar na Bronze!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ETAPA SILVER - Validação com LLM

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1. Configuração do LLM

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import time

# Configuração
ENDPOINT_NAME = "databricks-meta-llama-3-1-8b-instruct"
TEMPERATURE = 0.1
MAX_TOKENS = 10

print(f"🤖 LLM Configurado:")
print(f"   Endpoint: {ENDPOINT_NAME}")
print(f"   Temperature: {TEMPERATURE}")
print(f"   Max tokens: {MAX_TOKENS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2. Função de Validação LLM

# COMMAND ----------

def validar_pneumot_llm(trecho: str) -> tuple:
    """
    Valida se trecho indica pneumotórax real usando LLM
    
    Returns:
        tuple: (resposta_sim_nao, tempo_segundos)
    """
    if pd.isna(trecho) or str(trecho).strip() == '':
        return 'NAO', 0.0
    
    w = WorkspaceClient()
    
    prompt = f"""Você é um assistente médico. Analise o trecho abaixo de um laudo médico.
Responda APENAS "SIM" se o trecho indica presença de pneumotórax (com >50% de confiança).
Responda "NAO" se nega pneumotórax, é inconclusivo, ou menciona apenas risco/possibilidade.

Trecho: {trecho}

Resposta (SIM ou NAO):"""
    
    messages = [
        ChatMessage(
            role=ChatMessageRole.SYSTEM,
            content="Você é um assistente médico especializado. Responda apenas SIM ou NAO."
        ),
        ChatMessage(
            role=ChatMessageRole.USER,
            content=prompt
        )
    ]
    
    start_time = time.time()
    
    try:
        response = w.serving_endpoints.query(
            name=ENDPOINT_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        elapsed = time.time() - start_time
        response_text = response.choices[0].message.content.strip().upper()
        
        # Extrair SIM ou NAO
        if 'SIM' in response_text:
            return 'SIM', elapsed
        else:
            return 'NAO', elapsed
            
    except Exception as e:
        print(f"⚠️ Erro ao chamar LLM: {e}")
        return 'ERRO', time.time() - start_time

print("✅ Função validar_pneumot_llm() criada")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3. Processar Bronze e Validar com LLM

# COMMAND ----------

# Ler tabela Bronze
df_bronze_spark = spark.table(TABELA_BRONZE)
df_bronze_pd = df_bronze_spark.toPandas()

print(f"📊 Registros na Bronze: {len(df_bronze_pd)}")

# Validar cada registro
resultados_llm = []
tempos_llm = []

for _, row in tqdm(df_bronze_pd.iterrows(), total=len(df_bronze_pd), desc="Validando com LLM"):
    resposta, tempo = validar_pneumot_llm(row['TRECHO_PNEUMOT'])
    resultados_llm.append(resposta)
    tempos_llm.append(tempo)

df_bronze_pd['INF_LLM'] = resultados_llm
df_bronze_pd['TEMPO_LLM_S'] = tempos_llm

print(f"\n📊 Resultados LLM:")
print(f"   SIM: {sum(1 for x in resultados_llm if x == 'SIM')}")
print(f"   NAO: {sum(1 for x in resultados_llm if x == 'NAO')}")
print(f"   ERRO: {sum(1 for x in resultados_llm if x == 'ERRO')}")
print(f"   Tempo médio: {sum(tempos_llm) / len(tempos_llm):.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4. Salvar Tabela Silver no Delta Lake

# COMMAND ----------

# Converter para Spark
df_silver_spark = spark.createDataFrame(df_bronze_pd)

# Salvar no Delta Lake
df_silver_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(TABELA_SILVER)

print(f"✅ Tabela Silver salva: {TABELA_SILVER}")
print(f"   Registros salvos: {len(df_bronze_pd)}")

# Mostrar amostra
display(spark.table(TABELA_SILVER).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ETAPA GOLD - Buscar RX de Tórax

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1. Filtrar Casos Positivos (SIM)

# COMMAND ----------

# Ler Silver
df_silver_spark = spark.table(TABELA_SILVER)
df_silver_pd = df_silver_spark.toPandas()

# Filtrar apenas SIM
df_positivos = df_silver_pd[df_silver_pd['INF_LLM'] == 'SIM'].copy()
cd_atendimentos_positivos = df_positivos['CD_ATENDIMENTO'].dropna().astype(int).unique().tolist()

print(f"📊 Casos positivos (SIM): {len(df_positivos)}")
print(f"📊 Atendimentos únicos: {len(cd_atendimentos_positivos)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2. Buscar RX de Tórax dos Casos Positivos

# COMMAND ----------

# Lista para acumular RX encontrados
rx_encontrados = []

# Processar em batch de 500
for fonte in ['HSP', 'PSC']:
    for inicio in range(0, len(cd_atendimentos_positivos), 500):
        chunk = cd_atendimentos_positivos[inicio:inicio + 500]
        valores_in = ', '.join(str(x) for x in chunk)
        
        try:
            # Buscar RX de tórax
            query_rx = f"""
            SELECT 
                CD_ATENDIMENTO,
                CD_OCORRENCIA,
                CD_ORDEM,
                CD_PROCEDIMENTO
            FROM RAWZN.RAW_{fonte}_TB_PROCEDIMENTO_REALIZADO
            WHERE CD_ATENDIMENTO IN ({valores_in})
              AND CD_PROCEDIMENTO IN ({','.join(map(str, codigos_rx_torax))})
            """
            
            df_rx = run_sql(query_rx)
            
            if len(df_rx) > 0:
                df_rx = pd.DataFrame(df_rx)
                df_rx['FONTE'] = fonte
                rx_encontrados.append(df_rx)
                
        except Exception as e:
            print(f"⚠️ Erro ao buscar RX {fonte}: {e}")
            continue

if rx_encontrados:
    df_rx_todos = pd.concat(rx_encontrados, ignore_index=True)
    print(f"✅ Total de RX de tórax encontrados: {len(df_rx_todos)}")
else:
    print("⚠️ Nenhum RX de tórax encontrado!")
    df_rx_todos = pd.DataFrame(columns=['CD_ATENDIMENTO', 'CD_OCORRENCIA', 'CD_ORDEM', 'CD_PROCEDIMENTO', 'FONTE'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3. Buscar Laudos dos RX

# COMMAND ----------

if len(df_rx_todos) > 0:
    # Buscar laudos em batch
    lista_chaves_rx = list(zip(
        df_rx_todos['CD_ATENDIMENTO'],
        df_rx_todos['CD_OCORRENCIA'],
        df_rx_todos['CD_ORDEM'],
        df_rx_todos['FONTE']
    ))
    
    laudos_rx_dict = {}
    
    # Agrupar por fonte
    for fonte in ['HSP', 'PSC']:
        chaves_fonte = [(a, o, ord) for a, o, ord, f in lista_chaves_rx if f == fonte]
        
        if not chaves_fonte:
            continue
        
        for inicio in range(0, len(chaves_fonte), 500):
            chunk = chaves_fonte[inicio:inicio + 500]
            
            condicoes = ' OR '.join([
                f"(CD_ATENDIMENTO = {atend} AND CD_OCORRENCIA = {ocorr} AND CD_ORDEM = {ordem})"
                for atend, ocorr, ordem in chunk
            ])
            
            try:
                query_laudos_rx = f"""
                SELECT 
                    CD_ATENDIMENTO,
                    CD_OCORRENCIA,
                    CD_ORDEM,
                    DS_LAUDO_MEDICO
                FROM RAWZN.RAW_{fonte}_TB_LAUDO_PACIENTE
                WHERE {condicoes}
                """
                
                df_laudos_rx = run_sql(query_laudos_rx)
                
                if len(df_laudos_rx) > 0:
                    df_laudos_rx = pd.DataFrame(df_laudos_rx)
                    for _, row in df_laudos_rx.iterrows():
                        chave = (row['CD_ATENDIMENTO'], row['CD_OCORRENCIA'], row['CD_ORDEM'])
                        # Converter LONG para string em Python
                        laudo = row['DS_LAUDO_MEDICO']
                        if laudo is not None:
                            laudo = str(laudo)
                        laudos_rx_dict[chave] = laudo
                        
            except Exception as e:
                print(f"⚠️ Erro ao buscar laudos RX {fonte}: {e}")
                continue
    
    # Adicionar laudos ao DataFrame
    df_rx_todos['DS_LAUDO_MEDICO'] = df_rx_todos.apply(
        lambda row: laudos_rx_dict.get((row['CD_ATENDIMENTO'], row['CD_OCORRENCIA'], row['CD_ORDEM']), None),
        axis=1
    )
    
    # Substituir None por "SEM LAUDO"
    df_rx_todos['DS_LAUDO_MEDICO'] = df_rx_todos['DS_LAUDO_MEDICO'].fillna('SEM LAUDO')
    
    print(f"✅ Laudos anexados")
    print(f"   Com laudo: {sum(df_rx_todos['DS_LAUDO_MEDICO'] != 'SEM LAUDO')}")
    print(f"   Sem laudo: {sum(df_rx_todos['DS_LAUDO_MEDICO'] == 'SEM LAUDO')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4. Criar ACCESSION_NUMBER e Join CD_PACIENTE

# COMMAND ----------

if len(df_rx_todos) > 0:
    # Criar ACC_NUM
    df_rx_todos['ACC_NUM'] = (
        df_rx_todos['CD_ATENDIMENTO'].astype(str) +
        df_rx_todos['CD_OCORRENCIA'].astype(str) +
        df_rx_todos['CD_ORDEM'].astype(str)
    )
    
    # Adicionar NM_PROCEDIMENTO
    df_rx_todos['NM_PROCEDIMENTO'] = df_rx_todos['CD_PROCEDIMENTO'].map(dict_proc_rx)
    
    # Join com CD_PACIENTE da Silver
    df_pacientes_silver = df_positivos[['CD_ATENDIMENTO', 'CD_PACIENTE']].drop_duplicates()
    df_pacientes_silver['CD_ATENDIMENTO'] = pd.to_numeric(df_pacientes_silver['CD_ATENDIMENTO'], errors='coerce')
    
    df_rx_todos['CD_ATENDIMENTO'] = pd.to_numeric(df_rx_todos['CD_ATENDIMENTO'], errors='coerce')
    df_rx_todos = df_rx_todos.merge(df_pacientes_silver, on='CD_ATENDIMENTO', how='left')
    
    # Buscar CD_MOTIVO_ATENDIMENTO (1-URGENCIA, 2-ELETIVO)
    cd_atendimentos_rx = df_rx_todos['CD_ATENDIMENTO'].dropna().astype(int).unique().tolist()
    
    motivo_atend_dict = {}
    
    for fonte in ['HSP', 'PSC']:
        atends_fonte = df_rx_todos[df_rx_todos['FONTE'] == fonte]['CD_ATENDIMENTO'].dropna().astype(int).unique().tolist()
        
        if not atends_fonte:
            continue
        
        for inicio in range(0, len(atends_fonte), 500):
            chunk = atends_fonte[inicio:inicio + 500]
            valores_in = ', '.join(str(x) for x in chunk)
            
            try:
                query_motivo = f"""
                SELECT 
                    CD_ATENDIMENTO,
                    CD_MOTIVO_ATENDIMENTO
                FROM RAWZN.RAW_{fonte}_TM_ATENDIMENTO
                WHERE CD_ATENDIMENTO IN ({valores_in})
                """
                
                df_motivo = run_sql(query_motivo)
                
                if len(df_motivo) > 0:
                    df_motivo = pd.DataFrame(df_motivo)
                    for _, row in df_motivo.iterrows():
                        motivo_atend_dict[int(row['CD_ATENDIMENTO'])] = row['CD_MOTIVO_ATENDIMENTO']
                        
            except Exception as e:
                print(f"⚠️ Erro ao buscar motivo atendimento {fonte}: {e}")
                continue
    
    # Adicionar CD_MOTIVO_ATENDIMENTO ao DataFrame
    df_rx_todos['CD_MOTIVO_ATENDIMENTO'] = df_rx_todos['CD_ATENDIMENTO'].map(motivo_atend_dict)
    
    # Mapear para texto legível
    df_rx_todos['TIPO_ATENDIMENTO'] = df_rx_todos['CD_MOTIVO_ATENDIMENTO'].map({
        1: 'URGENCIA',
        2: 'ELETIVO'
    })
    
    # Adicionar timestamp
    df_rx_todos['DT_PROCESSAMENTO'] = datetime.now()
    
    print(f"✅ Metadados adicionados")
    print(f"   Urgência: {sum(df_rx_todos['TIPO_ATENDIMENTO'] == 'URGENCIA')}")
    print(f"   Eletivo: {sum(df_rx_todos['TIPO_ATENDIMENTO'] == 'ELETIVO')}")
    print(f"   Sem info: {df_rx_todos['TIPO_ATENDIMENTO'].isna().sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5. Salvar Tabela Gold no Delta Lake

# COMMAND ----------

if len(df_rx_todos) > 0:
    # Reordenar colunas
    colunas_gold = [
        'CD_PACIENTE', 'CD_ATENDIMENTO', 'CD_OCORRENCIA', 'CD_ORDEM',
        'ACC_NUM', 'CD_PROCEDIMENTO', 'NM_PROCEDIMENTO', 'DS_LAUDO_MEDICO',
        'CD_MOTIVO_ATENDIMENTO', 'TIPO_ATENDIMENTO', 'FONTE', 'DT_PROCESSAMENTO'
    ]
    df_gold_pd = df_rx_todos[colunas_gold]
    
    print(f"📊 Total de registros Gold: {len(df_gold_pd)}")
    
    # Converter para Spark
    df_gold_spark = spark.createDataFrame(df_gold_pd)
    
    # Salvar no Delta Lake
    df_gold_spark.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(TABELA_GOLD)
    
    print(f"✅ Tabela Gold salva: {TABELA_GOLD}")
    print(f"   Registros salvos: {len(df_gold_pd)}")
    
    # Mostrar amostra
    display(spark.table(TABELA_GOLD).limit(10))
else:
    print("⚠️ Nenhum registro para salvar na Gold!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Estatísticas Finais

# COMMAND ----------

print("="*80)
print("📊 ESTATÍSTICAS FINAIS DO PIPELINE")
print("="*80)

# Bronze
df_bronze_stats = spark.table(TABELA_BRONZE)
total_bronze = df_bronze_stats.count()
print(f"\n🥉 BRONZE: {total_bronze} registros")
print(f"   Laudos com menção a pneumotórax em {len(codigos_busca)} tipos de exames")

# Silver
df_silver_stats = spark.table(TABELA_SILVER)
total_silver = df_silver_stats.count()
sim_count = df_silver_stats.filter(col("INF_LLM") == "SIM").count()
nao_count = df_silver_stats.filter(col("INF_LLM") == "NAO").count()

print(f"\n🥈 SILVER: {total_silver} registros")
print(f"   Validados com LLM (Llama 3.1 8B)")
print(f"   - SIM (pneumotórax confirmado): {sim_count}")
print(f"   - NAO (negado/inconclusivo): {nao_count}")

# Gold
df_gold_stats = spark.table(TABELA_GOLD)
total_gold = df_gold_stats.count()
com_laudo = df_gold_stats.filter(col("DS_LAUDO_MEDICO") != "SEM LAUDO").count()
sem_laudo = df_gold_stats.filter(col("DS_LAUDO_MEDICO") == "SEM LAUDO").count()

print(f"\n🥇 GOLD: {total_gold} registros")
print(f"   RX de tórax dos casos positivos")
print(f"   - Com laudo: {com_laudo}")
print(f"   - Sem laudo (requer auditoria): {sem_laudo}")

print(f"\n{'='*80}")
print("✅ Pipeline completo executado com sucesso!")
print(f"{'='*80}")

# COMMAND ----------



