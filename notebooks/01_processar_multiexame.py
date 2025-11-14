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
df_proc_rx = pd.read_csv("/Workspace/Innovation/t_eduardo.caminha/radiologia-pneumotorax-multiexame/config/procedimentos_rx_torax.csv")
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

# MAGIC %md
# MAGIC ### 3.2.1. TESTE - Processar 1 Mês de Amostra (Validação)

# COMMAND ----------

# TESTAR COM 1 MÊS ANTES DE PROCESSAR TUDO
print("🧪 MODO TESTE: Processando apenas 1 mês para validação")
print("=" * 60)

mes_teste = meses[0]  # Primeiro mês disponível
mes_inicio_teste, mes_fim_teste, mes_label_teste = mes_teste

dados_teste = []

for fonte in ['HSP', 'PSC']:
    try:
        print(f"\n🔍 Testando {fonte} - {mes_label_teste}...")
        
        query_proc_teste = f"""
        SELECT 
            CD_ATENDIMENTO,
            CD_OCORRENCIA,
            CD_ORDEM,
            CD_PROCEDIMENTO,
            DT_PROCEDIMENTO_REALIZADO
        FROM RAWZN.RAW_{fonte}_TB_PROCEDIMENTO_REALIZADO
        WHERE CD_PROCEDIMENTO IN ({','.join(map(str, codigos_busca))})
          AND DT_PROCEDIMENTO_REALIZADO >= DATE '{mes_inicio_teste}'
          AND DT_PROCEDIMENTO_REALIZADO < DATE '{mes_fim_teste}'
          AND ROWNUM <= 10  -- APENAS 10 REGISTROS PARA TESTE
        """
        
        df_proc_teste = run_sql(query_proc_teste)
        
        if len(df_proc_teste) == 0:
            print(f"   ⏭️  Nenhum registro encontrado em {fonte}")
            continue
        
        df_proc_teste = pd.DataFrame(df_proc_teste)
        print(f"   ✅ {len(df_proc_teste)} procedimentos encontrados")
        
        # Testar busca de laudos
        lista_chaves_teste = list(zip(
            df_proc_teste['CD_ATENDIMENTO'],
            df_proc_teste['CD_OCORRENCIA'],
            df_proc_teste['CD_ORDEM']
        ))
        
        condicoes_teste = ' OR '.join([
            f"(CD_ATENDIMENTO = {atend} AND CD_OCORRENCIA = {ocorr} AND CD_ORDEM = {ordem})"
            for atend, ocorr, ordem in lista_chaves_teste[:5]  # Apenas 5
        ])
        
        query_laudos_teste = f"""
        SELECT 
            CD_ATENDIMENTO,
            CD_OCORRENCIA,
            CD_ORDEM,
            DS_LAUDO_MEDICO
        FROM RAWZN.RAW_{fonte}_TB_LAUDO_PACIENTE
        WHERE {condicoes_teste}
        """
        
        df_laudos_teste = run_sql(query_laudos_teste)
        print(f"   ✅ {len(df_laudos_teste)} laudos encontrados")
        
        # Testar detecção de pneumotórax
        if len(df_laudos_teste) > 0:
            df_laudos_teste = pd.DataFrame(df_laudos_teste)
            laudo_exemplo = str(df_laudos_teste.iloc[0]['DS_LAUDO_MEDICO']) if len(df_laudos_teste) > 0 else None
            
            if laudo_exemplo:
                tem_pneumot, trecho = detectar_pneumot(laudo_exemplo)
                print(f"   ✅ Detecção pneumot funcionando: {tem_pneumot}")
                if tem_pneumot:
                    print(f"      Trecho: {trecho[:100]}...")
        
        print(f"   ✅ Pipeline de {fonte} funcionando corretamente!")
        dados_teste.append(df_proc_teste)
        
    except Exception as e:
        print(f"   ❌ ERRO em {fonte}: {e}")
        print(f"      Corrija antes de processar todos os meses!")
        raise

print("\n" + "=" * 60)
print("✅ TESTE CONCLUÍDO COM SUCESSO!")
print("   Todas as etapas funcionam corretamente.")
print("   Pode prosseguir com processamento completo.")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2.2. Processar Todos os Meses (Produção)

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
# MAGIC ### 4.3. TESTE - Validar 3 Registros com LLM (Amostra)

# COMMAND ----------

# TESTAR LLM COM 3 REGISTROS ANTES DE PROCESSAR TUDO
print("🧪 MODO TESTE: Validando 3 registros com LLM")
print("=" * 60)

df_bronze_spark_teste = spark.table(TABELA_BRONZE)
df_bronze_pd_teste = df_bronze_spark_teste.limit(3).toPandas()

print(f"📊 Testando com {len(df_bronze_pd_teste)} registros")

for idx, row in df_bronze_pd_teste.iterrows():
    print(f"\n🔍 Teste {idx + 1}/3:")
    print(f"   Trecho: {row['TRECHO_PNEUMOT'][:100]}...")
    
    try:
        resposta, tempo = validar_pneumot_llm(row['TRECHO_PNEUMOT'])
        print(f"   ✅ Resposta LLM: {resposta} (tempo: {tempo:.2f}s)")
    except Exception as e:
        print(f"   ❌ ERRO: {e}")
        print(f"      Corrija a função LLM antes de processar tudo!")
        raise

print("\n" + "=" * 60)
print("✅ TESTE LLM CONCLUÍDO COM SUCESSO!")
print("   LLM está respondendo corretamente.")
print("   Pode prosseguir com validação completa.")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4. Processar Bronze Completa e Validar com LLM (Produção)

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
# Usar len() ao invés de sum() para evitar conflito com PySpark
total_sim = len([x for x in resultados_llm if x == 'SIM'])
total_nao = len([x for x in resultados_llm if x == 'NAO'])
total_erro = len([x for x in resultados_llm if x == 'ERRO'])

# Calcular tempo médio usando reduce ao invés de sum() do PySpark
from functools import reduce
tempo_total = reduce(lambda a, b: a + b, tempos_llm, 0)
tempo_medio = tempo_total / len(tempos_llm) if len(tempos_llm) > 0 else 0

print(f"   SIM: {total_sim}")
print(f"   NAO: {total_nao}")
print(f"   ERRO: {total_erro}")
print(f"   Tempo médio: {tempo_medio:.2f}s")

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

# Filtrar apenas SIM e preparar timestamps
df_positivos = df_silver_pd[df_silver_pd['INF_LLM'] == 'SIM'].copy()

# Preparar informações necessárias: CD_ATENDIMENTO, DT/HR do procedimento original, FONTE
# Vamos buscar essas informações da Bronze (que tem DT_PROCEDIMENTO_REALIZADO)
df_bronze_info = spark.table(TABELA_BRONZE).select(
    'CD_ATENDIMENTO', 'CD_OCORRENCIA', 'CD_ORDEM', 
    'DT_PROCEDIMENTO_REALIZADO', 'FONTE'
).toPandas()

# Merge para trazer data do procedimento positivo
df_positivos = df_positivos.merge(
    df_bronze_info,
    on=['CD_ATENDIMENTO', 'CD_OCORRENCIA', 'CD_ORDEM'],
    how='left',
    suffixes=('', '_bronze')
)

# Verificar se FONTE veio do merge
if 'FONTE' not in df_positivos.columns and 'FONTE_bronze' in df_positivos.columns:
    df_positivos['FONTE'] = df_positivos['FONTE_bronze']

print(f"📊 Casos positivos (SIM): {len(df_positivos)}")
print(f"📊 Atendimentos únicos: {df_positivos['CD_ATENDIMENTO'].nunique()}")
print(f"📋 Colunas disponíveis: {list(df_positivos.columns)}")

# Verificar se FONTE está presente
if 'FONTE' not in df_positivos.columns:
    print("⚠️ ATENÇÃO: Coluna FONTE não encontrada!")
    print("   Verificando fonte dos procedimentos...")
    # Pegar FONTE diretamente da Bronze para cada registro
    for idx, row in df_positivos.iterrows():
        if pd.isna(df_positivos.loc[idx, 'FONTE'] if 'FONTE' in df_positivos.columns else None):
            # Buscar na bronze_info
            match = df_bronze_info[
                (df_bronze_info['CD_ATENDIMENTO'] == row['CD_ATENDIMENTO']) &
                (df_bronze_info['CD_OCORRENCIA'] == row['CD_OCORRENCIA']) &
                (df_bronze_info['CD_ORDEM'] == row['CD_ORDEM'])
            ]
            if len(match) > 0:
                df_positivos.loc[idx, 'FONTE'] = match.iloc[0]['FONTE']

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2. TESTE - Buscar RX de 3 Atendimentos (Amostra)

# COMMAND ----------

# TESTAR BUSCA DE RX COM 3 PROCEDIMENTOS POSITIVOS
print("🧪 MODO TESTE: Buscando RX 12h antes de 3 procedimentos positivos")
print("=" * 60)

df_teste = df_positivos.head(3)

for idx, row in df_teste.iterrows():
    fonte = row['FONTE']
    cd_atend = int(row['CD_ATENDIMENTO'])
    dt_proc = row['DT_PROCEDIMENTO_REALIZADO']
    
    print(f"\n🔍 Teste {idx + 1}/3: Atendimento {cd_atend} ({fonte})")
    print(f"   Data procedimento positivo: {dt_proc}")
    
    try:
        # Buscar RX realizados 12h ANTES do procedimento positivo
        query_rx_teste = f"""
        WITH PROC_POSITIVO AS (
            SELECT 
                CD_ATENDIMENTO,
                CD_OCORRENCIA,
                CD_ORDEM,
                CAST(
                    TO_DATE(DT_PROCEDIMENTO_REALIZADO, 'DD/MM/RR')
                    + NUMTODSINTERVAL(HR_PROCEDIMENTO_REALIZADO, 'SECOND')
                    AS TIMESTAMP
                ) AS TS_PROCEDIMENTO
            FROM RAWZN.RAW_{fonte}_TB_PROCEDIMENTO_REALIZADO
            WHERE CD_ATENDIMENTO = {cd_atend}
              AND CD_OCORRENCIA = {int(row['CD_OCORRENCIA'])}
              AND CD_ORDEM = {int(row['CD_ORDEM'])}
        )
        SELECT 
            RX.CD_ATENDIMENTO,
            RX.CD_OCORRENCIA,
            RX.CD_ORDEM,
            RX.CD_PROCEDIMENTO,
            CAST(
                TO_DATE(RX.DT_PROCEDIMENTO_REALIZADO, 'DD/MM/RR')
                + NUMTODSINTERVAL(RX.HR_PROCEDIMENTO_REALIZADO, 'SECOND')
                AS TIMESTAMP
            ) AS TS_RX
        FROM RAWZN.RAW_{fonte}_TB_PROCEDIMENTO_REALIZADO RX
        CROSS JOIN PROC_POSITIVO PP
        WHERE RX.CD_ATENDIMENTO = PP.CD_ATENDIMENTO
          AND RX.CD_PROCEDIMENTO IN ({','.join(map(str, codigos_rx_torax))})
          AND CAST(
                TO_DATE(RX.DT_PROCEDIMENTO_REALIZADO, 'DD/MM/RR')
                + NUMTODSINTERVAL(RX.HR_PROCEDIMENTO_REALIZADO, 'SECOND')
                AS TIMESTAMP
              ) BETWEEN PP.TS_PROCEDIMENTO - INTERVAL '12' HOUR 
                    AND PP.TS_PROCEDIMENTO
        """
        
        df_rx_teste = run_sql(query_rx_teste)
        print(f"   ✅ {len(df_rx_teste)} RX encontrados 12h antes")
        
        if len(df_rx_teste) > 0:
            df_rx_teste_pd = pd.DataFrame(df_rx_teste)
            print(f"      Timestamps RX: {df_rx_teste_pd['TS_RX'].tolist()}")
        
    except Exception as e:
        print(f"   ❌ ERRO: {e}")
        print(f"      Corrija antes de processar todos!")
        raise

print("\n" + "=" * 60)
print("✅ TESTE BUSCA RX CONCLUÍDO!")
print("   Query de RX com janela 12h funcionando.")
print("   Pode prosseguir com busca completa.")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3. Buscar RX de Tórax de Todos os Casos Positivos (Produção)

# COMMAND ----------

# Lista para acumular RX encontrados
rx_encontrados = []

# Processar por FONTE e em batch
for fonte in ['HSP', 'PSC']:
    print(f"\n🔍 Processando {fonte}...")
    
    # Filtrar positivos desta fonte
    df_positivos_fonte = df_positivos[df_positivos['FONTE'] == fonte].copy()
    
    if len(df_positivos_fonte) == 0:
        print(f"   ⏭️  Nenhum positivo em {fonte}")
        continue
    
    print(f"   📊 {len(df_positivos_fonte)} procedimentos positivos")
    
    # Processar em batch de 100 (menor para não pesar query com timestamps)
    for inicio in range(0, len(df_positivos_fonte), 100):
        chunk_df = df_positivos_fonte.iloc[inicio:inicio + 100]
        
        # Montar condições para múltiplos procedimentos positivos
        condicoes_proc = []
        for _, row in chunk_df.iterrows():
            condicoes_proc.append(
                f"(PP.CD_ATENDIMENTO = {int(row['CD_ATENDIMENTO'])} "
                f"AND PP.CD_OCORRENCIA = {int(row['CD_OCORRENCIA'])} "
                f"AND PP.CD_ORDEM = {int(row['CD_ORDEM'])})"
            )
        
        condicoes_str = ' OR '.join(condicoes_proc)
        
        try:
            # Buscar RX realizados 12h ANTES de cada procedimento positivo
            query_rx = f"""
            WITH PROC_POSITIVO AS (
            SELECT 
                CD_ATENDIMENTO,
                CD_OCORRENCIA,
                CD_ORDEM,
                    CAST(
                        TO_DATE(DT_PROCEDIMENTO_REALIZADO, 'DD/MM/RR')
                        + NUMTODSINTERVAL(HR_PROCEDIMENTO_REALIZADO, 'SECOND')
                        AS TIMESTAMP
                    ) AS TS_PROCEDIMENTO
                FROM RAWZN.RAW_{fonte}_TB_PROCEDIMENTO_REALIZADO PP
                WHERE {condicoes_str}
            )
            SELECT DISTINCT
                RX.CD_ATENDIMENTO,
                RX.CD_OCORRENCIA,
                RX.CD_ORDEM,
                RX.CD_PROCEDIMENTO,
                CAST(
                    TO_DATE(RX.DT_PROCEDIMENTO_REALIZADO, 'DD/MM/RR')
                    + NUMTODSINTERVAL(RX.HR_PROCEDIMENTO_REALIZADO, 'SECOND')
                    AS TIMESTAMP
                ) AS TS_RX
            FROM RAWZN.RAW_{fonte}_TB_PROCEDIMENTO_REALIZADO RX
            INNER JOIN PROC_POSITIVO PP 
                ON RX.CD_ATENDIMENTO = PP.CD_ATENDIMENTO
            WHERE RX.CD_PROCEDIMENTO IN ({','.join(map(str, codigos_rx_torax))})
              AND CAST(
                    TO_DATE(RX.DT_PROCEDIMENTO_REALIZADO, 'DD/MM/RR')
                    + NUMTODSINTERVAL(RX.HR_PROCEDIMENTO_REALIZADO, 'SECOND')
                    AS TIMESTAMP
                  ) BETWEEN PP.TS_PROCEDIMENTO - INTERVAL '12' HOUR 
                        AND PP.TS_PROCEDIMENTO
            """
            
            df_rx = run_sql(query_rx)
            
            if len(df_rx) > 0:
                df_rx = pd.DataFrame(df_rx)
                df_rx['FONTE'] = fonte
                rx_encontrados.append(df_rx)
                print(f"   ✅ Batch {inicio//100 + 1}: {len(df_rx)} RX encontrados")
            else:
                print(f"   ⏭️  Batch {inicio//100 + 1}: nenhum RX")
                
        except Exception as e:
            print(f"   ⚠️ Erro batch {inicio//100 + 1}: {e}")
            continue

if rx_encontrados:
    df_rx_todos = pd.concat(rx_encontrados, ignore_index=True)
    print(f"\n✅ Total de RX de tórax encontrados (12h antes): {len(df_rx_todos)}")
else:
    print("\n⚠️ Nenhum RX de tórax encontrado!")
    df_rx_todos = pd.DataFrame(columns=['CD_ATENDIMENTO', 'CD_OCORRENCIA', 'CD_ORDEM', 'CD_PROCEDIMENTO', 'TS_RX', 'FONTE'])

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
    # Usar len() ao invés de sum() para evitar conflito com PySpark
    com_laudo = len(df_rx_todos[df_rx_todos['DS_LAUDO_MEDICO'] != 'SEM LAUDO'])
    sem_laudo = len(df_rx_todos[df_rx_todos['DS_LAUDO_MEDICO'] == 'SEM LAUDO'])
    print(f"   Com laudo: {com_laudo}")
    print(f"   Sem laudo: {sem_laudo}")

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
    
    # NM_PROCEDIMENTO removido - causa erro no Spark
    # Pode ser buscado depois via JOIN com tabela de procedimentos se necessário
    
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
    # Usar len() ao invés de sum() para evitar conflito com PySpark
    total_urgencia = len(df_rx_todos[df_rx_todos['TIPO_ATENDIMENTO'] == 'URGENCIA'])
    total_eletivo = len(df_rx_todos[df_rx_todos['TIPO_ATENDIMENTO'] == 'ELETIVO'])
    total_sem_info = df_rx_todos['TIPO_ATENDIMENTO'].isna().sum()  # Este sum() é do Pandas, funciona
    print(f"   Urgência: {total_urgencia}")
    print(f"   Eletivo: {total_eletivo}")
    print(f"   Sem info: {total_sem_info}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5. Salvar Tabela Gold no Delta Lake

# COMMAND ----------

if len(df_rx_todos) > 0:
    # Reordenar colunas (NM_PROCEDIMENTO removido - causa erro no Spark)
    colunas_gold = [
        'CD_PACIENTE', 'CD_ATENDIMENTO', 'CD_OCORRENCIA', 'CD_ORDEM',
        'ACC_NUM', 'CD_PROCEDIMENTO', 'DS_LAUDO_MEDICO',
        'CD_MOTIVO_ATENDIMENTO', 'TIPO_ATENDIMENTO', 'FONTE', 'DT_PROCESSAMENTO'
    ]
    
    # Adicionar TS_RX se existir
    if 'TS_RX' in df_rx_todos.columns:
        colunas_gold.insert(colunas_gold.index('DS_LAUDO_MEDICO'), 'TS_RX')
    
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
# MAGIC ## 5.6. Validar Laudos de RX com LLM (Pneumotórax)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.6.1. TESTE - Validar 3 Laudos de RX com LLM

# COMMAND ----------

# Ler Gold
df_gold_spark = spark.table(TABELA_GOLD)
df_gold_pd = df_gold_spark.toPandas()

# Filtrar apenas RX com laudo
df_com_laudo = df_gold_pd[df_gold_pd['DS_LAUDO_MEDICO'] != 'SEM LAUDO'].copy()

print(f"📊 RX na Gold: {len(df_gold_pd)}")
print(f"📊 RX com laudo: {len(df_com_laudo)}")
print(f"📊 RX sem laudo: {len(df_gold_pd) - len(df_com_laudo)}")

# TESTAR COM 3 LAUDOS
print("\n🧪 MODO TESTE: Validando 3 laudos de RX com LLM")
print("=" * 60)

df_teste_rx = df_com_laudo.head(3)

for idx, row in df_teste_rx.iterrows():
    print(f"\n🔍 Teste {idx + 1}/3:")
    print(f"   ACC_NUM: {row['ACC_NUM']}")
    print(f"   Laudo: {str(row['DS_LAUDO_MEDICO'])[:150]}...")
    
    try:
        resposta, tempo = validar_pneumot_llm(str(row['DS_LAUDO_MEDICO']))
        print(f"   ✅ Resposta LLM: {resposta} (tempo: {tempo:.2f}s)")
    except Exception as e:
        print(f"   ❌ ERRO: {e}")
        raise

print("\n" + "=" * 60)
print("✅ TESTE LLM RX CONCLUÍDO!")
print("   Pode prosseguir com validação completa.")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.6.2. Validar Todos os Laudos de RX com LLM (Produção)

# COMMAND ----------

# Validar apenas RX com laudo
resultados_llm_rx = []
tempos_llm_rx = []

print(f"🤖 Validando {len(df_com_laudo)} laudos de RX com LLM...")

for _, row in tqdm(df_com_laudo.iterrows(), total=len(df_com_laudo), desc="Validando laudos RX"):
    resposta, tempo = validar_pneumot_llm(str(row['DS_LAUDO_MEDICO']))
    resultados_llm_rx.append(resposta)
    tempos_llm_rx.append(tempo)

# Adicionar resultados ao DataFrame com laudo
df_com_laudo.loc[:, 'INF_LLM_PNEUMOT'] = resultados_llm_rx
df_com_laudo.loc[:, 'TEMPO_LLM_RX_S'] = tempos_llm_rx

# Estatísticas
from functools import reduce
total_sim_rx = len([x for x in resultados_llm_rx if x == 'SIM'])
total_nao_rx = len([x for x in resultados_llm_rx if x == 'NAO'])
total_erro_rx = len([x for x in resultados_llm_rx if x == 'ERRO'])
tempo_total_rx = reduce(lambda a, b: a + b, tempos_llm_rx, 0)
tempo_medio_rx = tempo_total_rx / len(tempos_llm_rx) if len(tempos_llm_rx) > 0 else 0

print(f"\n📊 Resultados LLM para Laudos de RX:")
print(f"   SIM (pneumotórax confirmado): {total_sim_rx}")
print(f"   NAO (sem pneumotórax): {total_nao_rx}")
print(f"   ERRO: {total_erro_rx}")
print(f"   Tempo médio: {tempo_medio_rx:.2f}s")

# Verificar se colunas foram criadas
print(f"\n🔍 Debug - Colunas em df_com_laudo: {list(df_com_laudo.columns)}")
print(f"   INF_LLM_PNEUMOT presente: {'INF_LLM_PNEUMOT' in df_com_laudo.columns}")
print(f"   TEMPO_LLM_RX_S presente: {'TEMPO_LLM_RX_S' in df_com_laudo.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.6.3. Atualizar Tabela Gold com Validação LLM

# COMMAND ----------

# Merge validação LLM de volta ao DataFrame Gold completo
df_gold_pd = df_gold_pd.merge(
    df_com_laudo[['ACC_NUM', 'INF_LLM_PNEUMOT', 'TEMPO_LLM_RX_S']],
    on='ACC_NUM',
    how='left'
)

# RX sem laudo ficam com NULL (não sabemos se tem pneumotórax)
print(f"✅ Validação LLM anexada ao Gold")
print(f"   Com validação LLM: {df_gold_pd['INF_LLM_PNEUMOT'].notna().sum()}")
print(f"   Sem validação (SEM LAUDO): {df_gold_pd['INF_LLM_PNEUMOT'].isna().sum()}")

# Converter para Spark e salvar
df_gold_spark_final = spark.createDataFrame(df_gold_pd)

df_gold_spark_final.write \
    .format("delta") \
    .option("mergeSchema", "true") \
    .mode("overwrite") \
    .saveAsTable(TABELA_GOLD)

print(f"\n✅ Tabela Gold atualizada com INF_LLM_PNEUMOT: {TABELA_GOLD}")

# Mostrar amostra
display(spark.table(TABELA_GOLD).limit(10))

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



