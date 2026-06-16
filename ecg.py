import wfdb
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from scipy.signal import savgol_filter, butter, filtfilt
import scipy.signal as scipy_signal
import csv
import sys

# ============================
# Função auxiliar: validação de configuração
# ============================
def obter_config_obrigatoria(config, chave, tipo_esperado=None):
    """Obtém uma chave obrigatória do YAML e valida o tipo."""
    if chave not in config:
        raise ValueError(f"❌ Erro: chave obrigatória '{chave}' ausente no config.yaml.")
    valor = config[chave]
    if tipo_esperado is not None and not isinstance(valor, tipo_esperado):
        raise TypeError(
            f"❌ Erro: valor inválido para '{chave}'. "
            f"Esperado {tipo_esperado.__name__}, mas obtido {type(valor).__name__}."
        )
    return valor


# ============================
# Carregar configuração
# ============================
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    sys.exit("❌ Erro: arquivo config.yaml não encontrado.")
except yaml.YAMLError as e:
    sys.exit(f"❌ Erro ao ler config.yaml:\n{e}")

# ============================
# Validação das variáveis obrigatórias
# ============================
gerar_graficos_ecg_completo = obter_config_obrigatoria(config, "gerar_graficos_ecg_completo", str)
if gerar_graficos_ecg_completo not in ["plot", "save", "both", "none"]:
    sys.exit("❌ Erro: 'gerar_graficos_ecg_completo' deve ser 'plot', 'save', 'none' ou 'both'.")

sinal = obter_config_obrigatoria(config, "sinal", str)
usar_todos_registros = obter_config_obrigatoria(config, "usar_todos_registros", bool)
registros_especificos = obter_config_obrigatoria(config, "registros_especificos", list)
pasta_resultados_ecg_completo = obter_config_obrigatoria(config, "pasta_resultados_ecg_completo", str)
pasta_csv = obter_config_obrigatoria(config, "pasta_csv", str)

# Converte ~
pasta_csv = os.path.expanduser(pasta_csv)

# Normaliza barras
pasta_csv = pasta_csv.replace("\\", "/")

# SE o caminho começa com C:/, D:/ etc → converter para /mnt/c/
import re
match = re.match(r"([A-Za-z]):/(.*)", pasta_csv)
if match:
    drive = match.group(1).lower()
    rest = match.group(2)
    pasta_csv = f"/mnt/{drive}/{rest}"

pasta_csv = os.path.abspath(pasta_csv)

# Criar pasta
os.makedirs(pasta_csv, exist_ok=True)


usar_intervalo = obter_config_obrigatoria(config, "usar_intervalo", bool)
intervalo = obter_config_obrigatoria(config, "intervalo", list)
ajustar_pico = obter_config_obrigatoria(config, "ajustar_pico", bool)

lista_filtros = obter_config_obrigatoria(config, "lista_filtros", list)

janela = obter_config_obrigatoria(config, "janela_savitz", int)
ordem = obter_config_obrigatoria(config, "ordem_savitz", int)

butter_ordem = obter_config_obrigatoria(config, "butter_ordem", int)
butter_lowcut = obter_config_obrigatoria(config, "butter_lowcut", float)
butter_highcut = obter_config_obrigatoria(config, "butter_highcut", float)

shift = obter_config_obrigatoria(config, "shift", int)
gerar_graficos_segmentos = obter_config_obrigatoria(config, "gerar_graficos_segmentos", bool)

usar_separacao_por_batimento = obter_config_obrigatoria(config, "usar_separacao_por_batimento", bool)
num_batimentos = obter_config_obrigatoria(config, "num_batimentos", int)
inicio_batimento_heuristica = obter_config_obrigatoria(config, "inicio_batimento_heuristica", float)
fim_batimento_heuristica = obter_config_obrigatoria(config, "fim_batimento_heuristica", float)
plot_etapas_pan_tompkins = obter_config_obrigatoria(config, "plot_etapas_pan_tompkins", bool)

# ============================
# Validações adicionais de coerência
# ============================
if janela % 2 == 0:
    sys.exit("❌ Erro: 'janela_savitz' deve ser um número ímpar.")
if butter_highcut <= butter_lowcut:
    sys.exit("❌ Erro: 'butter_highcut' deve ser maior que 'butter_lowcut'.")
if num_batimentos < 1:
    sys.exit("❌ Erro: 'num_batimentos' deve ser maior ou igual a 1.")
if shift < 1:
    sys.exit("❌ Erro: 'shift' deve ser maior ou igual a 1.")
if not isinstance(intervalo, list) or len(intervalo) != 2:
    sys.exit("❌ Erro: 'intervalo' deve ser uma lista com dois valores: [inicio, fim].")


# ============================
# Criação de pastas
# ============================
if gerar_graficos_ecg_completo in ["save", "both"]:
    os.makedirs(pasta_resultados_ecg_completo, exist_ok=True)
os.makedirs(pasta_csv, exist_ok=True)

# ============================
# Funções de filtro
# ============================
def butter_bandpass(lowcut, highcut, fs, ordem=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(ordem, [low, high], btype="band")
    return b, a

def aplicar_filtro_butter(sinal, fs, lowcut, highcut, ordem=4):
    b, a = butter_bandpass(lowcut, highcut, fs, ordem=ordem)
    return filtfilt(b, a, sinal)

# ============================
# Detecção de picos R (Pan-Tompkins, usado no MIT-BIH)
# ============================
def detectar_picos_pan_tompkins(sinal, fs):
    """
    Detecta picos R usando uma implementação inspirada em Pan-Tompkins com limiarização
    mais robusta (mediana + MAD) e uso de 'prominence' para estabilidade.
    - sinal: vetor 1D (preferencialmente filtrado)
    - fs: frequência de amostragem (Hz)
    - plot_etapas_pan_tompkins: se True, plota as etapas (mantém os plots abertos e espera enter)
    Retorna: array de índices (amostras) dos picos detectados
    """
    x = np.asarray(sinal, dtype=float)
    if len(x) == 0:
        return np.array([], dtype=int)

    # 1) passa-faixa 5–15 Hz (ordem 2)
    lowcut, highcut = 5, 15
    b, a = butter(2, [lowcut / (fs/2), highcut / (fs/2)], btype='band')
    x_f = filtfilt(b, a, x)

    # 2) derivada (realça inclinações)
    derivada = np.diff(x_f, prepend=x_f[0])
    # normalizar a derivada para evitar divisões por zero
    max_abs_der = np.max(np.abs(derivada)) + 1e-12
    derivada = derivada / max_abs_der

    # 3) quadrado (torna tudo positivo; mas mantemos sinal original para refinamento)
    quadrado = derivada ** 2

    # 4) integração móvel (~150 ms)
    janela_ms = 150
    janela = int(fs * janela_ms / 1000)
    if janela < 1:
        janela = 1
    integracao = np.convolve(quadrado, np.ones(janela)/janela, mode='same')

    # estatísticas robustas da integração
    med = np.median(integracao)
    mad = np.median(np.abs(integracao - med))  # median absolute deviation

    # definir limiar inicial robusto
    initial_thresh = med + 3.0 * mad  
    # definir prominence base (relativa ao desvio padrão)
    base_prominence = 0.4 * (np.std(integracao) + 1e-12)

    distancia = int(0.4 * fs)  # 200 ms

    # Busca iterativa: começa com limiar robusto e relaxa se não encontrar picos
    limiar = initial_thresh
    peaks = np.array([], dtype=int)
    for attempt in range(6):
        peaks, props = scipy_signal.find_peaks(
            integracao,
            height=limiar,
            distance=distancia,
            prominence=base_prominence
        )
        if len(peaks) >= 1:
            break
        # relaxa limiar: mistura mediana e percentil para não descer demais
        pct = np.percentile(integracao, max(50 - 5*attempt, 20))  # 50,45,40,... até 20
        limiar = 0.6 * limiar + 0.4 * pct

    # se ainda nada, usar percentile 65 como fallback
    if len(peaks) == 0:
        peaks, props = scipy_signal.find_peaks(
            integracao,
            height=np.percentile(integracao, 65),
            distance=distancia
        )

    # 6) Refinamento: escolher ponto de maior magnitude absoluta na janela ao redor do pico
    picos_refinados = []
    janela_busca = int(0.1 * fs)  # 100 ms
    for p in peaks:
        ini = max(0, p - janela_busca)
        fim = min(len(x_f), p + janela_busca)
        if fim <= ini:
            continue
        # índice relativo com maior magnitude absoluta (aceita R negativo ou positivo)
        idx_rel = np.argmax(np.abs(x_f[ini:fim]))
        picos_refinados.append(ini + idx_rel)

    picos_refinados = np.array(sorted(set(picos_refinados)), dtype=int)

    # --- plot das etapas (opcional) ---
    if plot_etapas_pan_tompkins:
        etapas = [
            ("Sinal Original", x),
            ("Após Filtro Passa-Faixa (5–15 Hz)", x_f),
            ("Derivada", derivada),
            ("Sinal Quadrado", quadrado),
            ("Integração Móvel (~150 ms)", integracao),
        ]
        plt.ion()
        for titulo, sinal_etapa in etapas:
            plt.figure(figsize=(14, 4))
            plt.plot(sinal_etapa, color='black')
            plt.title(f"{titulo} — Etapa Pan-Tompkins")
            plt.xlabel("Amostras")
            plt.ylabel("Amplitude normalizada")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.pause(0.001)
            input("Pressione [Enter] para continuar...")
        # plot final: picos sobre integracao
        plt.figure(figsize=(14, 4))
        plt.plot(integracao, label="Integração móvel", color='black')
        if len(peaks) > 0:
            plt.plot(peaks, integracao[peaks], "ro", label="Picos (pre-refinamento)")
        plt.axhline(limiar, color='blue', linestyle='--', label=f"Limiar final ({limiar:.4g})")
        plt.title("Detecção de Picos — Integração móvel")
        plt.xlabel("Amostras")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.pause(0.001)
        input("Pressione [Enter] para continuar...")

    return picos_refinados


# ============================
# Função para salvar gráfico
# ============================
def salvar_grafico(y, titulo, nome_arquivo, inicio_plot=0, anotacoes_plot=None, simbolos_plot=None, registro=None):
    """
    Exibe e/ou salva um gráfico dependendo da configuração gerar_graficos_ecg_completo.
    """
    # Prepara título completo com identificação do registro se fornecida
    if registro is not None:
        titulo_completo = f"Registro {registro} — {titulo}"
    else:
        titulo_completo = titulo

    # Cria figura
    plt.figure(figsize=(14, 6))
    eixo_x = np.arange(len(y)) + inicio_plot
    plt.plot(eixo_x, y, label=titulo, alpha=0.8)

    offset = 0.025 * (np.max(y) - np.min(y)) if len(y) > 0 else 0
    if anotacoes_plot is not None:
        for i, amostra in enumerate(anotacoes_plot):
            idx_relativo = amostra - inicio_plot
            if 0 <= idx_relativo < len(y):
                plt.plot(idx_relativo + inicio_plot, y[idx_relativo], "ro", markersize=4)
                if simbolos_plot is not None and i < len(simbolos_plot):
                    if y[idx_relativo] >= 0:
                        y_text = y[idx_relativo] + offset
                        va = "bottom"
                    else:
                        y_text = y[idx_relativo] - offset
                        va = "top"
                    plt.text(idx_relativo + inicio_plot, y_text, simbolos_plot[i],
                             fontsize=8, color="red", ha="center", va=va)

    plt.title(f"{titulo_completo} - Sinal {sinal}")
    plt.xlabel("Amostras")
    plt.ylabel("mV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    caminho_saida = os.path.join(pasta_resultados_ecg_completo, nome_arquivo)

    # Salvar gráfico se gerar_graficos_ecg_completo for 'save' ou 'both'
    if gerar_graficos_ecg_completo in ["save", "both"]:
        plt.savefig(caminho_saida)

    # Mostrar gráfico se gerar_graficos_ecg_completo for 'plot' ou 'both'
    if gerar_graficos_ecg_completo in ["plot", "both"]:
        plt.ion()
        plt.show()
        plt.pause(0.001)
        input("Press [enter] para continuar.")
    else:
        plt.close()

# ============================
# Inicializa listas para CSVs
# ============================
resultados_registro_list = []
resultados_segmentos_list = []

# ============================
# Função de separação heurística
# ============================
def separar_batimentos_por_heuristica(sinal_filtrado, anotacoes, simbolos, registro, config, fs):
    """
    Separa segmentos por número de batimentos usando heurística fisiológica:
    Cada batimento começa ~inicio_batimento_heuristica s antes do pico R
    e termina ~fim_batimento_heuristica s depois do último pico R.
    """

    print(f"\n🔹 Separando batimentos por heurística do registro {registro}...")
    print(f"→ num_batimentos={num_batimentos}, shift={shift}, inicio={inicio_batimento_heuristica}s, fim={fim_batimento_heuristica}s")

    total = len(anotacoes)
    if total < num_batimentos:
        print("⚠️ Poucos picos R detectados — separação ignorada.")
        return

    num_segmentos = 0
    segmentos_normais = 0
    segmentos_anormais = 0

    samples_antes_R = int(inicio_batimento_heuristica * fs)
    samples_depois_R = int(fim_batimento_heuristica * fs)

    for i in range(0, total - num_batimentos + 1, shift):
        # início do primeiro batimento do segmento
        ini = max(anotacoes[i] - samples_antes_R, 0)
        # fim do último batimento do segmento
        fim = min(anotacoes[i + num_batimentos - 1] + samples_depois_R, len(sinal_filtrado) - 1)

        segmento = sinal_filtrado[ini:fim+1]
        anotacoes_segmento = anotacoes[i:i+num_batimentos]
        simbolos_segmento = simbolos[i:i+num_batimentos]

        # Guardar TODAS as anotações para o gráfico
        anotacoes_completas = np.copy(anotacoes[i:i+num_batimentos])
        simbolos_completos = np.copy(simbolos[i:i+num_batimentos])
        
        # --- Aplicar regra de ignorar primeiro/último batimento ---
        anotacoes_validas = np.copy(anotacoes_segmento)
        simbolos_validos = np.copy(simbolos_segmento)

        if inicio_batimento_heuristica == 0 and len(anotacoes_segmento) > 1:
            anotacoes_segmento = anotacoes_segmento[1:]
            simbolos_segmento = simbolos_segmento[1:]

        if fim_batimento_heuristica == 0 and len(anotacoes_segmento) > 1:
            anotacoes_segmento = anotacoes_segmento[:-1]
            simbolos_segmento = simbolos_segmento[:-1]

        # Novo número real de batimentos usados
        num_batimentos_reais = len(anotacoes_segmento)

        # --- Determinar label do segmento ---
        # Considera apenas os batimentos válidos (já filtrados acima)
        if num_batimentos_reais == 0:
            # segmento vazio, ignora
            continue
        
        if all(s == 'N' for s in simbolos_segmento):
            label = 'N'
            segmentos_normais += 1
        else:
            label = 'A'
            segmentos_anormais += 1

        num_segmentos += 1

        # Registrar no CSV de segmentos
        resultados_segmentos_list.append([
            registro,
            i,
            ini,
            fim + 1,
            num_batimentos_reais,  # número ajustado de batimentos válidos
            sum(s != 'N' for s in simbolos_segmento),  # batimentos anormais válidos
            label
        ])

        if gerar_graficos_segmentos:
            salvar_grafico(
                segmento,
                f"{registro}_seg_{i:04d}_heur",
                f"{registro}_seg_{i:04d}_heur.png",
                inicio_plot=ini,
                anotacoes_plot=anotacoes_completas,
                simbolos_plot=simbolos_completos
            )
        
    # Estatísticas do registro
    batimento_min = np.min(sinal_filtrado)
    batimento_max = np.max(sinal_filtrado)
    batimento_medio = np.mean(sinal_filtrado)
    total_batimentos = len(anotacoes)
    total_normais = sum(s == 'N' for s in simbolos)
    total_anormais = sum(s != 'N' for s in simbolos)

    resultados_registro_list.append([
        registro,
        batimento_min,
        batimento_max,
        batimento_medio,
        num_segmentos,
        segmentos_normais,
        segmentos_anormais,
        total_batimentos,
        total_normais,
        total_anormais
    ])


    print("✅ Separação por heurística concluída.")

# ============================
# Caminho dos dados
# ============================
base_path = obter_config_obrigatoria(config, "caminho_mitbih", str)

# Normalizar caminhos
base_path = os.path.expanduser(base_path)
base_path = base_path.replace("\\", "/")

import re
match = re.match(r"([A-Za-z]):/(.*)", base_path)
if match:
    drive = match.group(1).lower()
    rest = match.group(2)
    base_path = f"/mnt/{drive}/{rest}"

base_path = os.path.abspath(base_path)

if not os.path.exists(base_path):
    sys.exit(f"❌ Caminho do MIT-BIH não encontrado: {base_path}")


# ============================
# Listar registros
# ============================
arquivos = [f.split(".")[0] for f in os.listdir(base_path) if f.endswith(".dat")]
arquivos = sorted(list(set(arquivos)))
registros = arquivos if usar_todos_registros else [r for r in registros_especificos if r in arquivos]
print(f"Registros selecionados: {registros}")

# ============================
# Processar registros
# ============================
for registro in registros:
    caminho = os.path.join(base_path, registro)
    print(f"\nLendo {registro}...")

    inicio, fim = intervalo

    # 🔧 Ler o registro completo ou apenas o intervalo
    if usar_intervalo:
        record = wfdb.rdrecord(caminho, sampto=fim)
        dados = record.p_signal[inicio:fim, :]
        inicio_plot = inicio
    else:
        record = wfdb.rdrecord(caminho)  # lê tudo
        dados = record.p_signal
        inicio_plot = 0

    fs = record.fs
    canais = record.sig_name

    if sinal not in canais:
        print(f"Sinal {sinal} não encontrado em {registro}. Canais disponíveis: {canais}")
        continue

    idx = canais.index(sinal)
    sinal_original = dados[:, idx]


    sinal_filtrado = np.copy(sinal_original)

    for filtro in lista_filtros:

        # ----------------------------
        # FILTRO SAVITZKY-GOLAY
        # ----------------------------
        if filtro == "savgol":
            sinal_filtrado = savgol_filter(sinal_filtrado, janela, ordem)

        # ----------------------------
        # FILTRO BUTTERWORTH
        # ----------------------------
        elif filtro == "butter":
            b, a = butter(butter_ordem, [butter_lowcut, butter_highcut], btype="bandpass", fs=fs)
            sinal_filtrado = filtfilt(b, a, sinal_filtrado)


    # Ler anotações (picos R e símbolos)
    try:
        ann = wfdb.rdann(caminho, "atr")
        anotacoes = np.array(ann.sample)
        simbolos = np.array(ann.symbol)
    except Exception as e:
        print("⚠️ Não foi possível ler anotações com rdann:", e)
        anotacoes = np.array([], dtype=int)
        simbolos = np.array([], dtype=str)

    # 🔧 Aplicar máscara SOMENTE se usar_intervalo=True
    if usar_intervalo:
        mask = (anotacoes >= inicio) & (anotacoes < fim)
        anotacoes = anotacoes[mask]
        simbolos = simbolos[mask]
    else:
        # caso contrário, usa todas as anotações
        pass

    # ============================
    # Detecção automática e comparação visual
    # ============================
    print("ℹ️ Detectando picos R automaticamente (Pan–Tompkins) para comparação...")
    picos_detectados = detectar_picos_pan_tompkins(sinal_filtrado, fs)

    # Sempre plota ambos:
    # - ❌ Azul: anotações reais (se existirem)
    # - 🔴 Vermelho: picos automáticos

    # Se houver anotações, manter as originais
    if len(anotacoes) > 0:
        print(f"✅ {len(anotacoes)} anotações encontradas — comparando com {len(picos_detectados)} picos automáticos.")
    else:
        print(f"⚠️ Nenhuma anotação encontrada — exibindo apenas picos detectados automaticamente.")
        anotacoes = np.array([], dtype=int)
        simbolos = np.array([], dtype=str)

    # Plot de comparação
    plt.figure(figsize=(14, 6))
    plt.plot(sinal_filtrado, label=f"Sinal Filtrado ({sinal})", color='black', alpha=0.8)

    # Anotações originais (❌ azul)
    if len(anotacoes) > 0:
        plt.plot(anotacoes, sinal_filtrado[anotacoes], 'bx', label="Picos R (anotações MIT-BIH)", markersize=8, markeredgewidth=2)

    # Picos detectados (🔴 círculo vermelho)
    if len(picos_detectados) > 0:
        plt.plot(picos_detectados, sinal_filtrado[picos_detectados], 'ro', label="Picos R (detecção automática)", markersize=5)

    if gerar_graficos_ecg_completo in ["plot", "both"]:
        plt.title(f"Comparação: Anotações vs Detecção Automática — Registro {registro}")
        plt.xlabel("Amostras")
        plt.ylabel("mV")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        input("Pressione [Enter] para continuar...")

    # Ajuste de pico R
    if ajustar_pico:
        print("🔧 Ajustando picos R...")
        sinal_para_busca = sinal_filtrado
        anotacoes_ajustadas = np.copy(anotacoes)
        janela_pico = 50
        for j, amostra in enumerate(anotacoes):
            idx_relativo = amostra - inicio_plot
            ini_j = max(0, idx_relativo - janela_pico)
            fim_j = min(len(sinal_para_busca), idx_relativo + janela_pico)
            if fim_j <= ini_j:
                continue
            idx_max_rel = np.argmax(sinal_para_busca[ini_j:fim_j] ** 2)
            anotacoes_ajustadas[j] = inicio_plot + ini_j + idx_max_rel
    else:
        anotacoes_ajustadas = np.copy(anotacoes)

    # ============================
    # Gráficos com nomes descritivos
    # ============================

    nomes_filtros = []
    sufixos_filtros = []

    if "savgol" in lista_filtros:
        nomes_filtros.append("Savitz-Golay")
        sufixos_filtros.append("savgol")

    if "butter" in lista_filtros:
        nomes_filtros.append("Butterworth")
        sufixos_filtros.append("butter")

    if len(nomes_filtros) == 0:
        tipo_filtro = "SemFiltro"
        sufixo_filtro = ""
    else:
        tipo_filtro = "_".join(nomes_filtros)
        sufixo_filtro = "_".join(sufixos_filtros)

    # 1️⃣ Sinal original
    salvar_grafico(
        sinal_original,
        f"Sinal Original ({sinal})",
        f"{registro}_{sinal}_original.png",
        inicio_plot=inicio_plot,
        anotacoes_plot=anotacoes,
        simbolos_plot=simbolos,
        registro=registro
    )

    # 2️⃣ Sinal filtrado (identifica o filtro)
    salvar_grafico(
        sinal_filtrado,
        f"Sinal Filtrado - {tipo_filtro} ({sinal})",
        f"{registro}_{sinal}_filtrado_{sufixo_filtro}.png",
        inicio_plot=inicio_plot,
        anotacoes_plot=anotacoes,
        simbolos_plot=simbolos,
        registro=registro
    )

    # 3️⃣ Sinal ajustado (caso ajustar_pico=True)
    salvar_grafico(
        sinal_filtrado,
        f"Sinal Ajustado (Picos R ajustados) - {tipo_filtro} ({sinal})",
        f"{registro}_{sinal}_ajustado_{sufixo_filtro}.png",
        inicio_plot=inicio_plot,
        anotacoes_plot=anotacoes_ajustadas,
        simbolos_plot=simbolos,
        registro=registro
    )

    if usar_separacao_por_batimento:
        separar_batimentos_por_heuristica(sinal_filtrado, anotacoes_ajustadas, simbolos, registro, config, fs)

    # ============================
    # Salvar CSV de segmentos do registro atual
    # ============================
    nome_csv_segmentos = f"segmentos_{registro}.csv"
    caminho_csv_segmentos = os.path.join(pasta_csv, nome_csv_segmentos)
    with open(caminho_csv_segmentos, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Registro', 'Segmento_idx', 'Start_sample', 'End_sample_exclusive',
                         'Num_beats_in_segment', 'Num_abnormal_beats', 'Label'])
        # escreve apenas os segmentos deste registro
        for linha in resultados_segmentos_list:
            if linha[0] == registro:
                writer.writerow(linha)

    print(f"✅ CSV de segmentos salvo em: {caminho_csv_segmentos}")

    # Limpa a lista de segmentos antes do próximo registro
    resultados_segmentos_list.clear()

# ============================
# Salvar CSV geral de registros (resumo)
# ============================
caminho_csv_resumo = os.path.join(pasta_csv, 'resultados_registro.csv')
with open(caminho_csv_resumo, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Registro', 'Batimento_min', 'Batimento_max', 'Batimento_medio',
                     'Num_segmentos', 'Segmentos_normais', 'Segmentos_anormais', 'Total_batimentos', 'Total_normais', 'Total_anormais'])

    writer.writerows(resultados_registro_list)

# ============================
# Gerar arquivo de documentação da segmentação
# ============================
documentacao_path = os.path.join(pasta_csv, "documentacao_segmentacao.txt")

with open(documentacao_path, 'w', encoding='utf-8') as doc:
    doc.write("===========================================\n")
    doc.write("📘 DOCUMENTAÇÃO DE SEGMENTAÇÃO\n")
    doc.write("===========================================\n\n")

    doc.write("Filtros aplicados:\n")
    if "savgol" in lista_filtros:
        doc.write(f"- Savitz-Golay (janela={janela}, ordem={ordem})\n")
    if "butter" in lista_filtros:
        doc.write(f"- Butterworth (ordem={butter_ordem}, lowcut={butter_lowcut} Hz, highcut={butter_highcut} Hz)\n")
    if len(lista_filtros) == 0:
        doc.write("- Nenhum filtro aplicado\n")


    doc.write(f"\nAjuste de picos R: {'Sim' if ajustar_pico else 'Não'}\n")
    doc.write(f"Modo de geração do ECG completo: {gerar_graficos_ecg_completo}\n")

    doc.write("\n-------------------------------------------\n")
    if usar_separacao_por_batimento:
        doc.write("🔹 Modo de separação: Por batimentos (heurística fisiológica)\n")
        doc.write(f"   - Número de batimentos por segmento: {num_batimentos}\n")
        doc.write(f"   - Deslocamento (shift): {shift} batimentos\n")
        doc.write(f"   - Início do batimento (antes do R): {inicio_batimento_heuristica} s ({int(inicio_batimento_heuristica*fs)} amostras)\n")
        doc.write(f"   - Fim do batimento (após o R): {fim_batimento_heuristica} s ({int(fim_batimento_heuristica*fs)} amostras)\n")
        doc.write(f"   - Geração de gráficos de segmentos: {'Sim' if gerar_graficos_segmentos else 'Não'}\n")
    else:
        doc.write("⚠️ Nenhum modo de separação foi ativado.\n")

    doc.write("\n-------------------------------------------\n")
    doc.write("📁 Pastas utilizadas:\n")
    doc.write(f"   - Resultados (gráficos): {pasta_resultados_ecg_completo}\n")
    doc.write(f"   - CSVs: {pasta_csv}\n")

    doc.write("\n-------------------------------------------\n")
    doc.write("🕒 Registros processados:\n")
    for r in registros:
        doc.write(f"   - {r}\n")

    doc.write("\n✅ Arquivo gerado automaticamente pelo script de segmentação.\n")

print(f"📝 Arquivo de documentação salvo em: {documentacao_path}")

print(f"✅ CSV geral salvo em: {caminho_csv_resumo}")
