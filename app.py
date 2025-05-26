
from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

########################
# Configurações gerais #
########################
BASE_PATH = Path(__file__).parent
CSV_PATH = BASE_PATH / "dados.csv"
CERT_PATH = BASE_PATH / "certidoes"
LOGO_PATH = BASE_PATH / "logoo.png"
CLIENT_NAME = "Elielcio"
MODEL_NAME = "deepseek-r1-distill-llama-70b"

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("Variável GROQ_API_KEY não encontrada.")
    st.stop()

#########################
# Configuração do logger #
#########################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.FileHandler(BASE_PATH / "app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

######################################
# Index simples das certidões locais #
######################################

def indexar_certidoes(pasta: Path) -> dict[str, Path]:
    """Cria um dicionário {slug_ascii: filepath} para pesquisa rápida."""
    index: dict[str, Path] = {}
    if pasta.exists():
        for arquivo in pasta.glob("*.pdf"):
            slug = (
                arquivo.stem.lower()
                .replace("_", " ")
                .replace("-", " ")
                .replace("  ", " ")
                .strip()
            )
            slug_ascii = unicodedata.normalize("NFKD", slug).encode("ascii", "ignore").decode("ascii")
            index[slug_ascii] = arquivo
    return index

CERTIDOES = indexar_certidoes(CERT_PATH)
logger.info("Certidões indexadas: %s", CERTIDOES)

###############################
# Utilitários de normalização #
###############################

def normalizar_txt(txt: str) -> str:  # noqa: D401
    """Remove acentos, caixa alta e símbolos não alfanuméricos."""
    ascii_txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    ascii_txt = re.sub(r"[^a-z0-9 ]", " ", ascii_txt.lower())
    ascii_txt = re.sub(r"\s+", " ", ascii_txt).strip()
    return ascii_txt

###############################
# Funções utilitárias do CSV  #
###############################
@st.cache_data(show_spinner=False)

def carregar_df(caminho: Path) -> pd.DataFrame:
    """Lê o CSV do cliente com colunas: Data, faturamento, inss_retido."""
    try:
        df = pd.read_csv(caminho, sep=";", dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        def _parse_money(txt: str):
            if pd.isna(txt) or txt == "":
                return 0.0
            txt = txt.strip().replace("\u00A0", "")
            txt = txt.replace(".", "").replace(",", ".")
            return float(txt)

        for col in ("faturamento", "inss_retido"):
            df[col] = df[col].apply(_parse_money)

        df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
        return df
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Erro ao carregar CSV: %s", exc)
        st.error(f"Erro ao carregar {caminho.name}: {exc}")
        return pd.DataFrame()


def df_para_prompt(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, sep=";")


dados_df = carregar_df(CSV_PATH)

# Resumos financeiros
faturamento_total = dados_df["faturamento"].sum() if not dados_df.empty else 0.0
inss_retido_total = dados_df["inss_retido"].sum() if not dados_df.empty else 0.0

###################################
# Configuração do modelo (Groq LLM) #
###################################
client = ChatGroq(api_key=API_KEY, model=MODEL_NAME)
MEMORIA_PADRAO = ConversationBufferWindowMemory(k=5, return_messages=True)

system_message = f"""
Você é Victor, assistente virtual do cliente \"{CLIENT_NAME}\".
Fale **sempre** em português brasileiro, de forma clara e objetiva.
Ignore qualquer texto entre as tags <think> e </think>; trate‑o como nota interna que NÃO deve ser respondida nem exibida.
Você interage exclusivamente com {CLIENT_NAME}.

Resumo financeiro até {datetime.now().strftime('%d/%m/%Y')}:
- **Faturamento acumulado:** R$ {faturamento_total:,.2f}
- **INSS retido acumulado:** R$ {inss_retido_total:,.2f}

Eu também tenho acesso às seguintes certidões (PDF):
{', '.join(CERTIDOES.keys()) or 'nenhuma'}

Base de dados detalhada:
###
{df_para_prompt(dados_df)}
###

- As colunas correspondem a `Data`, `Faturamento` e `INSS Retido`.
- Todos os valores estão em Reais (BRL).

Se precisar de uma certidão, basta pedir — exemplos: “quero a CND estadual”, “quero a CND FGTS”, "quero a CND federal", "quero a CND fiscal etc.

Termine perguntando se precisa de algo mais.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("placeholder", "{chat_history}"),
    ("user", "{input}"),
])

chain = prompt_template | client

#######################################
# Funções para envio de certidões PDF #
#######################################

CATEGORIAS_CERT = {
    "estadual": "Estadual",
    "federal": "Federal",
    "municipal": "Municipal",
    "fgts": "FGTS",
    "fiscal": "Fiscal",          # ← novo
}

def tentar_enviar_certidao(mensagem: str) -> Tuple[Optional[Path], Optional[str]]:
    """Retorna (Path, categoria) da certidão solicitada ou (None, None)."""
    txt = normalizar_txt(mensagem)

    # Verifica se mensagem contém 'cnd' ou 'certidao'
    if "cnd" not in txt and "certidao" not in txt:
        return None, None

    for cat_slug, cat_nome in CATEGORIAS_CERT.items():
        if cat_slug in txt:
            # Procura arquivo cujo slug contenha a categoria
            for slug, path in CERTIDOES.items():
                if cat_slug in slug and ("cnd" in slug or "certidao" in slug):
                    return path, cat_nome
    return None, None

###########################
# Interface Streamlit      #
###########################

def desenhar_sidebar() -> None:
    with st.sidebar:
        st.image(LOGO_PATH, use_container_width=True)
        abas = st.tabs(["Conversas", "Configurações"])

        with abas[0]:
            if st.button("🗑️ Apagar Histórico", use_container_width=True):
                st.session_state["memoria"] = ConversationBufferWindowMemory(k=5, return_messages=True)
                st.success("Histórico apagado!")

        with abas[1]:
            st.header("⚙️ Configurações")
            st.markdown(
                f"""
                - **Modelo:** {MODEL_NAME}
                - **Usuário autorizado:** {CLIENT_NAME}
                - **Linhas CSV:** {len(dados_df)}
                - **Certidões disponíveis:** {', '.join(CERTIDOES) or 'nenhuma'}
                """
            )


def pagina_chat() -> None:
    st.header("🤖 Analista Fiscal & Contábil")

    memoria: ConversationBufferWindowMemory = st.session_state.get("memoria", MEMORIA_PADRAO)

    # Exibir histórico
    for msg in memoria.buffer_as_messages:
        st.chat_message(msg.type).markdown(msg.content)

    entrada = st.chat_input("Fale com o Analista")
    if not entrada:
        return

    # Remover blocos <think>…</think>
    entrada_limpa = re.sub(r"<think>.*?</think>", "", entrada, flags=re.DOTALL | re.IGNORECASE).strip()
    if not entrada_limpa:
        st.info("(Comentário interno ignorado.)")
        return

    st.chat_message("human").markdown(entrada_limpa)

    # Verificar pedido de certidão
    cert_path, categoria = tentar_enviar_certidao(entrada_limpa)
    if cert_path and categoria:
        resposta = f"Aqui está a CND {categoria} conforme solicitado."
        bot_msg = st.chat_message("ai")
        bot_msg.markdown(resposta)
        with open(cert_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            f"📄 Baixar CND {categoria}",
            data=pdf_bytes,
            file_name=cert_path.name,
            mime="application/pdf",
        )
        memoria.chat_memory.add_user_message(entrada_limpa)
        memoria.chat_memory.add_ai_message(resposta + f" [CND {categoria} anexada]")
        st.session_state["memoria"] = memoria
        return

    # Caso contrário, consultar o LLM
    bot_container = st.chat_message("ai")
    resposta_llm = bot_container.write_stream(
        chain.stream({"input": entrada_limpa, "chat_history": memoria.buffer_as_messages})
    )
    memoria.chat_memory.add_user_message(entrada_limpa)
    memoria.chat_memory.add_ai_message(resposta_llm)
    st.session_state["memoria"] = memoria


def main() -> None:
    desenhar_sidebar()
    pagina_chat()


if __name__ == "__main__":
    main()
