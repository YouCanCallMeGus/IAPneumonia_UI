import json
import os
import sys
import time
from html import escape

import streamlit as st

PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PASTA))

from physionet_modelV1 import carpediem, prever

CSS = """
<style>
.stApp { background: #f5f6fa; }
header[data-testid="stHeader"] { background: transparent; }
.stButton button { background: #ffffff; color: #1f2330; border: 1px solid #d9dce6; }
.stButton button:hover { color: #a8690a; border-color: #a8690a; }
.stButton button:disabled { color: #6b7188; background: #ffffff; border-color: #d9dce6; }
.stCheckbox p { color: #5f667c; }

.topo h1 { color: #1f2330; font-size: 2rem; margin: 0; padding: 0; }
.topo p { color: #5f667c; margin: 4px 0 8px; }
.progresso { height: 6px; border-radius: 3px; background: #d9dce6; margin-bottom: 24px; }
.progresso div { height: 100%; border-radius: 3px; background: #e9ac2e; transition: width .4s; }

.janela {
  display: flex;
  flex-direction: column-reverse;
  height: max(300px, calc(100vh - 420px));
  margin-bottom: 20px;
  padding: 10px 0;
  overflow-y: auto;
  container-type: inline-size;
}
.arvore { width: 320px; margin: 0 auto auto max(0px, calc(50% - 360px)); }
.no {
  position: relative;
  box-sizing: border-box;
  width: 200px;
  margin-left: calc(60px + var(--x));
  padding: 12px 16px;
  border: 2px solid #e9ac2e;
  border-radius: 12px;
  background: #ffffff;
}
.no.foco { border-color: #a8690a; box-shadow: 0 0 0 7px rgba(233, 172, 46, .3); }
.no b { display: block; color: #1f2330; font-size: .9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.no strong { display: block; color: #a8690a; font-size: 1.6rem; line-height: 1.3; }
.barra { height: 5px; border-radius: 3px; background: #f3e6c8; }
.barra div { height: 100%; border-radius: 3px; background: #e9ac2e; }
.arvore svg { display: block; }
.arvore path { fill: none; stroke: #e9ac2e; stroke-width: 5; stroke-linecap: round; }

.motivos {
  position: absolute;
  bottom: 0;
  left: calc(100% + 16px);
  box-sizing: border-box;
  width: max-content;
  max-width: min(360px, calc(100cqw - max(0px, 50cqw - 360px) - 280px - var(--x)));
  overflow-wrap: anywhere;
  padding: 10px 14px;
  border: 1px solid #d9dce6;
  border-left: 4px solid #6b7188;
  border-radius: 12px;
  background: #ffffff;
  color: #1f2330;
  font-size: .85rem;
  line-height: 1.45;
  white-space: pre-line;
  animation: surge .4s .7s backwards;
}
.motivos.subiu, .motivos.caiu, .motivos.neutro { color: #1f2330; background: #ffffff; }
.motivos.subiu { border-left-color: #15803d; }
.motivos.caiu { border-left-color: #c62828; }
.motivos span { display: block; margin-bottom: 4px; color: #6b7188; font-size: .75rem; font-weight: 700; }
.motivos em { margin-left: 6px; padding: 1px 8px; border-radius: 10px; font-size: .85rem; font-style: normal; }

.novo { animation: pop .35s .5s backwards cubic-bezier(.34, 1.56, .64, 1); }
.cresce path { stroke-dasharray: 100; animation: cresce .6s ease-out; }

.subiu { color: #15803d; background: rgba(21, 128, 61, .12); }
.caiu { color: #c62828; background: rgba(198, 40, 40, .1); }
.neutro { color: #5f667c; background: rgba(95, 102, 124, .12); }

.cartao {
  padding: 28px;
  border: 1px solid #d9dce6;
  border-radius: 18px;
  background: #ffffff;
  color: #1f2330;
  animation: entra .45s ease-out;
}
.cartao h2 { color: #a8690a; font-size: 1.7rem; margin: 0 0 12px; padding: 0; }
.cartao hr { border: 0; border-top: 1px solid #d9dce6; margin: 14px 0; }

.linha { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.linha i { width: 26px; height: 26px; border-radius: 50%; background: #e9ac2e; color: #1f2330; font-style: normal; font-weight: 700; text-align: center; line-height: 26px; }
.linha span:first-of-type { flex: 1; font-weight: 600; }
.linha em { font-style: normal; font-size: .85rem; font-weight: 700; background: none; }
.linha b { color: #a8690a; }
.total { display: flex; align-items: center; gap: 12px; color: #5f667c; }
.total span { flex: 1; }
.total em { font-style: normal; font-weight: 700; background: none; }
.total b { color: #a8690a; font-size: 2rem; }

@keyframes pop { from { transform: scale(.3); opacity: 0; } }
@keyframes cresce { from { stroke-dashoffset: 100; } to { stroke-dashoffset: 0; } }
@keyframes entra { from { transform: translateY(30px); opacity: 0; } }
@keyframes surge { from { opacity: 0; } }
</style>
"""


CABECALHOS = {"subiu": "POR QUE SUBIU?", "caiu": "POR QUE CAIU?", "neutro": "POR QUÊ?"}


def pct(p):
    s = f"{p * 100:.1f}".rstrip("0").rstrip(".")
    return f"{s}%"


def pp(d):
    s = f"{abs(d) * 100:.1f}".rstrip("0").rstrip(".")
    return f"{'+' if d >= 0 else '-'}{s} p.p."


def classe(d):
    if not d:
        return "neutro"
    return "subiu" if d > 0 else "caiu"


@st.cache_data(show_spinner="Calculando as chances...")
def prever_csv(caminho):
    salvo = prever.carregar_modelo()
    return prever.arvores(prever.prever_paciente(carpediem.carregar(caminho), salvo, explicar=3))


def ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def montar(dados):
    nos = [dados["inicio"]] + dados["passos"]
    for i, n in enumerate(nos):
        n["x"] = 0 if i == 0 else (40 if i % 2 else -40)
        n["variacao"] = None if i == 0 else round(n["prob"] - nos[i - 1]["prob"], 6)
    return dados["titulo"], nos


def html_aresta(a, b, nova):
    x0, x1 = 160 + a["x"], 160 + b["x"]
    return (f'<svg class="{"cresce" if nova else ""}" width="320" height="60">'
            f'<path pathLength="100" d="M{x0} 0 C{x0} 30 {x1} 30 {x1} 60"/></svg>')


def html_no(n, foco):
    motivos = ""
    if foco and n["variacao"] is not None:
        d = n["variacao"]
        motivos = (f'<div class="motivos {classe(d)}"><span>{CABECALHOS[classe(d)]}'
                   f'<em class="{classe(d)}">{pp(d) if d else "sem mudança"}</em></span>'
                   f'{escape(n["justificativa"])}</div>')
    return (f'<div class="no {"foco novo" if foco else ""}" style="--x: {n["x"]}px">'
            f'<b>{escape(n["rotulo"])}</b><strong>{pct(n["prob"])}</strong>'
            f'<div class="barra"><div style="width: {n["prob"] * 100}%"></div></div>{motivos}</div>')


def html_arvore(nos, passo):
    partes = []
    for i, n in enumerate(nos[:passo + 1]):
        if i > 0:
            partes.append(html_aresta(nos[i - 1], n, i == passo))
        partes.append(html_no(n, i == passo))
    return f'<div class="arvore">{"".join(partes)}</div>'


def html_resumo(nos):
    linhas = ""
    for i, n in enumerate(nos[1:], 1):
        d = n["variacao"]
        var = f'<em class="{classe(d)}">{pp(d)}</em>' if d else ""
        linhas += (f'<div class="linha"><i>{i}</i><span>{escape(n["rotulo"])}</span>'
                   f'{var}<b>{pct(n["prob"])}</b></div>')
    de, para = nos[0]["prob"], nos[-1]["prob"]
    d = round(para - de, 6)
    var = f'<em class="{classe(d)}">{pp(d)}</em>' if d else ""
    return (f'<div class="cartao"><h2>Evolução da probabilidade</h2>{linhas}<hr>'
            f'<div class="total"><span>De {pct(de)} para</span>{var}<b>{pct(para)}</b></div></div>')


st.set_page_config(page_title="Chance de PAV", layout="wide")
st.html(CSS)

if "passo" not in st.session_state:
    st.session_state.passo = 0
    st.session_state.inicio = time.time()


def ir_para(i):
    st.session_state.passo = i
    st.session_state.inicio = time.time()


caminho = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PASTA, "exemplo_pacientes.csv")
if caminho.endswith(".json"):
    titulo, nos = montar(ler_json(caminho))
else:
    try:
        pacientes = prever_csv(caminho)
    except ValueError as erro:
        st.error(str(erro))
        st.stop()
    if not pacientes:
        st.error("Nenhum dia intubado neste arquivo.")
        st.stop()
    escolhido = st.columns([1, 3])[0].selectbox("Paciente", list(pacientes), on_change=ir_para, args=(0,))
    titulo, nos = montar(pacientes[escolhido])


def duracao(n):
    return min(4 + 0.035 * len(n["justificativa"]), 13)


@st.fragment(run_every=1)
def tela():
    fim = len(nos)
    passo = st.session_state.passo
    if (st.session_state.get("auto", True) and passo < fim
            and time.time() - st.session_state.inicio >= duracao(nos[passo])):
        passo += 1
        ir_para(passo)

    if passo == fim:
        info = "Visão geral"
    elif passo == 0:
        info = f"Ponto de partida · {fim - 1} passos"
    else:
        info = f"Passo {passo} de {fim - 1}"

    st.html(f'<div class="topo"><h1>{escape(titulo)}</h1><p>{info}</p>'
            f'<div class="progresso"><div style="width: {(passo + 1) / (fim + 1) * 100}%"></div></div></div>')

    st.markdown(f'<div class="janela">{html_arvore(nos, passo)}</div>', unsafe_allow_html=True)
    if passo == fim:
        st.html(html_resumo(nos))
    voltar, proximo, reiniciar = st.columns(3)
    voltar.button("Voltar", on_click=ir_para, args=(passo - 1,), disabled=passo == 0, width="stretch")
    proximo.button("Próximo", on_click=ir_para, args=(passo + 1,), disabled=passo == fim, width="stretch")
    reiniciar.button("Reiniciar", on_click=ir_para, args=(0,), width="stretch")
    st.toggle("Automático", value=True, key="auto", on_change=ir_para, args=(passo,))


tela()
