import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from physionet_modelV1 import carpediem, prever
from physionet_modelV1.modelo_risco_pav import ARQUIVO_MODELO, RESULTADOS, limiares, metricas

FAIXAS = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
ROTULOS_FAIXAS = ['< 5%', '5–10%', '10–15%', '15–20%', '20–30%', '≥ 30%']

def prever_base(df, salvo):
    base = carpediem.montar_base(df, salvo['horizonte'])
    base['Chance_PAV'] = salvo['modelo'].predict_proba(base[carpediem.FEATURES])[:, 1]
    return base

def intervalo_confianca(base, n=1000, semente=0):
    grupos = list(base.groupby(carpediem.PACIENTE).indices.values())
    if len(grupos) < 2:
        return pd.DataFrame({'IC95_inferior': np.nan, 'IC95_superior': np.nan}, index=['AUROC', 'AUPRC', 'Brier'])
    y, p = base['Alvo'].to_numpy(), base['Chance_PAV'].to_numpy()
    rng = np.random.default_rng(semente)
    valores = []
    for _ in range(n):
        idx = np.concatenate([grupos[i] for i in rng.integers(0, len(grupos), len(grupos))])
        if y[idx].min() != y[idx].max():
            valores.append(list(metricas(y[idx], p[idx]).values()))
    if not valores:
        valores = [[np.nan] * 3]
    inferior, superior = np.percentile(valores, [2.5, 97.5], axis=0)
    return pd.DataFrame({'IC95_inferior': inferior, 'IC95_superior': superior}, index=['AUROC', 'AUPRC', 'Brier'])

def tabela_faixas(base):
    faixa = pd.cut(base['Chance_PAV'], FAIXAS, right=False, labels=ROTULOS_FAIXAS)
    tabela = base.groupby(faixa, observed=False).agg(
        Dias=('Alvo', 'size'), Chance_media=('Chance_PAV', 'mean'), PAV_de_fato=('Alvo', 'mean'))
    tabela.insert(1, 'Pct_dias', tabela['Dias'] / len(base))
    tabela.index.name = 'Chance prevista'
    return tabela

def por_paciente(base):
    tabela = base.groupby(carpediem.PACIENTE).agg(
        Dias=('Alvo', 'size'), Chance_media=('Chance_PAV', 'mean'), Chance_maxima=('Chance_PAV', 'max'),
        Chance_ultimo_dia=('Chance_PAV', 'last'), Teve_PAV=('Alvo', 'max'))
    tabela['Dia_da_maxima'] = base.loc[base.groupby(carpediem.PACIENTE)['Chance_PAV'].idxmax()].set_index(
        carpediem.PACIENTE)[carpediem.DIA]
    return tabela.sort_values('Chance_maxima', ascending=False)

def auroc_por_paciente(tabela):
    if tabela['Teve_PAV'].nunique() < 2:
        return np.nan
    return roc_auc_score(tabela['Teve_PAV'], tabela['Chance_maxima'])

def avaliar(df, salvo):
    base = prever_base(df, salvo)
    if base['Alvo'].nunique() == 2:
        resumo = pd.Series(metricas(base['Alvo'], base['Chance_PAV'])).rename('Valor').to_frame()
    else:
        brier = ((base['Chance_PAV'] - base['Alvo']) ** 2).mean() if len(base) else np.nan
        resumo = pd.DataFrame({'Valor': [np.nan, np.nan, brier]}, index=['AUROC', 'AUPRC', 'Brier'])
    resumo = resumo.join(intervalo_confianca(base))
    sobrepostos = set(base[carpediem.PACIENTE]) & salvo.get('pacientes', set())
    return base, resumo, tabela_faixas(base), sobrepostos

def relatorio(base, resumo, faixas, sobrepostos, salvo):
    if sobrepostos:
        print(f'{len(sobrepostos)} sobrepostos')
    y, p = base['Alvo'], base['Chance_PAV']
    if base.empty:
        print('Nenhum dia avaliável')
        return

    print(f'Dias avaliados: {len(base)} | pacientes: {base[carpediem.PACIENTE].nunique()} | '
          f'com PAV nos próximos {salvo["horizonte"]} dias: {y.sum()} ({y.mean():.1%})\n')

    if not base['Alvo'].nunique() == 2:
        desfecho = 'com PAV' if y.all() else 'sem PAV'
        print(f'Todos os dias avaliados estão {desfecho} nos próximos {salvo["horizonte"]} dias.')

    referencia = {'AUROC': 0.5, 'AUPRC': y.mean(), 'Brier': y.mean() * (1 - y.mean())}
    tabela = resumo.assign(Referencia=pd.Series(referencia))
    if salvo.get('metricas'):
        tabela['No_treino'] = pd.Series(salvo['metricas'])
    print(tabela.round(3).to_string())
    print('\nPor faixa de chance:')
    porcentagem = lambda v: '-' if pd.isna(v) else f'{v:.1%}'
    print(faixas.assign(**{c: faixas[c].map(porcentagem) for c in ['Pct_dias', 'Chance_media', 'PAV_de_fato']}).to_string())
    print('\nSensibilidade e especificidade por corte:')
    cortes = limiares(y, p)
    cortes.index = cortes.index.map(lambda c: f'≥ {c:.1%}')
    print(cortes.map(porcentagem).to_string())

    pacientes_tabela = por_paciente(base)
    print(f'\nPor paciente (ordenado pela maior chance; "Teve_PAV" = PAV começou numa janela de {salvo["horizonte"]} dias depois de algum dia avaliado):')
    print(pacientes_tabela.assign(
        **{c: pacientes_tabela[c].map(porcentagem) for c in ['Chance_media', 'Chance_maxima', 'Chance_ultimo_dia']},
        Teve_PAV=pacientes_tabela['Teve_PAV'].map({1: 'sim', 0: 'não'})).to_string())
    auroc = auroc_por_paciente(pacientes_tabela)
    if not np.isnan(auroc):
        print(f'\nAUROC por paciente (chance máxima x teve PAV): {auroc:.3f}')


def salvar(base, resumo, faixas, destino):
    destino.mkdir(parents=True, exist_ok=True)
    resumo.round(4).to_csv(destino / 'metricas.csv')
    faixas.round(4).to_csv(destino / 'faixas.csv')
    limiares(base['Alvo'], base['Chance_PAV']).round(4).to_csv(destino / 'limiares.csv')
    base[carpediem.CHAVE + ['Alvo', 'Chance_PAV']].round(4).to_csv(destino / 'previsoes.csv', index=False)
    por_paciente(base).round(4).to_csv(destino / 'por_paciente.csv')
    print(f'\nArquivos salvos em {destino}')


def main():
    parser = argparse.ArgumentParser(description='Avalia o modelo salvo numa base com rótulo.')
    parser.add_argument('csv', help='base no formato do CarpeDiem, com episode_category para medir acerto')
    parser.add_argument('--modelo', default=ARQUIVO_MODELO)
    parser.add_argument('--saida', type=Path, default=RESULTADOS / 'avaliacao')
    args = parser.parse_args()

    salvo = joblib.load(args.modelo)
    try:
        df = carpediem.carregar(args.csv)
    except ValueError as erro:
        parser.error(str(erro))
    if 'episode_category' not in df.columns:
        prever.imprimir(prever.prever_paciente(df, salvo), salvo)
        return
    base, resumo, faixas, sobrepostos = avaliar(df, salvo)
    relatorio(base, resumo, faixas, sobrepostos, salvo)
    salvar(base, resumo, faixas, args.saida)


if __name__ == '__main__':
    main()
