import argparse
import re

import joblib
import numpy as np
import pandas as pd

from physionet_modelV1 import carpediem
from physionet_modelV1.modelo_risco_pav import ARQUIVO_MODELO

PREFIXOS_DERIVADAS = ('delta_', f'media{carpediem.JANELA}d_', f'tendencia{carpediem.JANELA}d_', 'aumento_')
ORIGINAIS = [*carpediem.ESTATICAS_NUMERICAS, *carpediem.ESTATICAS_CATEGORICAS, *carpediem.DIARIAS]


def carregar_modelo(caminho=ARQUIVO_MODELO):
    return joblib.load(caminho)


def grupo(coluna):
    if coluna in carpediem.TEMPO:
        return 'tempo de internação/ventilação'
    for prefixo in PREFIXOS_DERIVADAS:
        coluna = coluna.removeprefix(prefixo)
    return re.sub(r'_(avg|max|min)$', '', coluna)


def motivos(entrada, chance, salvo, n):
    X = entrada[carpediem.FEATURES]
    grupos = pd.Series(X.columns.map(grupo), index=X.columns)
    pares = []
    for posicoes in entrada.groupby(carpediem.PACIENTE, sort=False).indices.values():
        for ontem, hoje in zip(posicoes[:-1], posicoes[1:]):
            a, b = X.iloc[ontem], X.iloc[hoje]
            mudou = ~((a == b) | (a.isna() & b.isna()))
            pares += [(ontem, hoje, g) for g in grupos[mudou].unique()]
    resultado = {}
    if not pares:
        return resultado
    ontens, hojes, nomes = map(np.array, zip(*pares))
    simulado = X.iloc[ontens].reset_index(drop=True)
    for g in np.unique(nomes):
        colunas = list(grupos.index[grupos == g])
        linhas = np.flatnonzero(nomes == g)
        simulado.loc[linhas, colunas] = X.iloc[hojes[linhas]][colunas].to_numpy()
    efeito = salvo['modelo'].predict_proba(simulado)[:, 1] - chance[ontens]
    for hoje in np.unique(hojes):
        linhas = np.flatnonzero(hojes == hoje)
        maiores = linhas[np.argsort(-np.abs(efeito[linhas]))[:n]]
        resultado[hoje] = [(nomes[i], valores(entrada, ontens[i], hoje, grupos, nomes[i]), efeito[i]) for i in maiores]
    return resultado


def valores(entrada, ontem, hoje, grupos, nome):
    principal = [c for c in grupos.index[grupos == nome] if c in ORIGINAIS]
    if not principal or nome == grupo(carpediem.TEMPO[0]):
        return ''
    formatar = lambda v: '-' if pd.isna(v) else (f'{v:g}' if isinstance(v, (int, float, np.number)) else str(v))
    c = principal[0]
    return f'{c} {formatar(entrada.at[ontem, c])} → {formatar(entrada.at[hoje, c])}'


def prever_paciente(dias, salvo, explicar=0):
    dias = carpediem.garantir_paciente(carpediem.padronizar(dias))
    entrada = carpediem.montar_entrada(dias)
    chance = salvo['modelo'].predict_proba(entrada[carpediem.FEATURES])[:, 1]
    resultado = entrada[carpediem.CHAVE].assign(Chance_PAV=chance)
    if explicar:
        resultado.attrs['motivos'] = motivos(entrada, chance, salvo, explicar)
    return resultado


def linha_motivo(nome_grupo, mudanca, efeito):
    return f'{efeito * 100:+5.1f} p.p.  {nome_grupo}' + (f' ({mudanca})' if mudanca else '')


def arvores(resultado):
    explicacoes = resultado.attrs.get('motivos', {})
    saida = {}
    for paciente, dias in resultado.groupby(carpediem.PACIENTE, sort=False):
        nome = 'Paciente' if paciente == 'sem_id' else f'Paciente {paciente}'
        nos = [{'rotulo': f'Dia {int(dia.hosp_day)}', 'prob': float(dia.Chance_PAV),
                'justificativa': '\n'.join(linha_motivo(*m) for m in explicacoes.get(i, []))
                or 'Nenhuma variável mudou em relação ao dia anterior.'}
               for i, dia in dias.iterrows()]
        nos[0]['justificativa'] = 'Primeiro dia intubado: ainda não há dia anterior para comparar.'
        saida[paciente] = {'titulo': f'{nome}: chance de PAV', 'inicio': nos[0], 'passos': nos[1:]}
    return saida


def imprimir(resultado, salvo):
    horizonte, media = salvo['horizonte'], salvo['prevalencia']
    explicacoes = resultado.attrs.get('motivos')
    if resultado.empty:
        print('Nenhum dia intubado')
        return
    for paciente, dias in resultado.groupby(carpediem.PACIENTE, sort=False):
        nome = 'Paciente' if paciente == 'sem_id' else f'Paciente {paciente}'
        print(f'{nome}: chance de PAV')
        for _, dia in dias.iterrows():
            print(f'dia {int(dia.hosp_day)} de internação: {dia.Chance_PAV:6.1%}')
        if explicacoes is not None and len(dias) > 1:
            print('\nJustificativa (o que mais mudou a chance em relação ao dia anterior):')
            anterior = None
            for i, dia in dias.iterrows():
                if anterior is not None:
                    print(f'dia {int(anterior.hosp_day)} → {int(dia.hosp_day)}: '
                          f'{(dia.Chance_PAV - anterior.Chance_PAV) * 100:+.1f} p.p.')
                    for motivo in explicacoes.get(i, []):
                        print(f'    {linha_motivo(*motivo)}')
                anterior = dia
        print("\n")


def main():
    parser = argparse.ArgumentParser(description='Chance de um paciente intubado ter PAV nos próximos dias.')
    parser.add_argument('csv', help='dias de internação do paciente')
    parser.add_argument('--modelo', default=ARQUIVO_MODELO)
    parser.add_argument('--explicar', type=int, default=3,
                        help='quantas variáveis mostrar na justificativa de cada dia (0 desliga)')
    args = parser.parse_args()

    salvo = carregar_modelo(args.modelo)
    try:
        dias = carpediem.carregar(args.csv)
    except ValueError as erro:
        parser.error(str(erro))
    imprimir(prever_paciente(dias, salvo, args.explicar), salvo)


if __name__ == '__main__':
    main()
