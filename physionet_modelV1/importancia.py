import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from physionet_modelV1 import carpediem
from physionet_modelV1.modelo_risco_pav import ARQUIVO_MODELO, N_FOLDS, RESULTADOS, SEMENTE, media
from physionet_modelV1.prever import ORIGINAIS, grupo

REPETICOES = 5


def grupos_de(colunas):
    return pd.Series(colunas.map(grupo), index=colunas)


def importancia_permutacao(base, grupos, repeticoes=REPETICOES):
    X, y = base[carpediem.FEATURES], base['Alvo']
    rng = np.random.default_rng(SEMENTE)
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    quedas = {g: [] for g in grupos.unique()}
    for treino, teste in folds.split(X, y, groups=base[carpediem.PACIENTE]):
        modelo = media().fit(X.iloc[treino], y.iloc[treino])
        X_teste, y_teste = X.iloc[teste].reset_index(drop=True), y.iloc[teste]
        referencia = roc_auc_score(y_teste, modelo.predict_proba(X_teste)[:, 1])
        for g in quedas:
            colunas = list(grupos.index[grupos == g])
            for _ in range(repeticoes):
                embaralhado = X_teste.copy()
                embaralhado[colunas] = X_teste[colunas].iloc[rng.permutation(len(X_teste))].to_numpy()
                quedas[g].append(referencia - roc_auc_score(y_teste, modelo.predict_proba(embaralhado)[:, 1]))
    return pd.DataFrame({'Queda_AUROC': {g: np.mean(v) for g, v in quedas.items()},
                         'Queda_AUROC_dp': {g: np.std(v) for g, v in quedas.items()}})


def direcao(base, grupos, modelo):
    X = base[carpediem.FEATURES]
    linhas = {}
    for g in grupos.unique():
        colunas = [c for c in grupos.index[grupos == g] if c in carpediem.NUMERICAS]
        principal = [c for c in colunas if c in ORIGINAIS] or colunas
        if not principal:
            continue
        c = principal[0]
        baixo, alto = X[c].quantile([0.25, 0.75])
        if baixo == alto:
            baixo, alto = X[c].min(), X[c].max()
        if baixo == alto or pd.isna(baixo):
            continue
        chances = [modelo.predict_proba(X.assign(**{c: v}))[:, 1].mean() for v in (baixo, alto)]
        linhas[g] = {'Variavel': c, 'Valor_baixo': baixo, 'Valor_alto': alto,
                     'Efeito_pp': (chances[1] - chances[0]) * 100}
    return pd.DataFrame.from_dict(linhas, orient='index')


def main():
    parser = argparse.ArgumentParser(description='Quais variáveis mais pesam no modelo de chance de PAV, no geral.')
    parser.add_argument('csv', nargs='?', default=carpediem.ARQUIVO_PADRAO)
    parser.add_argument('--modelo', default=ARQUIVO_MODELO)
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()

    salvo = joblib.load(args.modelo)
    base = carpediem.montar_base(carpediem.carregar(args.csv), salvo['horizonte'])
    grupos = grupos_de(base[carpediem.FEATURES].columns)

    tabela = importancia_permutacao(base, grupos).join(direcao(base, grupos, salvo['modelo']))
    tabela = tabela.sort_values('Queda_AUROC', ascending=False)
    tabela.index.name = 'Grupo'
    tabela.round(4).to_csv(RESULTADOS / 'importancia.csv')

    formatar = lambda v: '-' if pd.isna(v) else f'{v:g}'
    exibir = tabela.head(args.top).assign(
        Queda_AUROC=lambda t: t['Queda_AUROC'].map('{:.4f}'.format),
        Queda_AUROC_dp=lambda t: t['Queda_AUROC_dp'].map('{:.4f}'.format),
        Valor_baixo=lambda t: t['Valor_baixo'].map(formatar),
        Valor_alto=lambda t: t['Valor_alto'].map(formatar),
        Efeito_pp=lambda t: t['Efeito_pp'].map(lambda v: '-' if pd.isna(v) else f'{v:+.2f}'),
    ).fillna('-')
    print(exibir.to_string())


if __name__ == '__main__':
    main()
