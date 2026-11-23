import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from physionet_modelV1 import carpediem

RESULTADOS = Path(__file__).parent / 'resultados'
ARQUIVO_MODELO = RESULTADOS / 'modelo_pav.joblib'
N_FOLDS = 5
SEMENTE = 42
CORTES = [0.05, 0.075, 0.10, 0.15, 0.20]

def logistica():
    preprocessamento = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), carpediem.NUMERICAS),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), carpediem.CATEGORICAS),
    ], verbose_feature_names_out=False)
    return make_pipeline(preprocessamento, LogisticRegression(C=0.003, max_iter=5000))

def boosting():
    preprocessamento = ColumnTransformer([
        ('num', 'passthrough', carpediem.NUMERICAS),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), carpediem.CATEGORICAS),
    ], sparse_threshold=0)
    return make_pipeline(preprocessamento, HistGradientBoostingClassifier(learning_rate=0.03, max_iter=400, max_leaf_nodes=8, min_samples_leaf=80, l2_regularization=5.0, random_state=SEMENTE))

def media():
    return VotingClassifier([('logistica', logistica()), ('boosting', boosting())], voting='soft')

MODELOS = {'Regressão logística': logistica, 'Gradient boosting': boosting, 'Média (modelo final)': media}

def validar(base, fabrica):
    X, y = base[carpediem.FEATURES], base['Alvo']
    probabilidades = np.zeros(len(base))
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    for treino, teste in folds.split(X, y, groups=base[carpediem.PACIENTE]):
        modelo = fabrica().fit(X.iloc[treino], y.iloc[treino])
        probabilidades[teste] = modelo.predict_proba(X.iloc[teste])[:, 1]
    return probabilidades

def metricas(y, p):
    return {'AUROC': roc_auc_score(y, p), 'AUPRC': average_precision_score(y, p),
            'Brier': brier_score_loss(y, p)}

def limiares(y, p, cortes=CORTES):
    y, p = np.asarray(y), np.asarray(p)
    linhas = []
    for corte in cortes:
        alerta = p >= corte
        vp, fp = (alerta & (y == 1)).sum(), (alerta & (y == 0)).sum()
        fn, vn = (~alerta & (y == 1)).sum(), (~alerta & (y == 0)).sum()
        dividir = lambda a, b: a / b if b else np.nan
        linhas.append({'Corte': corte, 'Pct_em_alerta': alerta.mean() if len(p) else np.nan,
                       'Sensibilidade': dividir(vp, vp + fn), 'Especificidade': dividir(vn, vn + fp),
                       'VPP': dividir(vp, vp + fp), 'VPN': dividir(vn, vn + fn)})
    return pd.DataFrame(linhas).set_index('Corte')

def odds_ratios(modelo):
    coef = modelo[-1].coef_[0]
    numericas = modelo[0].named_transformers_['num']
    n = len(carpediem.NUMERICAS)
    media = np.r_[numericas[-1].mean_, np.zeros(len(coef) - n)]
    desvio = np.r_[numericas[-1].scale_, np.ones(len(coef) - n)]
    preenchimento = np.r_[numericas[0].statistics_, np.zeros(len(coef) - n)]
    tabela = pd.DataFrame({'Variavel': modelo[0].get_feature_names_out(), 'Coeficiente': coef, 'Odds_ratio': np.exp(coef),
                           'Media': media, 'Desvio_padrao': desvio, 'Valor_se_faltante': preenchimento,
                           'Coef_por_unidade': coef / desvio, 'Odds_ratio_por_unidade': np.exp(coef / desvio)})
    return tabela.reindex(tabela['Coeficiente'].abs().sort_values(ascending=False).index)


def intercepto_por_unidade(modelo):
    tabela = odds_ratios(modelo)
    return modelo[-1].intercept_[0] - (tabela['Coef_por_unidade'] * tabela['Media']).sum()

def treinar(base, horizonte, metricas_validacao=None):
    modelo = media().fit(base[carpediem.FEATURES], base['Alvo'])
    return {'modelo': modelo, 'horizonte': horizonte, 'prevalencia': base['Alvo'].mean(), 'metricas': metricas_validacao, 'pacientes': set(base[carpediem.PACIENTE])}

def main():
    parser = argparse.ArgumentParser(description='Treina o modelo de chance de PAV no CarpeDiem.')
    parser.add_argument('csv', nargs='?', default=carpediem.ARQUIVO_PADRAO)
    parser.add_argument('--horizonte', type=int, default=carpediem.HORIZONTE_PADRAO,
                        help='dias à frente em que o início de PAV conta como evento')
    args = parser.parse_args()

    try:
        df = carpediem.carregar(args.csv)
    except ValueError as erro:
        parser.error(str(erro))
    base = carpediem.montar_base(df, args.horizonte)
    y = base['Alvo']

    RESULTADOS.mkdir(exist_ok=True)
    probabilidades = {nome: validar(base, fabrica) for nome, fabrica in MODELOS.items()}
    resumo = pd.DataFrame({nome: metricas(y, p) for nome, p in probabilidades.items()}).T
    resumo.index.name = 'Modelo'
    resumo.loc['Referência (prevalência)'] = [0.5, y.mean(), brier_score_loss(y, np.full(len(y), y.mean()))]

    salvo = treinar(base, args.horizonte, resumo.loc['Média (modelo final)'].to_dict())
    joblib.dump(salvo, ARQUIVO_MODELO)

    ors = odds_ratios(salvo['modelo'].named_estimators_['logistica'])

    resumo.round(4).to_csv(RESULTADOS / 'metricas.csv')
    ors.round(4).to_csv(RESULTADOS / 'odds_ratios.csv', index=False)
    limiares(y, probabilidades['Média (modelo final)']).round(4).to_csv(RESULTADOS / 'limiares.csv')
    previsoes = base[carpediem.CHAVE + ['Alvo']].copy()
    for nome, p in probabilidades.items():
        previsoes[f'Prob_{nome}'] = p.round(4)
    previsoes.to_csv(RESULTADOS / 'probabilidades_oof.csv', index=False)

if __name__ == '__main__':
    main()
