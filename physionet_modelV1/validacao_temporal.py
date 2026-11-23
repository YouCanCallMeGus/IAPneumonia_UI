import argparse

import joblib
import pandas as pd

from physionet_modelV1 import avaliar, carpediem
from physionet_modelV1.modelo_risco_pav import RESULTADOS, media, metricas, treinar, validar

ARQUIVO_V1 = 'data/CarpeDiem_v1/CarpeDiem_dataset.csv'
DESTINO = RESULTADOS / 'temporal'


def separar(df, ids_v1):
    antigos = df[carpediem.PACIENTE].isin(ids_v1)
    return df[antigos], df[~antigos]


def main():
    parser = argparse.ArgumentParser(description='Validação temporal do modelo de chance de PAV.')
    parser.add_argument('--v1', default=ARQUIVO_V1, help='CarpeDiem v1, só para saber quais pacientes são antigos')
    parser.add_argument('--horizonte', type=int, default=carpediem.HORIZONTE_PADRAO)
    args = parser.parse_args()

    df = carpediem.carregar()
    treino, teste = separar(df, set(pd.read_csv(args.v1, usecols=['Patient_id'])['Patient_id']))
    for nome, parte in [('Treino (pacientes da v1)', treino), ('Teste (pacientes novos)', teste)]:
        covid = parte['covid_status_enrollment'].astype(str).eq('True').mean()
        print(f'{nome}: {parte[carpediem.PACIENTE].nunique()} pacientes, {covid:.0%} dos dias com COVID')

    base_treino = carpediem.montar_base(treino, args.horizonte)
    validacao = metricas(base_treino['Alvo'], validar(base_treino, media))
    salvo = treinar(base_treino, args.horizonte, validacao)
    DESTINO.mkdir(parents=True, exist_ok=True)
    joblib.dump(salvo, DESTINO / 'modelo_temporal.joblib')

    print('Modelo treinado só nos pacientes da v1')
    base, resumo, faixas, sobrepostos = avaliar.avaliar(teste, salvo)
    avaliar.relatorio(base, resumo, faixas, sobrepostos, salvo)
    avaliar.salvar(base, resumo, faixas, DESTINO)


if __name__ == '__main__':
    main()
