import argparse

import numpy as np
import pandas as pd

from physionet_modelV1 import carpediem

ARQUIVO = 'physionet_modelV1/exemplo_pacientes.csv'
N_PACIENTES = 40
SEMENTE = 7
PROPORCAO_PAV = 0.35
PROPORCAO_FALSO_ALARME = 0.15


def piora_por_dia(n_dias, inicio, intensidade, duracao=4):
    piora = np.zeros(n_dias)
    for k, fator in zip(range(-2, duracao), [0.4, 0.8, 1, 1, 0.7, 0.4]):
        if 0 <= inicio + k < n_dias:
            piora[inicio + k] = fator * intensidade
    return piora


def gerar_paciente(numero, rng):
    n_dias = int(rng.integers(8, 17))
    dia_intubacao = int(rng.integers(1, 4))
    intubado = (np.arange(1, n_dias + 1) >= dia_intubacao).astype(int)

    sorteio = rng.random()
    tem_pav = sorteio < PROPORCAO_PAV
    falso_alarme = PROPORCAO_PAV <= sorteio < PROPORCAO_PAV + PROPORCAO_FALSO_ALARME
    inicio = int(rng.integers(dia_intubacao + 3, n_dias - 1)) if (tem_pav or falso_alarme) else None
    intensidade = rng.uniform(0.1, 1.0) if tem_pav else rng.uniform(0.4, 0.9) if falso_alarme else 0
    piora = piora_por_dia(n_dias, inicio, intensidade) if inicio is not None else np.zeros(n_dias)

    def ruido(escala):
        return rng.normal(0, escala, n_dias)

    idade = int(rng.integers(35, 86))
    dpoc, diabetes, icc, cancer = (rng.random(4) < [0.25, 0.3, 0.15, 0.2]).astype(int)
    charlson = (idade >= 50) + (idade >= 60) + (idade >= 70) + (idade >= 80) + dpoc + diabetes + icc + 2 * cancer
    vent = np.where(intubado == 1, 1.0, np.nan)

    fio2 = np.clip(np.round((40 + 25 * piora + ruido(4)) / 5) * 5, 30, 100) * vent
    peep = np.clip(np.round(6 + 4 * piora + ruido(0.8)), 5, 18) * vent
    plato = (18 + 5 * piora + ruido(1.2)) * vent
    pao2 = 100 - 10 * piora + ruido(8)
    temperatura = 98.6 + 2.2 * piora + ruido(0.35)
    leucocitos = rng.uniform(7, 12) + 5 * piora + ruido(0.7)
    pas = rng.uniform(105, 130) - 10 * piora + ruido(4)
    pad = pas * 0.55 + ruido(2)
    abx_hoje = np.zeros(n_dias, dtype=int)
    if rng.random() < 0.4:
        abx_hoje[dia_intubacao - 1:dia_intubacao + 2] = 1
    if tem_pav:
        abx_hoje[inicio:inicio + 7] = 1
    noradrenalina = np.clip(rng.uniform(-0.05, 0.08) + 0.05 * piora + ruido(0.01), 0, None)
    crp = np.where(rng.random(n_dias) < 0.25, 40 + 90 * piora + ruido(10), np.nan)
    procalcitonina = np.where(rng.random(n_dias) < 0.3, 0.15 + 0.8 * piora + np.abs(ruido(0.05)), np.nan)
    sofa = np.clip(np.round(rng.uniform(3, 8) + 3 * piora + ruido(0.6)), 0, 20)

    dias = {
        'anonymized_id': f'EX{numero:03d}', 'hosp_day': np.arange(1, n_dias + 1),
        'icu_day': np.arange(1, n_dias + 1), 'intub_flag': intubado,
        'adm_age': idade, 'bmi': round(float(rng.uniform(20, 36)), 1), 'external_transfer_flag': int(rng.random() < 0.2),
        'is_repeat_hospitalization': 0, 'charlson_comorbidity_index': charlson,
        'immunocompromised_flag': int(rng.random() < 0.1), 'myocardial_infarction': int(rng.random() < 0.1),
        'congestive_heart_failure': icc, 'chronic_pulmonary_disease': dpoc, 'diabetes': diabetes,
        'cancer': 2 * cancer, 'renal_disease': 0, 'liver_disease': 0, 'dementia': 0, 'cerebrovascular_disease': 0,
        'gender': rng.choice(['Male', 'Female']), 'smoking_history': rng.choice(['No', 'Former', 'Yes']),
        'covid_status_enrollment': bool(rng.random() < 0.2),
        'days_on_ventilator': np.cumsum(intubado),
        'day_since_first_intubation': np.where(intubado == 1, np.arange(1, n_dias + 1) - dia_intubacao + 1, -1),
        'ecmo_flag': 0, 'hd_flag': 0, 'crrt_flag': 0,
        'norepinephrine_avg': noradrenalina.round(3), 'norepinephrine_max': (noradrenalina * 1.5).round(3),
        'days_of_abx_until_today': np.cumsum(abx_hoje), 'hydrocort_equivalent_steroid_dose_day': 0,
        'cumulative_steroid_dose_until_today': 0, 'received_tocilizumab': 0, 'received_sarilumab': 0,
        'received_remdesivir': 0,
        'temperature_avg': temperatura.round(1), 'temperature_max': (temperatura + 0.6 + np.abs(ruido(0.3))).round(1),
        'heart_rate_avg': (88 + 18 * piora + ruido(6)).round(), 'heart_rate_max': (100 + 22 * piora + ruido(6)).round(),
        'systolic_blood_pressure_avg': pas.round(), 'systolic_blood_pressure_min': (pas - 12 + ruido(3)).round(),
        'diastolic_blood_pressure_avg': pad.round(), 'diastolic_blood_pressure_min': (pad - 8 + ruido(2)).round(),
        'mean_arterial_pressure_avg': ((pas + 2 * pad) / 3).round(), 'mean_arterial_pressure_min': ((pas + 2 * pad) / 3 - 8).round(),
        'respiratory_rate_avg': (18 + 6 * piora + ruido(1.5)).round(), 'respiratory_rate_max': (23 + 8 * piora + ruido(2)).round(),
        'oxygen_saturation_avg': np.clip(97 - 4 * piora + ruido(0.8), 80, 100).round(),
        'oxygen_saturation_min': np.clip(94 - 6 * piora + ruido(1.2), 75, 100).round(),
        'rass_avg': np.where(intubado == 1, -2, 0) + np.round(ruido(0.5)), 'rass_min': np.where(intubado == 1, -3, -1),
        'gcs_eye_opening_min': np.where(intubado == 1, 2, 4), 'gcs_motor_response_min': np.where(intubado == 1, 5, 6),
        'gcs_verbal_response_min': np.where(intubado == 1, 1, 5),
        'peep_avg': peep, 'peep_max': peep + 2, 'fio2_avg': fio2, 'fio2_max': np.clip(fio2 + 10, None, 100),
        'plateau_pressure_avg': plato.round(1), 'plateau_pressure_max': (plato + 3).round(1),
        'peak_airway_pressure_avg': (plato + 6).round(1), 'peak_airway_pressure_max': (plato + 10).round(1),
        'minute_ventilation_avg': ((8.5 + 2.5 * piora + ruido(0.5)) * vent).round(1),
        'minute_ventilation_max': ((10 + 3 * piora + ruido(0.5)) * vent).round(1),
        'driving_pressure': (plato - peep).round(1),
        'ph_art_avg': (7.40 - 0.06 * piora + ruido(0.02)).round(2), 'ph_art_min': (7.37 - 0.07 * piora + ruido(0.02)).round(2),
        'po2_art_avg': pao2.round(), 'po2_art_min': (pao2 - 12).round(),
        'pco2_art_avg': (40 + 7 * piora + ruido(2)).round(), 'pco2_art_max': (44 + 8 * piora + ruido(2)).round(),
        'po2_fio2_ratio_avg': (pao2 / (fio2 / 100)).round(), 'po2_fio2_ratio_min': (0.85 * pao2 / (fio2 / 100)).round(),
        'wbc_avg': leucocitos.round(1), 'wbc_max': (leucocitos + 0.8).round(1),
        'abs_lymphocytes_avg': (1.3 - 0.4 * piora + ruido(0.1)).clip(0.2).round(2),
        'abs_neutrophils_avg': (leucocitos * 0.75).round(1), 'abs_eosinophils_avg': np.abs(0.1 + ruido(0.03)).round(2),
        'hemoglobin_avg': (rng.uniform(9.5, 13) - 0.1 * np.arange(n_dias) + ruido(0.3)).round(1),
        'platelet_avg': (rng.uniform(150, 280) - 20 * piora + ruido(10)).round(),
        'platelet_min': (rng.uniform(140, 270) - 25 * piora + ruido(10)).round(),
        'rdw_avg': (14.2 + ruido(0.3)).round(1), 'sodium_avg': (139 + ruido(2)).round(),
        'bicarbonate_avg': (24 - 3 * piora + ruido(1)).round(), 'creatinine_avg': (1.0 + 0.3 * piora + np.abs(ruido(0.1))).round(2),
        'creatinine_max': (1.1 + 0.35 * piora + np.abs(ruido(0.1))).round(2), 'bun_avg': (18 + 6 * piora + ruido(3)).round(),
        'glucose_avg': (135 + 30 * piora + ruido(12)).round(), 'albumin_avg': (3.2 - 0.3 * piora + ruido(0.1)).round(1),
        'bilirubin_avg': (0.7 + np.abs(ruido(0.15))).round(1), 'pt_avg': (13 + 0.6 * piora + ruido(0.3)).round(1),
        'ptt_avg': (31 + 2 * piora + ruido(1)).round(), 'crp_avg': np.round(crp),
        'lactic_acid_avg': (1.2 + 1.0 * piora + np.abs(ruido(0.2))).round(1), 'd_dimer_avg': np.nan,
        'ferritin_avg': np.nan, 'ldh_avg': np.nan, 'procalcitonin_avg': procalcitonina.round(2), 'sofa_score': sofa,
        'episode_category': None, 'episode_duration': np.nan,
    }
    df = pd.DataFrame(dias)
    if tem_pav:
        df.loc[inicio, ['episode_category', 'episode_duration']] = ['VAP', 7]
    return df


def gerar(n=N_PACIENTES, semente=SEMENTE):
    rng = np.random.default_rng(semente)
    df = pd.concat([gerar_paciente(i, rng) for i in range(1, n + 1)], ignore_index=True)
    colunas = list(dict.fromkeys([*carpediem.COLUNAS_ENTRADA, 'episode_duration']))
    return df[colunas]


def main():
    parser = argparse.ArgumentParser(description='Gera pacientes fictícios, com rótulo, para testar o avaliar.')
    parser.add_argument('--saida', default=ARQUIVO)
    parser.add_argument('--pacientes', type=int, default=N_PACIENTES)
    parser.add_argument('--semente', type=int, default=SEMENTE)
    args = parser.parse_args()

    df = gerar(args.pacientes, args.semente)
    df.to_csv(args.saida, index=False)
    com_pav = df.loc[df['episode_category'].eq('VAP'), 'anonymized_id'].nunique()
    print(f'{args.pacientes} pacientes fictícios ({com_pav} com PAV), {len(df)} dias -> {args.saida}')


if __name__ == '__main__':
    main()
