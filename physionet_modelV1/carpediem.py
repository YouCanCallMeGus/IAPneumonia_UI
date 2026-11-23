import pandas as pd

ARQUIVO_PADRAO = 'data/CarpeDiem_v1.9.1/carpediem1.9.1_dataset.csv'
PACIENTE, DIA = 'anonymized_id', 'hosp_day'
CHAVE = [PACIENTE, DIA]
HORIZONTE_PADRAO = 2
JANELA = 3
DURACAO_PADRAO_EPISODIO = 7

ESTATICAS_NUMERICAS = [
    'adm_age', 'bmi', 'external_transfer_flag', 'is_repeat_hospitalization',
    'charlson_comorbidity_index', 'immunocompromised_flag', 'myocardial_infarction',
    'congestive_heart_failure', 'chronic_pulmonary_disease', 'diabetes', 'cancer',
    'renal_disease', 'liver_disease', 'dementia', 'cerebrovascular_disease',
]
ESTATICAS_CATEGORICAS = ['gender', 'smoking_history', 'covid_status_enrollment']

TEMPO = ['hosp_day', 'icu_day', 'days_on_ventilator', 'day_since_first_intubation']

SUPORTE = ['ecmo_flag', 'hd_flag', 'crrt_flag', 'norepinephrine_avg', 'norepinephrine_max']

TRATAMENTO = [
    'days_of_abx_until_today', 'hydrocort_equivalent_steroid_dose_day',
    'cumulative_steroid_dose_until_today', 'received_tocilizumab', 'received_sarilumab',
    'received_remdesivir',
]

SINAIS = [
    'temperature_avg', 'temperature_max', 'heart_rate_avg', 'heart_rate_max',
    'systolic_blood_pressure_avg', 'systolic_blood_pressure_min',
    'diastolic_blood_pressure_avg', 'diastolic_blood_pressure_min',
    'mean_arterial_pressure_avg', 'mean_arterial_pressure_min',
    'respiratory_rate_avg', 'respiratory_rate_max', 'oxygen_saturation_avg', 'oxygen_saturation_min',
    'rass_avg', 'rass_min', 'gcs_eye_opening_min', 'gcs_motor_response_min', 'gcs_verbal_response_min',
]

VENTILACAO = [
    'peep_avg', 'peep_max', 'fio2_avg', 'fio2_max', 'plateau_pressure_avg', 'plateau_pressure_max',
    'peak_airway_pressure_avg', 'peak_airway_pressure_max', 'minute_ventilation_avg',
    'minute_ventilation_max', 'driving_pressure',
    'ph_art_avg', 'ph_art_min', 'po2_art_avg', 'po2_art_min', 'pco2_art_avg', 'pco2_art_max',
    'po2_fio2_ratio_avg', 'po2_fio2_ratio_min',
]

LABORATORIO = [
    'wbc_avg', 'wbc_max', 'abs_lymphocytes_avg', 'abs_neutrophils_avg', 'abs_eosinophils_avg',
    'hemoglobin_avg', 'platelet_avg', 'platelet_min', 'rdw_avg', 'sodium_avg', 'bicarbonate_avg',
    'creatinine_avg', 'creatinine_max', 'bun_avg', 'glucose_avg', 'albumin_avg', 'bilirubin_avg',
    'pt_avg', 'ptt_avg', 'crp_avg', 'lactic_acid_avg', 'd_dimer_avg', 'ferritin_avg', 'ldh_avg',
    'procalcitonin_avg', 'sofa_score',
]

DIARIAS = [*TEMPO, *SUPORTE, *TRATAMENTO, *SINAIS, *VENTILACAO, *LABORATORIO]

COM_VARIACAO = ['temperature_max', 'wbc_max', 'fio2_avg', 'peep_avg', 'po2_fio2_ratio_min',
                'respiratory_rate_avg', 'oxygen_saturation_min', 'minute_ventilation_avg', 'sofa_score']
LINHA_DE_BASE_VENTILACAO = ['fio2_avg', 'peep_avg']

DERIVADAS = [
    'episodio_previo', 'dias_desde_ultimo_episodio', 'antibiotico_hoje',
    *[f'delta_{c}' for c in COM_VARIACAO],
    *[f'media{JANELA}d_{c}' for c in COM_VARIACAO],
    *[f'tendencia{JANELA}d_{c}' for c in COM_VARIACAO],
    *[f'aumento_{c}' for c in LINHA_DE_BASE_VENTILACAO],
]
NUMERICAS = [*ESTATICAS_NUMERICAS, *DIARIAS, *DERIVADAS]
CATEGORICAS = ESTATICAS_CATEGORICAS
FEATURES = NUMERICAS + CATEGORICAS

COLUNAS_ENTRADA = list(dict.fromkeys([*CHAVE, 'intub_flag', *ESTATICAS_NUMERICAS, *ESTATICAS_CATEGORICAS, *DIARIAS, 'episode_category']))

DE_PARA_ANTIGO = {
    'Patient_id': 'anonymized_id', 'Day_of_hospitalization': 'hosp_day', 'ICU_stay': 'icu_stay',
    'ICU_day': 'icu_day', 'Intubation_flag': 'intub_flag', 'Age': 'adm_age', 'BMI': 'bmi',
    'Gender': 'gender', 'Smoking_status': 'smoking_history', 'COVID_status': 'covid_status_enrollment',
    'External_transfer_flag': 'external_transfer_flag', 'Immunocompromised_flag': 'immunocompromised_flag',
    'SOFA_score': 'sofa_score', 'ECMO_flag': 'ecmo_flag', 'Hemodialysis_flag': 'hd_flag',
    'CRRT_flag': 'crrt_flag', 'Temperature': 'temperature_avg', 'Heart_rate': 'heart_rate_avg',
    'Systolic_blood_pressure': 'systolic_blood_pressure_avg',
    'Diastolic_blood_pressure': 'diastolic_blood_pressure_avg',
    'Mean_arterial_pressure': 'mean_arterial_pressure_avg', 'Norepinephrine_rate': 'norepinephrine_max',
    'Respiratory_rate': 'respiratory_rate_avg', 'Oxygen_saturation': 'oxygen_saturation_avg',
    'GCS_eye_opening': 'gcs_eye_opening_min', 'GCS_motor_response': 'gcs_motor_response_min',
    'GCS_verbal_response': 'gcs_verbal_response_min', 'RASS_score': 'rass_avg', 'PEEP': 'peep_avg',
    'FiO2': 'fio2_avg', 'Plateau_Pressure': 'plateau_pressure_avg', 'Minute_Ventilation': 'minute_ventilation_avg',
    'ABG_pH': 'ph_art_avg', 'ABG_PaCO2': 'pco2_art_avg', 'ABG_PaO2': 'po2_art_avg',
    'PaO2FIO2_ratio': 'po2_fio2_ratio_min', 'WBC_count': 'wbc_avg', 'Lymphocytes': 'abs_lymphocytes_avg',
    'Neutrophils': 'abs_neutrophils_avg', 'Hemoglobin': 'hemoglobin_avg', 'Platelets': 'platelet_min',
    'Bicarbonate': 'bicarbonate_avg', 'Creatinine': 'creatinine_max', 'Albumin': 'albumin_avg',
    'Bilirubin': 'bilirubin_max', 'CRP': 'crp_avg', 'D_dimer': 'd_dimer_avg', 'Ferritin': 'ferritin_avg',
    'LDH': 'ldh_avg', 'Lactic_acid': 'lactic_acid_avg', 'Procalcitonin': 'procalcitonin_avg',
    'Steroid_dose': 'hydrocort_equivalent_steroid_dose_day', 'Tocilizumab_received': 'received_tocilizumab',
    'Sarilumab_received': 'received_sarilumab', 'Remdesivir_received': 'received_remdesivir',
    'Episode_category': 'episode_category', 'Episode_duration': 'episode_duration',
}
TABAGISMO_ANTIGO = {'Never Smoker': 'No', 'Past Smoker': 'Former', 'Current Smoker': 'Yes'}

def eh_formato_antigo(df):
    return 'Patient_id' in df.columns and PACIENTE not in df.columns

def padronizar(df):
    if not eh_formato_antigo(df):
        return df
    df = df.rename(columns=DE_PARA_ANTIGO)
    df = df[[c for c in DE_PARA_ANTIGO.values() if c in df.columns]].copy()
    if 'minute_ventilation_avg' in df:
        df['minute_ventilation_avg'] = df['minute_ventilation_avg'] / 1000
    df['smoking_history'] = df['smoking_history'].replace(TABAGISMO_ANTIGO)
    df['episode_category'] = df['episode_category'].replace({'Non-PNA-ctrl': 'NPC'})
    covid = df['covid_status_enrollment']
    df['covid_status_enrollment'] = covid.where(covid.isna(), covid.astype(float).eq(1))
    if 'hosp_day' not in df:
        ordem = df.sort_values([PACIENTE, 'icu_stay', 'icu_day'])
        df['hosp_day'] = ordem.groupby(PACIENTE).cumcount().reindex(df.index) + 1
    df = df.sort_values(CHAVE).drop_duplicates(CHAVE)
    df['days_on_ventilator'] = df.groupby(PACIENTE)['intub_flag'].cumsum()
    return df.reset_index(drop=True)

def garantir_paciente(df):
    if PACIENTE in df.columns:
        return df
    return df.assign(**{PACIENTE: 'sem_id'})

def carregar(caminho=ARQUIVO_PADRAO):
    df = garantir_paciente(padronizar(pd.read_csv(caminho, low_memory=False)))
    faltando = [c for c in [DIA, 'intub_flag'] if c not in df.columns]
    if faltando:
        raise ValueError(f'{caminho} não tem as colunas {faltando}; o modelo usa o formato do CarpeDiem (v1.9.1, ou v1/v1.8, que são convertidas).')
    return df.sort_values(CHAVE).reset_index(drop=True)

def _pneumonia(df):
    return df['episode_category'].isin(['VAP', 'HAP', 'CAP'])

def alvo_pav(df, horizonte=HORIZONTE_PADRAO):
    inicio = df['episode_category'].eq('VAP')
    dias_pav = set(zip(df.loc[inicio, PACIENTE], df.loc[inicio, DIA]))
    alvo = pd.Series(0, index=df.index)
    for k in range(1, horizonte + 1):
        futuro = zip(df[PACIENTE], df[DIA] + k)
        alvo |= pd.Series([chave in dias_pav for chave in futuro], index=df.index).astype(int)
    return alvo

def em_episodio(df):
    duracao = df.get('episode_duration', pd.Series(float('nan'), index=df.index))
    fim = (df[DIA] + duracao.fillna(DURACAO_PADRAO_EPISODIO) - 1).where(_pneumonia(df))
    fim_vigente = fim.groupby(df[PACIENTE]).cummax().groupby(df[PACIENTE]).ffill()
    return df[DIA].le(fim_vigente)

def adicionar_features(df):
    extras = [c for c in COLUNAS_ENTRADA if c not in df.columns]
    df = pd.concat([df, pd.DataFrame(float('nan'), index=df.index, columns=extras)], axis=1)
    df = df.sort_values(CHAVE).reset_index(drop=True)
    paciente = df.groupby(PACIENTE, sort=False)

    pneumonia = _pneumonia(df)
    contagem = pneumonia.astype(int).groupby(df[PACIENTE]).cumsum() - pneumonia.astype(int)
    ultimo_inicio = df[DIA].where(pneumonia).groupby(df[PACIENTE]).ffill()
    ultimo_inicio = ultimo_inicio.where(~pneumonia).groupby(df[PACIENTE]).ffill()
    novas = {
        'episodio_previo': contagem.gt(0).astype(int),
        'dias_desde_ultimo_episodio': df[DIA] - ultimo_inicio,
        'antibiotico_hoje': paciente['days_of_abx_until_today'].diff().gt(0).astype(int),
    }
    for c in COM_VARIACAO:
        novas[f'delta_{c}'] = paciente[c].diff()
        novas[f'media{JANELA}d_{c}'] = paciente[c].transform(lambda s: s.rolling(JANELA, min_periods=1).mean())
        novas[f'tendencia{JANELA}d_{c}'] = paciente[c].diff(JANELA - 1)
    for c in LINHA_DE_BASE_VENTILACAO:
        minimo_anterior = paciente[c].transform(lambda s: s.shift(1).rolling(2, min_periods=1).min())
        novas[f'aumento_{c}'] = df[c] - minimo_anterior
    df = pd.concat([df, pd.DataFrame(novas)], axis=1).copy()
    for c in CATEGORICAS:
        df[c] = df[c].astype(str)
    return df

def _colunas(*extras):
    return list(dict.fromkeys([*CHAVE, *FEATURES, *extras]))

def montar_base(df, horizonte=HORIZONTE_PADRAO):
    df = adicionar_features(df)
    df['Alvo'] = alvo_pav(df, horizonte)
    base = df[df['intub_flag'].eq(1) & ~em_episodio(df)]
    return base[_colunas('Alvo')].reset_index(drop=True)

def montar_entrada(df):
    df = adicionar_features(df)
    return df.loc[df['intub_flag'].eq(1), _colunas()].reset_index(drop=True)
