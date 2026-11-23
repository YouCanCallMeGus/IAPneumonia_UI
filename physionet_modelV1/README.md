# Chance de PAV a partir dos dados diários (CarpeDiem/PhysioNet)

Esta pasta contém um modelo que recebe os registros diários de um paciente intubado e estima a **chance de ele desenvolver pneumonia associada à ventilação mecânica (PAV) nos próximos 2 dias**.

**Resumo:** as chances que o modelo dá são confiáveis (bem calibradas). A capacidade de separar quem vai de quem não vai ter PAV é moderada (AUROC 0,69). Na prática, o modelo é bom para **apontar quem tem risco baixo** e razoável para ordenar a fila de vigilância. Não serve para confirmar PAV.


## Arquivos

| Arquivo | Função |
|---|---|
| `carpediem.py` | Carrega o CarpeDiem, monta as variáveis de cada dia e define o evento (alvo) |
| `modelo_risco_pav.py` | Treina, valida e salva o modelo |
| `prever.py` | Calcula a chance de PAV de um paciente a partir de um CSV |
| `arvore_streamlit.py` | Mostra em uma página a evolução da chance de um paciente, dia a dia, com a justificativa do `prever` |
| `exemplo_pacientes.csv` | 40 pacientes **fictícios** identificados e com rótulo de PAV, para testar o `avaliar` |
| `gerar_exemplo_pacientes.py` | Gera o `exemplo_pacientes.csv` (reprodutível; `--pacientes` e `--semente` mudam a amostra) |
| `importancia.py` | Mostra quais variáveis mais pesam no modelo no geral e se aumentam ou diminuem a chance |
| `avaliar.py` | Aplica o modelo salvo a uma base com rótulo, sem retreinar: métricas, faixas de chance e chance de cada paciente (validação externa) |
| `validacao_temporal.py` | Treina com os pacientes da v1 e testa nos adicionados depois |
| `resultados/` | Modelo salvo, métricas, odds ratios, previsões e saídas do `avaliar`. Fica fora do git porque contém IDs de pacientes |

## Como rodar

Na raiz do repositório, com o venv ativado:

```bash
python -m physionet_modelV1.modelo_risco_pav                         # 1. treina e salva (horizonte de 2 dias)
python -m physionet_modelV1.prever paciente.csv                      # 2. chance de PAV de um ou mais pacientes
python -m physionet_modelV1.validacao_temporal                       # 3. teste em pacientes que o modelo não viu
python -m physionet_modelV1.avaliar outra_base.csv                   # 4. teste numa base externa com rótulo
python -m physionet_modelV1.importancia                              # 5. variáveis que mais pesam no modelo, no geral

python -m physionet_modelV1.gerar_exemplo_pacientes                  # gera 40 pacientes fictícios com rótulo
python -m physionet_modelV1.avaliar physionet_modelV1/exemplo_pacientes.csv  # chance de cada um e AUROC
```

- `--horizonte N` no treino muda a janela de "próximos N dias".
- Por padrão o treino usa `data/CarpeDiem_v1.9.1/carpediem1.9.1_dataset.csv`; outro arquivo pode ser passado como argumento.
- `streamlit run physionet_modelV1/arvore_streamlit.py -- paciente.csv` abre a evolução da chance na página, com um seletor quando o CSV tem vários pacientes. Sem arquivo, usa o `exemplo_pacientes.csv`. As dependências do modelo e do app estão em `physionet_modelV1/requirements.txt` (`pip install -r physionet_modelV1/requirements.txt`).
- O `prever` usa o último modelo treinado. Se você treinar com `--horizonte 5`, a previsão passa a ser para 5 dias.
- Depois da lista de chances de cada paciente, o `prever` mostra a justificativa de cada dia: as 3 variáveis que mais mexeram na chance em relação ao dia intubado anterior. `--explicar 5` mostra 5, e `--explicar 0` desliga.

### Como a justificativa funciona

Partindo dos dados de ontem, o script troca um grupo de variáveis por vez pelo valor de hoje e recalcula a chance. A diferença é o efeito daquele grupo, em pontos percentuais (p.p.). Um grupo junta todas as colunas da mesma medida: `fio2_avg`, `fio2_max`, `delta_fio2_avg`, `media3d_fio2_avg`, `tendencia3d_fio2_avg` e `aumento_fio2_avg` formam o grupo `fio2`. As colunas de tempo (`hosp_day`, `icu_day`, dias de ventilação) formam um único grupo.

- Entre parênteses aparece o valor de ontem e o de hoje da primeira coluna do grupo. Um grupo pode ter efeito mesmo quando esse valor não mudou (ex.: `peep_avg 6 → 6`), porque o máximo do dia ou a média e a tendência de 3 dias mudaram.
- Os efeitos não somam exatamente a variação total do dia: o gradient boosting tem interações entre variáveis, e cada grupo é testado isoladamente.
- O efeito mostra o que o modelo aprendeu, não uma relação causal. Um sinal contraintuitivo (ex.: PEEP caindo e chance subindo) reflete padrões da base de treino.

### Como o `importancia` funciona

Ele usa os mesmos grupos de variáveis da justificativa do `prever` e calcula duas coisas para cada grupo. O resultado vai para `resultados/importancia.csv`.

- **`Queda_AUROC` (quanto o grupo pesa):** o modelo é retreinado nos mesmos 5 folds por paciente da validação. Em cada fold de teste, os valores do grupo são embaralhados entre os dias 5 vezes, e o script mede quanto o AUROC cai. Uma queda grande quer dizer que o modelo depende daquele grupo. Como o AUROC do modelo é 0,69, quedas de 0,01 a 0,03 já contam. Se `Queda_AUROC_dp` for parecido com a queda ou maior que ela, a importância é incerta.
- **`Efeito_pp` (para que lado o grupo empurra):** usando o modelo salvo, o script coloca a variável principal do grupo no valor baixo (P25) e depois no alto (P75) em todos os dias, e compara a chance média, em pontos percentuais. Para variáveis 0/1, a comparação é entre 0 e 1. Só a variável principal muda: o delta, a média e a tendência dela continuam iguais. Por isso o efeito real de um aumento sustentado tende a ser maior. Variáveis categóricas (ex.: `gender`) aparecem sem direção.
- Como na justificativa do `prever`, a tabela mostra associações que o modelo aprendeu, não causas. Por exemplo, `episodio_previo` e `antibiotico_hoje` diminuem a chance porque o paciente já em tratamento raramente tem um *novo* início de PAV nos 2 dias seguintes.

## 1. O que foi avaliado

**A pergunta:** *"Este paciente, intubado hoje e ainda sem pneumonia, vai começar um episódio de PAV nos próximos 2 dias?"*

**Dados:** CarpeDiem v1.9.1, do PhysioNet, com acesso credenciado. São 704 pacientes de UTI do Northwestern Memorial Hospital (EUA), um registro por dia de internação. Todos os episódios de pneumonia foram adjudicados por um painel de clínicos: 417 episódios de PAV.

**Base de treino:**
- **Quais dias entram:** dias em que o paciente está intubado e **não** está no meio de um episódio de pneumonia (do início até o fim da duração adjudicada; 7 dias quando a duração não é informada). Para quem já tem pneumonia, a pergunta não se aplica. São **6.690 dias de 590 pacientes**.
- **O que conta como evento:** um episódio de PAV que começa em um dos 2 dias seguintes. Acontece em **8,6%** dos dias.

**Variáveis** (só o que se sabe até o fim daquele dia):

| Grupo | Exemplos |
|---|---|
| Paciente | idade, sexo, IMC, tabagismo, COVID, índice de Charlson, DPOC, ICC, câncer, imunossupressão |
| Tempo | dia de internação, dia de UTI, dias de ventilação, dias desde a primeira intubação |
| Tratamento | dias de antibiótico até hoje, antibiótico hoje, dose de corticoide (dia e acumulada), tocilizumabe, remdesivir |
| Suporte | ECMO, hemodiálise, noradrenalina |
| Sinais vitais | temperatura, FC, PA, FR, SatO2 (média e pior valor do dia), RASS, Glasgow |
| Ventilação e gasometria | PEEP, FiO2, pressões de platô e de pico, volume-minuto, driving pressure, pH, PaO2, PaCO2, P/F |
| Laboratório | leucócitos, linfócitos, neutrófilos, plaquetas, creatinina, PCR, procalcitonina, lactato, SOFA... |
| Histórico | variação em relação a ontem, média e tendência de 3 dias, aumento de FiO2/PEEP sobre o mínimo dos 2 dias anteriores (lógica do VAE do CDC), pneumonia prévia e dias desde ela |

**O que ficou de fora de propósito**, para o modelo não "colar" do futuro:
- **Informação da alta:** desfecho, tempo total de internação/UTI/intubação, traqueostomia, categoria adjudicada do paciente.
- **Resultado da investigação:** lavado broncoalveolar (BAL), patógenos, cura e impressão clínica.
- **`nat_score`:** é calculado a partir do resultado da cultura, então vazaria o diagnóstico.

**Modelos testados** (validação cruzada em 5 partes, agrupada por paciente; o modelo é sempre avaliado em pacientes que nunca viu):

| Versão | AUROC | AUPRC | Brier |
|---|---|---|---|
| v1 (585 pacientes), regressão logística (versão anterior) | 0,652 | 0,148 (ref. 0,092) | 0,082 (ref. 0,084) |
| v1.9.1, logística com as mesmas variáveis da v1 | 0,665 | 0,142 (ref. 0,086) | 0,077 (ref. 0,079) |
| v1.9.1, logística com todas as variáveis | 0,672 | 0,149 | 0,076 |
| v1.9.1, gradient boosting | 0,686 | 0,155 | 0,077 |
| **v1.9.1, média logística + boosting (modelo final)** | **0,692** | **0,158** | **0,076** |

Valores do v1.9.1 são a média de 3 sementes. A variação entre sementes ficou abaixo de 0,01 de AUROC.

**O que melhorou o resultado:**
- **Mais dados:** a v1.9.1 tem 20% mais pacientes e 26% mais episódios de PAV.
- **Variáveis novas:** antibiótico, corticoide, comorbidades e pior valor do dia.
- **Combinação de modelos:** com mais dados, o boosting passou a captar relações não lineares; a média com a logística mantém a calibração.

**O que não ajudou:** regularização diferente, splines no tempo de ventilação e um boosting mais lento.

## 2. Qual é a entrada

Um CSV com **uma linha por dia de internação**, usando os nomes de coluna do CarpeDiem v1.9.1. Veja `exemplo_pacientes.csv`, que traz as 98 colunas de entrada para 40 pacientes fictícios.

- **Obrigatórias:** `hosp_day` (dia de internação: 1, 2, 3...) e `intub_flag` (1 = intubado naquele dia).
- **As demais:** são opcionais, mas quanto mais vierem, melhor. Coluna ausente é preenchida com a mediana dos pacientes de treino, ou seja, o modelo assume um valor "típico". A lista completa está em `COLUNAS_ENTRADA`, no `carpediem.py`.
- **Vários dias:** mande o histórico desde a admissão, não só o dia de hoje. As tendências dependem dos dias anteriores.
- **Unidades do CarpeDiem:** temperatura em **°F** (38 °C = 100,4 °F), FiO2 em **%** (40, não 0,4), leucócitos em **mil/µL** (12,5, não 12.500).
- **Acumulados:** `days_on_ventilator` e `days_of_abx_until_today` contam até aquele dia.
- **Identificação:** `anonymized_id` é opcional. Sem ela, todas as linhas são tratadas como um paciente só. Para vários pacientes no mesmo arquivo, inclua a coluna; veja `exemplo_pacientes.csv`.
- **Formato antigo:** arquivos da v1 e da v1.8 do CarpeDiem também são aceitos e convertidos automaticamente (seção 5).

Na parte logística do modelo, as variáveis que mais pesam são: pneumonia prévia, antibiótico, corticoide, ECMO, tempo de internação e de ventilação, e as tendências de SOFA, P/F e PEEP. Vale priorizar essas colunas na entrada.

## 3. Como interpretar a saída

```
Paciente: chance de PAV nos próximos 2 dias
  dia   5 de internação:  12.3%
  dia   6 de internação:  21.2%
  dia   7 de internação:  17.6%
  -> hoje (dia 7): 17.6%, 2.1x a média dos pacientes intubados (8.6%)
```

- **17,6%:** entre pacientes parecidos com este, cerca de 18 em 100 começam uma PAV nos próximos 2 dias. É uma chance real, já calibrada.
- **2,1x a média:** essa é a leitura mais útil. Como a PAV é rara, o número absoluto quase sempre parece baixo; compare sempre com a média de 8,6%.
- **Evolução:** a chance subindo ao longo dos dias (no exemplo, de 8–12% nos dias estáveis para 21% no dia 6, junto com febre, leucocitose, queda da P/F e aumento de FiO2/PEEP depois que o antibiótico foi suspenso) é o sinal para ficar de olho.

**Guia de leitura** (horizonte de 2 dias):

| Chance | Leitura |
|---|---|
| < 5% | Risco baixo. Cerca de 3 em 100 têm PAV, um terço da média |
| 5–10% | Risco na média |
| 10–20% | Risco elevado, 1,5 a 2 vezes a média |
| ≥ 20% | Risco mais alto que o modelo consegue indicar (cerca de 20% de PAV de fato) |

## 4. A realidade das previsões

**O que acontece em cada faixa de chance** (previsões em pacientes não vistos, horizonte de 2 dias, média de 8,6%):

| Chance prevista | % dos dias | PAV de fato |
|---|---|---|
| < 5% | 31% | **2,8%** |
| 5–10% | 38% | 7,7% |
| 10–15% | 17% | 13,1% |
| 15–20% | 8,5% | 16,6% |
| 20–30% | 4,8% | 20,6% |
| ≥ 30% | 0,8% | 19,6% |

**Na prática:**
- **Calibração:** até cerca de 30%, a chance prevista bate com o que acontece de fato. Acima de 30% há poucos dias (0,8%), e o modelo exagera; trate tudo acima de 20% como "risco alto".
- **Onde funciona: descartar.** Quase um terço dos dias fica abaixo de 5% com só 2,8% de PAV. Os 20% de dias com menor chance têm PAV em 1,7%, cinco vezes menos que a média.
- **Onde é razoável: priorizar.** Vigiando os 31% de dias com chance ≥ 10%, captura-se 56% das PAV. Os 20% de dias mais arriscados concentram 39% delas.
- **Onde não funciona: confirmar.** Mesmo no topo, a PAV de fato fica perto de 20%. A cada 5 pacientes de "alto risco", 4 não terão PAV nos 2 dias seguintes.

**Com horizonte de 5 dias** (`--horizonte 5`): AUROC 0,688, AUPRC 0,301 (ref. 0,186), Brier 0,143 (ref. 0,151). As chances ficam cerca de 2 vezes maiores porque a janela é maior, não porque o modelo ficou melhor.

**Odds ratios:** estão em `resultados/odds_ratios.csv` (metade logística do modelo, por 1 desvio-padrão). Todos os pesos, como foram calculados e como usá-los num índice estão em [PESOS.md](PESOS.md). Os efeitos são pequenos (0,84 a 1,12) e **não são causais**. Por exemplo:
- **"Antibiótico hoje" e "pneumonia prévia" com OR < 1:** pacientes sob antibiótico ou recém-tratados têm menos PAV nova nos 2 dias seguintes. Isso é proteção do tratamento ou seleção, não um efeito a explorar.
- **Mais tempo de ventilação com OR < 1:** reflete quem sobrevive a uma longa ventilação sem PAV (sobrevivência), não proteção real.

## 5. Testando em outras bases

### Avaliando nas versões antigas (v1, v1.8, v1.9)

O `avaliar.py` lê todas as versões. A v1 e a v1.8 usam outros nomes de coluna e são convertidas automaticamente para o formato da v1.9.1 (`padronizar`, em `carpediem.py`):
- **Nomes e categorias:** as colunas são renomeadas e os valores de tabagismo, COVID e categoria de episódio são padronizados.
- **Volume-minuto:** convertido de mL/min para L/min.
- **Dias de ventilação:** reconstruídos a partir de `intub_flag`.
- **Dia de internação na v1:** como a v1 não tem dia de internação, os dias de UTI de cada paciente são numerados em sequência.

```bash
python -m physionet_modelV1.avaliar data/CarpeDiem_v1/CarpeDiem_dataset.csv --saida physionet_modelV1/resultados/avaliacao_v1
python -m physionet_modelV1.avaliar data/CarpeDiem_v1.8/carpediem_v1.8_dataset.csv --saida physionet_modelV1/resultados/avaliacao_v1.8
```

**Estas avaliações não são validação.** Os pacientes dessas versões estão no treino, e o `avaliar.py` avisa isso. Os números servem para conferir que a conversão funciona, não para medir o modelo:

| Base | Pacientes no treino | AUROC | Chance média prevista x PAV observada |
|---|---|---|---|
| v1.9 | 590 de 590 | 0,94 | 8,6% x 8,6% |
| v1.8 | 567 de 575 | 0,80 | 12,1% x 8,9% |
| v1 | 480 de 493 | 0,77 | 11,7% x 9,4% |
| *Validação cruzada (referência honesta)* | — | *0,69* | *8,6% x 8,6%* |

**Como ler esses números:**
- **AUROC alto:** reflete o modelo reconhecendo pacientes que viu no treino.
- **AUROC menor nas versões antigas que na v1.9:** faltam variáveis que só a v1.9.1 tem (pior valor do dia, antibiótico, comorbidades), e o modelo as preenche com a mediana.
- **Chances superestimadas na v1 e na v1.8:** vêm dessas mesmas variáveis faltantes. É um aviso útil para a validação externa: **uma base com menos colunas que o treino perde desempenho e calibração.**

### Validação temporal (`validacao_temporal.py`)

O teste mais próximo de uma base nova disponível hoje. A v1.9.1 tem 120 pacientes que não estavam na v1, incluídos em versões posteriores do dataset. O script:
1. treina o modelo só com os 584 pacientes da v1;
2. aplica esse modelo, sem retreinar, aos 120 novos (101 deles têm dias intubados avaliáveis);
3. calcula os intervalos de confiança por bootstrap de **pacientes**, porque os dias de um mesmo paciente não são independentes.

| Métrica | Pacientes novos (IC 95%) | Validação cruzada no treino | Referência |
|---|---|---|---|
| AUROC | **0,708** (0,639–0,771) | 0,700 | 0,500 |
| AUPRC | 0,159 (0,120–0,229) | 0,166 | 0,086 |
| Brier | 0,076 (0,058–0,098) | 0,076 | 0,078 |

| Chance prevista | % dos dias | Chance média | PAV de fato |
|---|---|---|---|
| < 5% | 46,7% | 3,0% | 3,4% |
| 5–10% | 36,1% | 6,9% | 10,3% |
| 10–15% | 11,0% | 12,2% | 18,1% |
| 15–20% | 4,8% | 17,3% | 24,7% |
| ≥ 20% | 1,4% | 24,2% | 9,1% (2 de 22 dias) |

**Conclusões:**
- **A ordenação se sustenta.** Em pacientes que o modelo nunca viu, o AUROC (0,71) é igual ao da validação cruzada. Não há sinal de sobreajuste.
- **Descartar continua funcionando:** quase metade dos dias fica abaixo de 5%, com PAV real de 3,4%.
- **A calibração mudou.** Nos novos pacientes, o modelo **subestima** o risco: prevê 6,4% em média, contra 8,6% observados. Entre 5% e 20%, a PAV real foi cerca de 1,4 vez a prevista. A explicação mais provável é a mudança de população: os novos têm bem menos COVID (22% dos dias, contra 40% no treino).
- **Ressalvas:** a amostra é pequena, e o IC do AUROC vai de 0,64 a 0,77. As faixas acima de 20% têm poucos dias e não permitem conclusão. O dataset não traz datas, então "novos" significa "incluídos em versões posteriores", o que provavelmente corresponde a internações mais recentes, mas sem confirmação.

**Lição para o A.C.Camargo:** é esperado que a ordenação se mantenha melhor que a calibração. Em outro hospital, com outra prevalência de PAV e outro perfil de pacientes, as chances absolutas provavelmente precisarão de **recalibração** com dados locais. A ordem de risco tende a valer mais que o número exato.

### Validação externa (`avaliar.py`)

Para testar o modelo em uma base nova, por exemplo do A.C.Camargo:

1. Monte um CSV com uma linha por dia de internação, nos nomes de coluna da v1.9.1 e nas unidades do CarpeDiem (seção 2).
2. Inclua o rótulo: `episode_category` (`VAP`, `HAP` ou `CAP` no dia de início do episódio) e, se houver, `episode_duration`.
3. Rode `python -m physionet_modelV1.avaliar base.csv`. O script usa o modelo salvo em `resultados/modelo_pav.joblib`; `--modelo` escolhe outro.

Só `hosp_day` e `intub_flag` são obrigatórias:
- **Sem `anonymized_id`:** o arquivo é tratado como um paciente só.
- **Sem `episode_category`:** não dá para medir acerto, e o script mostra apenas as chances previstas, como o `prever`.
- **Só dias sem PAV, ou só com PAV:** o script calcula só o Brier. AUROC e AUPRC precisam dos dois desfechos.
- **Poucos pacientes:** as métricas ficam instáveis (com menos de 30, trate como conferência). Com um único paciente, não há intervalo de confiança.

A saída mostra:
1. **Sobreposição com o treino:** um aviso, se algum paciente da base estava no treino.
2. **Resumo:** dias avaliados, pacientes e dias com PAV na janela.
3. **Métricas:** AUROC, AUPRC e Brier com IC 95%, ao lado dos valores do treino.
4. **Tabela por faixa de chance.**
5. **Tabela por paciente**, ordenada pela maior chance:
   - **Dias:** dias avaliados (intubado e fora de episódio de pneumonia).
   - **Chance_media, Chance_maxima e Chance_ultimo_dia:** a chance média, a maior e a do último dia avaliado.
   - **Teve_PAV:** se a PAV começou dentro da janela de previsão.
   - **Dia_da_maxima:** o dia em que a chance foi maior.
6. **AUROC por paciente:** responde à pergunta de vigilância: a maior chance de cada paciente separa quem teve PAV de quem não teve?

Os arquivos ficam em `resultados/avaliacao/` (ou na pasta indicada em `--saida`): `metricas.csv`, `faixas.csv`, `limiares.csv` (sensibilidade, especificidade, VPP e VPN por corte de chance), `previsoes.csv` (chance por dia) e `por_paciente.csv`.

Para ler o resultado:
- **AUROC próximo de 0,69:** o modelo ordena bem na base nova.
- **Chance prevista diferente da PAV de fato:** se a coluna `Chance_media` da tabela por faixa se afastar de `PAV_de_fato`, é preciso recalibrar.
- **IC largo:** faltam pacientes para concluir.

### Exemplo com pacientes fictícios

O `exemplo_pacientes.csv` traz 40 pacientes inventados (`EX001` a `EX040`), cada um com 8 a 16 dias, as 98 colunas de entrada e o rótulo:
- **Com PAV (15 pacientes):** a piora nos dias anteriores tem intensidade variável, e algumas PAV não dão sinal nenhum.
- **Falsos alarmes (cerca de 15%):** pacientes que pioram sem ter PAV.
- **Os demais:** ficam estáveis.

```bash
python -m physionet_modelV1.gerar_exemplo_pacientes   # --pacientes N e --semente S mudam a amostra
python -m physionet_modelV1.avaliar physionet_modelV1/exemplo_pacientes.csv --saida physionet_modelV1/resultados/exemplo_pacientes
```

```
Métricas:
       Valor  IC95_inferior  IC95_superior
AUROC  0.778          0.682          0.861
AUPRC  0.237          0.128          0.457
Brier  0.071          0.044          0.104

Por paciente (ordenado pela maior chance):
               Dias Chance_media Chance_maxima Chance_ultimo_dia Teve_PAV  Dia_da_maxima
EX035             9        14.4%         22.1%             22.1%      não             11
EX030             8        11.2%         17.6%             13.4%      sim              8
EX025             6        12.5%         17.3%             17.3%      não              8
EX003            10        13.6%         16.0%             15.5%      sim              6
...
EX001            15         6.3%          7.9%              6.3%      não             14
EX024             6         6.3%          7.7%              5.8%      não              7

AUROC por paciente (chance máxima x teve PAV): 0.699
```

**Como ler:**
- **No topo:** dos 13 pacientes de maior chance, 9 tiveram PAV.
- **No fim:** dos 13 de menor chance, só 3 tiveram.
- **Erros:** os falsos alarmes (EX035, EX025) e as PAV silenciosas (EX002, EX034) aparecem como apareceriam na prática.

**Esses números só mostram que o pipeline funciona.** Os pacientes foram gerados com uma piora "de livro" antes da PAV, e por isso o AUROC (0,78) é maior que o real. O desempenho de verdade é o da validação cruzada (0,69) e o da validação temporal (0,71).

## 6. Limitações

- **Os sinais aparecem tarde:** febre, piora de oxigenação e leucocitose surgem junto com o início da PAV, não antes. Nos dias anteriores, os dados diários mudam pouco, e isso limita o AUROC por volta de 0,7.
- **Faltam variáveis importantes:** aspecto da secreção traqueal, cultura de vigilância, raio-X e ausculta não estão no CarpeDiem.
- **Os dados são diários:** médias e piores valores por dia apagam pioras que aparecem em poucas horas.
- **O evento depende de investigação:** só existe "episódio de PAV" quando houve BAL. PAV não investigada aparece como "sem PAV".
- **A população é peculiar:** 36% dos dias de internação são de pacientes com COVID, todos de um único hospital americano, com protocolo de BAL frequente. **Antes de usar no A.C.Camargo, o modelo precisa ser validado com dados do hospital.**
- **Nomes de coluna:** o modelo usa os do CarpeDiem. Para pacientes do Tasy, seria preciso mapear as colunas e converter as unidades.

**Próximos passos com maior potencial:**
1. **Mapear o Tasy para estas colunas** e validar o modelo nos pacientes do A.C.Camargo.
2. **Acrescentar o que o hospital tem e o CarpeDiem não:** laudos de imagem, secreção e cultura.
3. **Testar dados com resolução maior que um dia,** se o Tasy exportar sinais vitais por hora.

## Privacidade

O CarpeDiem é de acesso credenciado no PhysioNet, e a licença **proíbe enviar os dados a serviços de LLM** (ChatGPT, Claude etc.). Este modelo foi desenvolvido olhando só o esquema das colunas e métricas agregadas, sem nenhuma linha de paciente.

> **Atenção:** em `data/CarpeDiem_v1.9.1/`, o arquivo `carpediem_v1.9.1_data_dictionary.csv` (20 MB) **não é um dicionário: contém os dados.** Trate-o com o mesmo cuidado que o dataset.
