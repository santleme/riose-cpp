# Modelos comportamentais individuais para o Campo Vivo

**Pesquisa focada:** comportamento, decisão e movimento de bovinos individuais, incluindo pastejo, água, termorregulação, memória e interação social. Fontes consultadas em outubro de 2026. O relatório diferencia resultados medidos em animais, equações ajustadas em populações específicas, modelos mecanísticos avaliados e propostas de engenharia ainda não validadas. Ele complementa [a pesquisa geral de arquitetura e dados](landscape.md).

## Síntese

Não há evidência para tratar uma rede neural única como “cérebro bovino”. O que as fontes sustentam é uma arquitetura híbrida: (1) balanços de recursos e estados biológicos explícitos; (2) atividades discretas persistentes, com distribuições de duração calibradas por dados; (3) decisão individual sob necessidades e custos; (4) navegação espacial em pastagem heterogênea, com memória e vínculos sociais; e (5) observação e validação por saídas mensuráveis. Cada camada resolve uma pergunta diferente. Um classificador de acelerômetro reconhece atividade observada; ele não ensina, por si só, o que uma vaca fará após mover um bebedouro.

Para o jogo, recomendo implementar agora um **ABM híbrido e determinístico**, com políticas estocásticas cuja aleatoriedade vem de uma semente por animal. A camada de seleção pode começar em regras/utilidade parametrizadas. Um HMM (ou semi-Markov) fornece transições e duração de atividades; um modelo de custo-benefício escolhe a meta; caminhada correlacionada e memória guiam o trajeto. Um modelo fisiológico/ambiental separado atualiza fome, hidratação, carga térmica e condição. Esses valores influenciam decisões e também mudam em resposta a ações. Isso gera individualidade observável sem atribuir conhecimento científico a parâmetros inventados.

Aprendizado por reforço é uma etapa posterior, não uma fonte de verdade biológica. Treinar uma política em um simulador arbitrário pode ensinar uma política que explora os erros do próprio simulador. Primeiro se calibram os alvos observáveis e as consequências mecanísticas; depois, RL pode otimizar escolhas num ambiente validado e ser comparado com a política interpretável, em dados separados por animal, fazenda e período.

## O que cada família de modelo consegue afirmar

| Família | Evidência que pode fornecer | O que não demonstra |
|---|---|---|
| Classificador supervisionado de atividade (HMM, LDA, CNN) | Probabilidade de estar pastando, ruminando, descansando ou caminhando a partir de sinais de sensores, no domínio e protocolo anotados. | Motivação, estado interno, causalidade de manejo ou política de navegação. Rótulos de sensor não são uma política de ação. |
| Série temporal/ABM de movimento | Distribuição de posições, trajetórias, velocidade, formação de grupo ou uso de recursos, se calibrado em trajetórias independentes. | Que a trajetória emergirá correta só porque o modelo parece natural. GPS em rebanho observa posição, não intenção. |
| Modelo de necessidades e fisiologia | Respostas de ingestão, água, peso ou temperatura no domínio de calibração e validação. | Validade universal fora de raça, sistema produtivo, clima, estágio de produção e resolução temporal usados. |
| Memória/forrageamento | Hipóteses testáveis sobre busca e retorno a patches, sobretudo em pastagens heterogêneas. | Memória diretamente observada. Experiência interna é inferida; movimentos semelhantes podem vir de percepção ou rotina. |
| RL / multiagente aprendido | Política que maximiza uma recompensa definida em ambientes de treinamento; pode capturar compromissos complexos. | Correspondência com comportamento real sem recompensa, ambiente e comparação empírica adequados. O retorno alto não valida a biologia. |

## Evidência por processo

### Atividade, rotina diária e duração de estados

Em vacas leiteiras a pasto, a atividade não alterna como ruído independente a cada quadro. Registros longitudinais mostram distribuição horária e sazonal de pastar, ruminar e ficar inativa; no conjunto de 54 vacas da Nova Zelândia, pastoreio foi mais concentrado nas horas de luz e ruminação predominou à noite, com variações entre estação e ordem de lactação. O estudo é forte como descrição de padrão no sistema estudado, mas não é uma regra universal para todo bovino, raça ou manejo [Jochims et al. (2023), *Journal of Animal Science*](https://doi.org/10.1093/jas/skad038). Outro estudo de vacas a pasto relatou que >95% do tempo observado estava em quatro classes amplas (pastando, andando, ruminando/descansando em pé e deitada); a composição depende do sistema de observação e das definições das classes [Kilgour (2012), *Applied Animal Behaviour Science*](https://doi.org/10.1016/j.applanim.2011.12.006).

Modelos de sensores mostram que é viável inferir classes, mas não são intercambiáveis. Em coleiras com acelerômetro, GPS e magnetômetro, o classificador baseado em HMM alcançou F-score 0,98 para pastejo, 0,87 ruminação, 0,85 repouso e 0,73 caminhada; o desempenho baixo de caminhada importa para movimento e gasto de atividade [Guo et al. (2009), *Ecological Modelling*](https://doi.org/10.1016/j.ecolmodel.2009.04.047). Classificação por coleira depende da posição do sensor, intervalo de amostragem, definição de “pastejo” e validação individual. Um método separado, treinado em duas experiências de campo com acelerômetro e GPS, classifica quatro comportamentos observados, mas os autores também destacam a falta de conjuntos públicos de benchmark [López et al. (2022), *Entropy*](https://doi.org/10.3390/e24030336). CNNs foram aplicadas a sinais de pescoço para classificar ruminação, alimentação e “outro”, mas esse “outro” não separa todos os comportamentos que o jogo precisa [Barwick et al. (2021), *Sensors*](https://doi.org/10.3390/s21124050).

**Implicação implementável:** ter estados `grazing`, `ruminating`, `standing`, `lying`, `walking`, `drinking`, `social` e `thermoregulating`, mas não amostrar nova ação a cada frame. Use transições em escala de minutos simulados e distribuições de duração por estado, moduladas por hora, estação, sexo/estágio produtivo, condição e indivíduo. Calibre primeiro as proporções e horários agregados. Trate a divisão detalhada de classes e as durações individuais como hipóteses até validação. HMMs estimam a sequência de estados latentes que explica observações; semi-Markov é uma extensão de engenharia útil quando a duração de cada atividade precisa ser explícita. Os dados de sensores identificam emissão/transição observada; motivação continua sendo outro módulo.

### Pastejo, seleção e ingestão

Uma identidade útil para ingestão em pastagem é:

```text
DMI_pasto (kg MS/dia) = tempo de pastejo (min/d)
                     × taxa de bocados (bocados/min)
                     × massa por bocado (kg MS/bocado)
```

Essa decomposição é útil porque conecta o agente ao mapa: a taxa e a massa do bocado respondem à altura/densidade/estrutura do pasto; o animal também paga custos de busca, apreensão, mastigação, ruminação e deslocamento. Não é, sozinha, uma fórmula universal calibrada: estudos de ingestão por sensores ainda pedem validação em pastagens diversas e medição de forragem concomitante [González et al. (2017), *Crop & Pasture Science*](https://doi.org/10.1071/CP16383); uma revisão de mensuração enfatiza que seletividade e composição do pasto mudam no tempo, limitando a transferência de estimações curtas [Boval & Sauvant (2019), *Animal*](https://doi.org/10.1017/S1751731119001330).

Há uma base de evidência para relacionar altura da forragem, oferta, bocado e consumo, mas os modelos individuais perdem precisão em sistemas e animais fora de calibração. Na avaliação on-farm de modelos para vacas leiteiras em gramados seminaturais, uma versão modificada do Mertens II teve erro percentual relativo de 13,4% para médias por fazenda-período; para animais individuais recebendo suplementação alta, o erro foi 24,7%, acima do critério de adequação dos autores [Léger et al. (2023), *Animal*](https://doi.org/10.1016/j.animal.2023.100822). Um modelo mecanístico para gado de corte explicita a contribuição de rumen fill, digestão e passagem para predizer DMI diária; não prevê movimento nem cada bocado [Hyer, Oltjen & Galyean (1991), *Journal of Animal Science*](https://doi.org/10.2527/1991.692827x).

ABMs já demonstraram como representar escolha individual, recurso espacial e efeitos no ecossistema. Um modelo espacial de pastejo de resíduos de milho usou estado interno (fome/hidratação), distribuição de forragem, água, ciclo dia-noite e manejo, e foi ancorado em ensaio de pastejo de três anos [Dillard et al. (2016), *Agricultural Systems*](https://doi.org/10.1016/j.agsy.2016.06.007). Um ABM de bovinos expostos a uma planta tóxica combina tendência de aproximação ao rebanho e seleção de patches com forragem; é útil como padrão de implementação, mas suas regras de ação foram construídas para pergunta e ecologia específicas, não são uma política bovina geral [Bailey et al. (2018), *Ecological Modelling*](https://doi.org/10.1016/j.ecolmodel.2017.11.018). Um ABM de steers no shortgrass steppe reproduziu seleção por patches e declive e incorporou memória de patches de maior qualidade; é dissertação/modelo de caso, evidência mais fraca que validação multi-fazenda [Clark (2018), Colorado State University](https://mountainscholar.org/items/4bfd89c8-7da5-482e-b241-0d46421a50c6).

**Implicação implementável:** cada célula fornece matéria seca disponível, altura, qualidade e distância; a mordida remove biomassa limitada pelo recurso e pela capacidade do animal. Atualize massa, energia e saciedade/enchimento com unidades declaradas. Distância e baixa qualidade elevam custo e podem disparar busca/alternância de patch. Os coeficientes de mordida devem manter faixa por espécie forrageira e ser marcados como calibrados, transferidos ou ilustrativos. Não converter simplesmente minutos pastando em DMI sem altura/estrutura e checagem de oferta.

### Água, sede e competição no bebedouro

Água diária depende de peso, matéria seca ingerida e ambiente; medi-la em oportunidades é questão diferente de estimar demanda diária. Em 41 vacas leiteiras monitoradas individualmente no inverno, consumo médio livre foi 83,6 ± 17,1 L/d em 7,3 ± 2,8 episódios/d; a publicação aponta efeitos de clima, dieta, produção e DMI e alerta que padrões sazonais poderiam diferir no verão [Cardot et al. (2008), *Journal of Dairy Science*](https://doi.org/10.3168/jds.2007-0208). Em gado de corte crescente, água variou com temperatura, radiação, umidade, vento e DMI, mas equações apresentaram R² moderados e dispersão individual importante [Mader & Davis (2018), *Journal of Animal Science*](https://doi.org/10.1093/jas/sky267).

Um estudo de bovinos Nelore em condições tropicais propôs, com validação leave-one-out dentro do conjunto estudado, a equação:

```text
WI (L/d) = 9,449 + 0,190 × MBW + 0,271 × Tmáx − 0,259 × UR + 0,489 × DMI
```

onde `MBW = peso vivo^0,75` (kg^0,75), `Tmáx` é temperatura máxima diária em °C, `UR` é umidade relativa em porcentagem e `DMI` é kg de matéria seca/d. A equação foi desenvolvida para Bos indicus em ambiente tropical; não deve ser transplantada diretamente para leiteiras, taurinos, bezerros ou outro regime sem recalibração [Azevedo et al. (2019), *Journal of Animal Science*](https://doi.org/10.1093/jas/skz003). Em estudo de vacas após ordenha, características individuais e sucesso competitivo foram associados a diferenças de comportamento no bebedouro, embora o consumo de água não tenha sido medido nesse estudo; isso apoia modelar oportunidade e competição à parte da demanda fisiológica [Foris et al. (2025), *Animals*](https://doi.org/10.3390/ani15040534).

**Implicação implementável:** use a predição diária somente como alvo de demanda de escala diária, com domínio explicitado. Internamente, acumule déficit/desejo de beber e gere eventos de visita com variabilidade individual. A visita requer caminho navegável, água disponível e tempo/fluxo de ingestão; interação/ocupação do ponto pode atrasar ou interromper o acesso. Isso é uma composição de demanda validada em contexto e regras de eventos ainda a calibrar — não existe fórmula acima que determine o minuto em que um animal vai beber.

### Calor e resposta térmica

THI é um resumo de ambiente, não temperatura corporal. Uma formulação comum para gado leiteiro é:

```text
THI = 0,8 × T + (UR/100) × (T − 14,4) + 46,4
```

com temperatura de bulbo seco `T` em °C. O índice usa temperatura e umidade; omite vento, radiação solar, sombra, pelagem, aclimatação, metabolismo e características do animal. Estudos de resposta em vacas leiteiras encontram ganho ao incluir vento, produção e hora/dia em modelos de frequência respiratória e temperatura retal, em vez de usar apenas um corte único de THI [Li et al. (2020), *Journal of Dairy Science*](https://doi.org/10.3168/jds.2019-17583). Um modelo termorregulatório mais mecanístico usa estado dinâmico de calor corporal e saídas como temperatura central, pele e pelagem; foi desenvolvido e avaliado com dados experimentais de vacas leiteiras e permite resposta transitória a temperatura, umidade e velocidade do ar [Liu et al. (2022), *Computers and Electronics in Agriculture*](https://doi.org/10.1016/j.compag.2022.107170).

**Implicação implementável:** para versão inicial, THI pode modular probabilidade de busca de sombra/água, reduzir atividade e ajustar ingestão de forma gradual, com coeficientes e limiares por população. Esses efeitos qualitativos são suportados; uma função individual universal não é. Para fidelidade futura, modelar armazenamento de calor/temperatura central com passo de minutos, incidência solar, vento, sombra, tamanho corporal, estado produtivo e adaptação. A curva térmica terá de ser validada por temperatura/respiração de fonte apropriada antes de ser apresentada como previsão fisiológica. THI apenas como ícone ou corte duro produz saltos que não representam dinâmica térmica.

### Movimento, relações sociais e memória espacial

Um modelo de deslocamento individual a partir de GPS de alta frequência, magnetômetro e acelerômetro combinou HMM de atividade e previsão de trajetória de maior prazo; a acurácia do classificador variou consideravelmente por classe (F-score entre 0,73 e 0,98) [Guo et al., DOI acima](https://doi.org/10.1016/j.ecolmodel.2009.04.047). O resultado demonstra possibilidade de classificar/estimar movimento em animais cercados e dados daquele estudo, não que o caminho seja universalmente previsível. Sensores GPS comuns podem ter erro de vários metros e frequência baixa demais para inferir interação próxima; redes de proximidade exigem escala espacial e temporal definidas [Doyle et al. (2021), *Scientific Reports*](https://doi.org/10.1038/s41598-021-95032-0).

Estudo de bovinos de corte em pastagem relacionou organização social, atividade e condição física; no grupo grande estudado, resultados foram mais consistentes com auto-organização do rebanho do que com hierarquia de dominância explicando toda a distribuição espacial [Gabrieli & Malkinson (2022), *Applied Animal Behaviour Science*](https://doi.org/10.1016/j.applanim.2022.105723). Vínculos/afiliações e traços individuais também influenciam quais animais iniciam movimentos coletivos, e podem importar mais que posição de dominância em certas condições [Sueur et al. (2009), *Animal*](https://doi.org/10.1017/S1751731109004790). Isso recomenda relações diádicas estáveis, distância de conforto, alinhamento local e influência de vizinhos como opções a testar; evitar atribuir ao rebanho um “líder” fixo sem evidência do caso representado.

Memória espacial tem suporte experimental direto, ainda que difícil de inferir de trajetórias livres: bovinos retornaram por períodos extensos a locais previamente associados a alimento preferido em um experimento controlado de recipientes [experimento com steers, DPI eResearch Archive](https://era.dpi.qld.gov.au/id/eprint/8369/). Em modelo contínuo geral de forrageamento, duas trilhas de memória — uma longa e atrativa para qualidade aprendida e uma curta e repulsiva para área recentemente explorada — são combinadas como `M = L − ψS`; esta é uma proposta de modelagem, não equação fisiológica bovina. Simulações mostraram que a vantagem de memória emerge principalmente em paisagens heterogêneas; o próprio artigo destaca ausência de bioenergética, rotina diária, percepção e necessidade de calibrar em dados de campo [Bracis et al. (2015), *PLOS ONE*](https://doi.org/10.1371/journal.pone.0136057).

**Implicação implementável:** deslocamento: velocidade/ângulo correlacionados + alvos locais de água, sombra e alimento + custo de rota/declive/cerca. Coesão: atração moderada aos vizinhos associados, alinhamento local e distância mínima; não simular todos os agentes com uma regra rígida de “siga o líder”. Memória por animal: armazenar qualidade/visitas de patches em uma grade esparsa com decaimento. A forma de dois tempos é opção plausível e barata; taxa de aprendizagem, esquecimento, alcance sensorial e peso social são parâmetros de jogo até confronto com GPS/experimento. Memória não deve permitir localização perfeita do alimento sem pista sensorial.

## Arquitetura recomendada para implementar e treinar

```text
Ambiente por célula: forragem, qualidade, água, sombra, solo, barreiras, clima
                         │ percepções locais e memória
                         ▼
Bovino individual
  Estado lento: peso, reservas, condição, saúde, adaptação térmica
  Necessidades: energia, água, descanso, termorregulação
  Estado rápido: atividade, duração, velocidade, direção, destino
  Relações: arestas sociais e vizinhos observados
                         │
                         ▼
  Proposta de ação → utilidade/custo → amostragem com seed individual
                         │
                         ├─ mover/seguir/evitar
                         ├─ pastar/beber/ruminar/deitar/buscar sombra
                         └─ interação social
                         │
        consumo e movimento alteram pasto, solo, água e estado biológico
```

**Separação de escalas recomendada (decisão de engenharia):** renderer/animação em tempo real; locomoção em passo fixo curto; decisões em intervalos de dezenas de segundos/minutos simulados; eventos de beber e alimentar em minutos; massa corporal, demanda e recuperação em horas/dias; crescimento vegetal em horas/dias. A separação permite que quadro visual não altere comportamento nem produza diferenças de resultado em máquinas com FPS distintos.

**Seleção de ação como baseline transparente (proposta a calibrar):** para cada opção `a` (comer no patch, beber, procurar sombra, deitar/ruminar, manter coesão), calcular uma utilidade configurável `U_i(a) = benefício de necessidade + benefício local + preferência individual + influência social − custo de deslocamento − risco/incômodo`. Uma softmax com temperatura individual escolhe estocasticamente entre opções permitidas. Não afirmar que softmax reproduz a psicologia do animal: é uma escolha de engenharia interpretável para gerar variação e calibrar frequências. Ações devem ser restringidas por disponibilidade espacial, estado físico e duração mínima. Atualizar necessidades por balanço, não por decrementos cosméticos.

**Plano de treinamento:**

1. **Baseline de medição:** treinar classificador de atividade por segmentos de sensor e rotular unidade, população e protocolo. Usar divisão por animal/grupo/fazenda/período, e reportar F1 por classe, matriz de confusão, calibração de probabilidades e erro de duração/bouts. Não separar janelas aleatoriamente: janelas próximas do mesmo animal vazariam rotina para teste.
2. **Ajuste de sequência:** usar HMM para transições/probabilidades de observação; comparar com CNN de série e classificador simples. Se padrões observados mostram duração não geométrica, usar distribuições explícitas de duração/semi-Markov como extensão. Comparar duração e frequência dos bouts, não só acurácia por janela.
3. **Política de decisão:** treinar primeiro coeficientes por máxima verossimilhança/inferência de escolha em dados de movimento e disponibilidade ambiental; validar a distribuição de próximas ações. A identificação requer dados de ambiente/contexto e, para causalidade de manejo, intervenções ou estudos controlados.
4. **RL só com ambiente calibrado:** usar RL offline ou simulação com domínio randomizado apenas depois de ajustar estado ecológico e dinâmica; penalizar violação de invariantes, comparar com baseline em cenários não vistos e manter trilha de decisões/recompensa. Avaliar adesão a padrões de atividade, distâncias, dieta/ingestão, uso espacial e resposta contrafactual experimental. A recompensa é hipótese operacional e precisa constar na documentação do modelo.
5. **Promoção:** um modelo aprendido só substitui camada se melhora previsão retida e cenários de intervenção sem quebrar balanços e limites. Versionar pesos, dados/licenças, atributos, split, semente, métricas e validade por raça/sistema.

GPU Kaggle pode acelerar CNNs e varredura de experimentos; inferência para poucas dezenas/centenas de bovinos não exige GPU remota. Treinamento offline não transfere automaticamente uma política para o jogo. Use como pré-requisito dados de sensores/situação que cubram os comportamentos propostos, e não apenas vídeos ou classes sem contexto ambiental. O catálogo de conjuntos e disponibilidade de GPU está na [pesquisa geral](landscape.md).

## Resultados validados vs. aproximações para o produto

| Mecanismo a exibir | Base de evidência | Categoria correta no Campo Vivo agora |
|---|---|---|
| Horários/bouts de pastejo, ruminação e repouso | Observação longitudinal e classificadores avaliados em sistemas descritos. | **Faixas observadas** por classe; usar horários/taxas como alvos de calibração, não lei universal. |
| Ruminação e ingestão | Relações com atividade, dieta e oferta; modelos de ingestão de sucesso variável entre pasto e indivíduo. | **Mecanístico/empírico local** somente depois de escolher população e calibrar; o produto atual não deve alegar DMI real. |
| Consumo de água | Equações ajustadas/validadas para populações específicas, mais associação entre competição e acesso. | **Estimativa condicional**; evento de beber e hidratação intracorporal simplificada continuam como aproximação. |
| Resposta a calor | THI e respostas medidas; há modelos térmicos dinâmicos experimentais. | **THI como sinal ambiental**; corpo térmico é aproximação até reimplementar e validar equações/condições. |
| Exploração, retorno e padrão social | Trajetórias observadas, estudos de vínculos, experimento de memória, ABMs de caso. | **ABM fundamentado em hipóteses**; os parâmetros do jogo não são política validada para uma fazenda real. |
| Preferências, humor, personalidade | Individualidade e repetibilidade são observáveis indiretamente, mas “traços” do jogo precisam ser estimados por dados longitudinais. | **Parâmetros de personagem**, não fenótipo nem diagnóstico científico, até validação. |

## Fontes principais

- Guo, Y. et al. (2009). “Using accelerometer, high sample rate GPS and magnetometer data to develop a cattle movement and behaviour model.” *Ecological Modelling* 220, 2068–2075. [DOI](https://doi.org/10.1016/j.ecolmodel.2009.04.047).
- Jochims, F. et al. (2023). “Variations in the 24 h temporal patterns and time budgets of grazing, rumination, and idling behaviors in grazing dairy cows in a New Zealand system.” *Journal of Animal Science*. [DOI](https://doi.org/10.1093/jas/skad038).
- Barwick, J. et al. (2021). “Classification of Cattle Behaviours Using Neck-Mounted Accelerometer-Equipped Collars and Convolutional Neural Networks.” *Sensors* 21(12), 4050. [DOI](https://doi.org/10.3390/s21124050).
- López, A. et al. (2022). “Analysis of Accelerometer and GPS Data for Cattle Behaviour Identification and Anomalous Events Detection.” *Entropy* 24(3), 336. [DOI](https://doi.org/10.3390/e24030336).
- Dillard, S. L. et al. (2016). “A spatially explicit, agent-based model for simulating movements of cattle grazing corn residues.” *Agricultural Systems*. [DOI](https://doi.org/10.1016/j.agsy.2016.06.007).
- Bailey, D. W. et al. (2018). “An agent-based model of cattle grazing toxic Geyer’s larkspur.” *Ecological Modelling*. [DOI](https://doi.org/10.1016/j.ecolmodel.2017.11.018).
- Léger, L. et al. (2023). “On-farm evaluation of models to predict herbage intake of dairy cows grazing temperate semi-natural grasslands.” *Animal*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/37148624/).
- González, L. A. et al. (2017). “Use of sensor-determined behaviours to develop algorithms for pasture intake by individual grazing cattle.” *Crop & Pasture Science*. [DOI](https://doi.org/10.1071/CP16383).
- Hyer, J. C., Oltjen, J. W. & Galyean, M. L. (1991). “Development of a model to predict forage intake by grazing cattle.” *Journal of Animal Science* 69(2), 827–835. [DOI](https://doi.org/10.2527/1991.692827x).
- Azevedo, R. A. et al. (2019). “Prediction of water intake to Bos indicus beef cattle raised under tropical conditions.” *Journal of Animal Science*. [DOI](https://doi.org/10.1093/jas/skz003).
- Mader, T. L. & Davis, M. S. (2018). “Environmental effects on water intake and water intake prediction in growing beef cattle.” *Journal of Animal Science*. [DOI](https://doi.org/10.1093/jas/sky267).
- Cardot, V., Le Roux, Y. & Jurjanz, S. (2008). “Drinking behavior of lactating dairy cows and prediction of their water intake.” *Journal of Dairy Science* 91, 2257–2264. [DOI](https://doi.org/10.3168/jds.2007-0208).
- Li, H. et al. (2020). “Predicting rectal temperature and respiration rate responses in lactating dairy cows exposed to heat stress.” *Journal of Dairy Science*. [DOI](https://doi.org/10.3168/jds.2019-17583).
- Liu, K. et al. (2022). “Development and evaluation of a thermoregulatory model for predicting thermal responses of dairy cows.” *Computers and Electronics in Agriculture*. [DOI](https://doi.org/10.1016/j.compag.2022.107170).
- Gabrieli, R. & Malkinson, D. (2022). “Social organization and fitness response in grazing beef cows – Understanding through interactions and activity measuring.” *Applied Animal Behaviour Science* 254, 105723. [DOI](https://doi.org/10.1016/j.applanim.2022.105723).
- Sueur, C. et al. (2009). “Individual and social determinants of spontaneous group movements in cattle and sheep.” *Animal* 3(9), 1319–1326. [DOI](https://doi.org/10.1017/S1751731109004790).
- “Cattle Do Remember Locations of Preferred Food over Extended Periods.” [DPI eResearch Archive](https://era.dpi.qld.gov.au/id/eprint/8369/).
- Bracis, C., Gurarie, E., Van Moorter, B. & Goodwin, R. A. (2015). “Memory Effects on Movement Behavior in Animal Foraging.” *PLOS ONE* 10(8), e0136057. [DOI](https://doi.org/10.1371/journal.pone.0136057).
- “Pre-Grazing Herbage Mass Affects Grazing Behavior, Herbage Disappearance, and the Residual Nutritive Value of a Pasture during the First Grazing Session.” [Artigo integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC7070351/).
- “Spatial and temporal variation in proximity networks of commercial dairy cattle in Great Britain.” [Artigo integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC8385416/).

