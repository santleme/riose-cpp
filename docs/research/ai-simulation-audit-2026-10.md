# Auditoria de IA, simulação e fundamentação científica do Campo Vivo

**Data:** 2026-10-22  
**Escopo:** comportamento, modelos, treinamento, simulação, dados e validação. A camada visual foi excluída.

## Diagnóstico executivo

O estado atual é um **protótipo de agente baseado em regras com duas experiências de ML**, não um simulador científico de bovinos. Há evidência útil de que o pipeline Kaggle executou em uma Tesla T4, de que o classificador consegue separar parte das janelas do conjunto usado e de que o DQN aprendeu a maximizar a recompensa da sua pequena MDP sintética. Não há evidência de que o DQN aprendeu comportamento bovino, de que o classificador generaliza para fazendas/raças/sensores diferentes ou de que os estados chamados fome, sede, saúde, estresse, pasto e calor correspondam a grandezas fisiológicas mensuradas.

O achado central é um **desalinhamento de três mundos**: `train_policy.py` aprende transições artificiais; `src/simulation.ts` implementa outras transições artificiais; e o dataset supervisionado descreve atividades observáveis, não estados internos nem decisões causais. A integração do DQN transfere apenas o argmax de cinco ações entre mundos com semântica e custos diferentes.

| Área | O que foi demonstrado | O que não foi demonstrado |
|---|---|---|
| DQN em Kaggle | 50 mil passos, 12.373 atualizações, CUDA/T4; supera uma heurística fraca na recompensa da mesma MDP sintética e nos 20 seeds registrados. | Bem-estar real, escolha de pastejo, resposta à água/calor, interação multiagente ou validade no motor TypeScript. |
| Classificador IMU | Uma divisão por animal (treina em cinco, valida em uma, testa em uma) alcançou macro-F1 ~0,49 no MLP selecionado por validação. | Generalização estimada com precisão, atividade de cinco classes com rótulos de procedência confirmada, transferibilidade ou inferência de motivação/fisiologia. |
| Motor de simulação | Regras explícitas e determinísticas/stochasticamente inicializadas para quatro agentes, recursos abstratos e ações. | Conservação de matéria/energia/água, crescimento, ingestão, termorregulação, doença, dinâmica de pasto, causalidade ou comportamento social validado. |

**Prioridade técnica:** congelar a política DQN como experimento de infraestrutura, não tratá-la como cérebro do rebanho; corrigir e ampliar primeiro a camada de dados e o modelo mecanístico. Não há motivo de desempenho para migrar agora para C++/WASM: quatro bovinos e uma MLP pequena não justificam esse custo. O bloqueio é validade e desenho experimental, não capacidade de CPU/GPU.

## 1. O que o repositório realmente contém

### 1.1 Política DQN

`training/policy/train_policy.py` define um ambiente de **um único bovino** com nove valores entre 0 e 1, cinco ações e horizonte de 240 passos. Cada episódio começa amostrando valores independentes em intervalos escolhidos manualmente. A dinâmica, ruído, regeneração de água/capim, custo das ações, coeficientes da recompensa e limiares são autoria do código, sem ajuste a observações. O próprio artefato declara que aprende apenas a dinâmica sintética.

O treino registrado foi um seed (41), 50.000 interações, rede 9→64→64→5, replay buffer, AdamW e desconto 0,985. Foram feitas 12.373 atualizações em CUDA/Tesla T4. O modelo compara-se com uma política gulosa de utilidade de um passo, também construída com os mesmos coeficientes inventados. A comparação pareada sobre 20 episódios resultou em retorno médio −17,35 ± 3,42 para DQN contra −50,20 ± 4,88 para a heurística; a diferença pareada foi +32,85 ± 2,86 (desvio-padrão entre episódios). A média final dos cinco estados de “necessidade” foi 0,133 contra 0,307. São resultados válidos **dentro da implementação e dos seeds publicados**, e só dentro dela.

O comparador é fraco: calcula benefícios instantâneos de fome/sede/fadiga/calor/socialidade e escolhe o máximo; ignora a recompensa completa, a persistência das necessidades, o futuro descontado e a interação entre transições. O DQN vencer esta referência mostra aprendizagem da função objetivo definida, mas não estabelece que a função objetivo seja boa, que a política seja biologicamente plausível ou que RL seja melhor que uma política devidamente otimizada. Uma comparação mais informativa inclui baseline de prioridade por necessidade, MPC/oráculo com modelo conhecido, ablações e políticas com regras científicas.

Há só uma semente de treinamento. O arquivo de avaliação informa que os seeds de avaliação são distintos dos usados pelo treino; isso reduz vazamento direto do gerador de episódio, mas não substitui repetir o treino, testar famílias de ambientes reservadas ou manter um conjunto de avaliação selado entre versões. Os 20 seeds visíveis são válidos para a comparação pareada registrada, mas o repositório não preserva um conjunto de teste separado e selado para revisões futuras da recompensa/política. O relatório também registra retorno e médias finais, mas não curvas de aprendizagem, variância entre seeds de modelo, sensibilidade a parâmetros, calibração, teste fora de distribuição nem análise de trajetórias.

A recompensa é uma escolha normativa, não uma medida de bem-estar. Os pesos 0,42/0,52/0,18/0,34/0,12, o custo de recurso, a penalidade por ação indisponível e os cortes acima de 0,88 são arbitrários. O termo de diferença de potencial está misturado com uma recompensa de estado; portanto não deve ser interpretado como shaping que preserva automaticamente a política ótima. A sede persiste no DQN (média final 0,523), apesar da recompensa melhor, e água/capim regem-se em taxas fixas sem consumo/produção física. Contagens de ação mostram uso intenso de buscar sombra mesmo quando a carga térmica final está praticamente zero; isso exige inspeção de política e ablação, não é evidência de preferência bovina.

Há ainda uma questão de horizonte no código: os episódios terminam por truncamento, mas a transição gravada marca como terminal apenas `terminated`, que é sempre falso. Isso faz o alvo bootstrap atravessar o limite de 240 passos. Pode ser uma formulação intencional de tarefa contínua, mas não corresponde a um horizonte finito terminal e precisa ser declarado e comparado com ambas as formulações.

### 1.2 Classificador de atividade IMU

`training/train.py` transforma sinais tri-axiais em janelas não sobrepostas de 5 s, extrai médias/desvios/mínimos/máximos e estatísticas da magnitude e treina árvores e MLP. A divisão principal é melhor que uma divisão aleatória de janelas: o animal 1 é usado para seleção, o animal 6 fica reservado e os demais entram no ajuste. Isso evita misturar janelas da mesma vaca entre treino e teste. É um ponto metodológico positivo.

O melhor modelo **escolhido pela validação**, MLP scikit-learn, teve macro-F1 0,543 na vaca de validação e 0,491 na vaca de teste. Extra Trees e Random Forest tiveram macro-F1 de teste 0,488 e 0,485. A MLP PyTorch no CUDA obteve 0,510 no teste, mas apenas 0,420 na validação; não substituir o modelo selecionado pelo resultado do teste é correto. F1 ponderado de 0,814 parece forte porque as janelas dominantes são caminhada, alimentação e ruminação. No teste há apenas 14 janelas de “deitado” e 8 de “em pé”; na validação, 3 de “deitado” e nenhuma de “em pé”. O macro-F1 e o suporte por classe são, portanto, leituras mais honestas que a acurácia/ponderada. A incerteza efetiva é de **uma vaca de teste**, não 6.521 amostras independentes.

O trabalho fonte ligado ao conjunto de anel nasal reporta sete bovinos e reconhece três rótulos gerais — alimentação, ruminação e “outro”; os autores dizem não conseguir separar com segurança parado em pé de deitado. O script Campo Vivo lê cinco códigos (`feeding`, `rumination`, `standing`, `lying`, `walking`). Essa diferença pode resultar de versão ou anotação posterior do upload Kaggle, mas a procedência e a definição operacional das cinco classes não estão demonstradas pelo paper. Até rastrear os rótulos do CSV à anotação original, não atribuir ao estudo original a validade das classes separadas em pé/deitado/caminhando.

Riscos do processamento que ainda precisam ser verificados nos dados brutos:

- O código remove linhas com qualquer sensor/rótulo ausente **antes** de formar janelas. Se faltam amostras no meio de um sinal, pontos de lados diferentes de uma lacuna tornam-se vizinhos falsos dentro da janela. Ele não confere timestamp monotônico, duplicado, taxa de amostragem efetiva, gaps nem reinicia janela após descontinuidade.
- O código confia na ordem recebida por vaca/dia; a coluna `time` é exigida, mas não é usada para ordenar/verificar o sinal. A janela pressupõe 10 Hz e usa maioria ≥80%; uma janela que contenha até 20% de transição recebe um rótulo único.
- As estatísticas simples descartam a forma temporal e podem confundir atividades com postura/orientação do sensor. Não há baseline de persistência, ROCKET/TCN, calibração, intervalo de confiança agrupado por vaca ou validação cruzada repetida por animal.
- São 7 animais e um domínio. A separação por indivíduo evita a forma mais óbvia de vazamento, mas uma validação e um teste em animais únicos não estimam bem a variabilidade animal, sessão, lote, sensor, raça ou fazenda.
- `CC0-1.0` é uma constante no script e metadado nos relatórios do experimento; o arquivo bruto e um snapshot/hash/licença verificada não estão versionados. O classificador treinado também não está nos artefatos locais listados. Reproduzir exige recuperar a mesma versão do Kaggle e conferir licença, esquema e conteúdo.

O classificador é de **atividade observada**. Não mede fome, sede, motivação, saúde ou efeito causal de irrigação/manejo. Uma etiqueta de ruminação não identifica os nutrientes ingeridos, e um acelerômetro não demonstra que um animal bebeu água.

### 1.3 Simulação em execução

`src/simulation.ts` cria quatro indivíduos com raça, idade, atributos e necessidades inicializados por fórmulas e valores escritos à mão. São parâmetros de personalidade de jogo, não estimativas individuais. Os estados são escalas 0–100 sem unidade. A pastagem tem 32 células em grade e condições iniciais determinísticas por índice; o clima tem quatro categorias e temperaturas fixas.

No `step`, solo, água, umidade e capim atualizam por coeficientes fixos. `soil` não alimenta nenhum cálculo posterior; chuva não reabastece o reservatório de água, e água não afeta pasto. O capim regenera por umidade e pressão média de lotação, sem radiação, temperatura, espécie forrageira, biomassa seca/ha, composição, altura, fenologia, nutrientes ou água disponível do solo. O pasto global é apenas média de células. Não existe balanço de massa entre ingestão, recurso e crescimento.

Fome e sede aumentam linearmente, sem peso corporal, ingestão de matéria seca, digestibilidade, energia metabolizável, proteína, rumen, sal/minerais, excreção ou manutenção. A ação de pastejar subtrai `0.38 × horas` da célula e reduz fome em `0.95 × horas`, sem relação com massa do animal, bites, disponibilidade/qualidade ou custo de pastejo. `rumination` sobe diretamente ao pastar e descansar, sem um ciclo de ingestão, ruminação e passagem. “Saúde” melhora ou piora por limiares arbitrários de fome/sede/estresse, sem patologia, diagnóstico, risco, recuperação ou mortalidade. Não há peso, crescimento, condição corporal, reprodução, lactação ou idade fisiológica.

O calor é somente temperatura do ar menos um limiar dependente de um atributo manual `heatTolerance`. Não há umidade, vento, radiação solar, carga térmica, aclimatação, pelagem por raça calibrada, temperatura corporal ou resposta fisiológica. Estresse mistura calor, capim global baixo e decaimento constante. As diferenças de raça que existem são três traços escolhidos no código.

O rebanho é reduzido ao centróide calculado uma vez antes de percorrer animais. Não há rede de relações, preferências de parceiros, dominância, liderança, proteção de bezerro, facilitação, competição localizada, sincronização ou formação de subgrupos. Como animais atualizam patches em sequência, animais posteriores podem observar pasto já alterado no mesmo passo, mas todos usam o centróide antigo; o resultado depende da ordem do array. “Seguir rebanho” move ao centróide (incluindo o próprio animal), e não é uma dinâmica coletiva aprendida.

### 1.4 DQN integrado ao motor

O DQN treinado recebe `[hunger, thirst, fatigue, heat_load, grass, water, shade, neighbor_presence, social_stress]`. O adaptador preenche essas entradas com proxies de fome/sede/fadiga/estresse do motor, distância à sombra/rebanho, e disponibilidade abstrata. A política não vê raça, idade, traço individual, hora, duração de ação, distância à água, alvo ou histórico.

Os nomes das ações coincidem, mas as consequências não. No treino, `graze` reduz diretamente fome e capim, `drink` reduz sede, `rest` recupera fadiga e `seek_shade` reduz calor instantaneamente. No app, essas ações comandam deslocamento e só produzem efeito depois de chegar a um ponto; caminhar tem custo, as velocidades e taxas são outras, água é consumida globalmente, e o DQN não foi exposto ao mapa nem às consequências de movimento. O DQN é chamado a cada passo de 0,25 h, enquanto a MDP sintética não estabelece nem valida um mapeamento temporal/fisiológico para esse intervalo. Portanto o retorno Kaggle não é teste do comportamento integrado.

## 2. Evidência científica e ativos de dados aproveitáveis

Os conjuntos abaixo servem a perguntas diferentes. Nenhum, isoladamente, ensina uma política causal completa.

| Fonte | O que mede e pode ensinar | Limite crítico |
|---|---|---|
| [Nose-ring, paper original](https://pmc.ncbi.nlm.nih.gov/articles/PMC11047668/) e [mirror Kaggle](https://www.kaggle.com/datasets/fandaoerji/cow-nose-ring-data-set) | IMU sincronizada com vídeo, experimento com sete vacas; baseline observável de alimentação/ruminação/outro. | Um único ambiente e poucas vacas; ontologia descrita no artigo não separa claramente as cinco classes usadas pelo script Campo Vivo. |
| [AcTBeCalf](https://zenodo.org/records/13259482), 30 bezerros e 13 semanas | IMU de 25 Hz, identidades e segmentos comportamentais; contém split por sujeito e baselines Random Forest/mini-ROCKET. Útil para comparar pipelines temporais e testar leakage-safe. | Bezerros leiteiros, não Nelore adulto em pasto; ainda não é preferência nem fisiologia. |
| [Precision Beef](https://zenodo.org/records/4064802) | Sensores em 18 bovinos e três ensaios; rótulos de halter Rumiwatch incluem eating/rumination/other e protocolo de teste documentado. | Rótulo derivado de sensor/proxy, não necessariamente observação independente; beber é mesclado em eating segundo a documentação. |
| [MmCows](https://github.com/neis-lab/mmcows), [paper NeurIPS](https://papers.nips.cc/paper_files/paper/2024/file/6d8f3f71b22f9d2e9320d7bdb73acea7-Paper-Datasets_and_Benchmarks_Track.pdf), [card HF](https://huggingface.co/datasets/neis-lab/mmcows) | Dados multimodais: IMU/UWB, vídeo anotado, temperatura corporal e ambiente; 16 vacas em instalação leiteira, sensores em subconjunto; benchmark útil para atividade, localização e sincronização. | Um único local e observação manual densa em parte do período/animais; escala grande. Código MIT não significa automaticamente licença permissiva para dados/pesos; card HF declara CC BY-NC-SA 4.0. |
| [IoT collar open pasture](https://github.com/WASP-lab/db-cow-walking), [paper/dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC12426180/) | 10 vacas a pasto em ~5 meses; aceleração + velocidade angular, 441 eventos / 7h34 de rótulos, 11 comportamentos. A publicação relata SVM com F1 0,9625 no protocolo do trabalho. | Muitas observações/eventos, mas poucas horas rotuladas e 10 animais; essa métrica não equivale a generalização para outra fazenda. Bom teste externo de domínio, após auditar split. |
| [Japanese Black behavior](https://zenodo.org/records/5849025) | 6 bovinos, 25 Hz, um dia e 13 classes com voto de três anotadores; cobre amplo etograma. | Amostra muito pequena, forte desbalanceamento; os metadados públicos de licença precisam ser reconciliados antes de uso derivado. |
| [Sahel GPS/HMM, CIRAD](https://dataverse.cirad.fr/dataset.xhtml?persistentId=doi:10.18167/DVN1/GHJKQO) | 15 colares / 21 zebu em 19 rebanhos, posição a cada 30 min entre 2021–2023; comportamento/local inferidos via HMM e contexto pastoril. Excelente para padrões de movimento e mudança de domínio. | Rótulos são saídas de HMM, não observação direta; licença CC BY-NC-SA 4.0 limita redistribuição e uso comercial. |
| [Nellore supplementation data](https://data.mendeley.com/datasets/3wdjvxmvsp/7) | 26 Nelore, piquete de Brachiaria, pesos seriados, suplementação/visitas ao cocho e índices ambientais/satélite ao longo de ~10 meses. É a fonte mais aderente encontrada para começar calibração brasileira de ganho de peso. | Uma fazenda/piquete e coorte pequena; medidas de crescimento são espaçadas e não registram trajetórias contínuas, ingestão individual total ou intervenção randomizada suficiente para identificar toda causalidade. |
| [Dataset de colares/feeding, Zenodo](https://zenodo.org/records/6784671) | Até 48 vacas com acelerômetros por 68 dias e dados de feeder, útil para atividade e eventos de alimentação. | Rótulo via presença/uso de feeder não prova ingestão/ruminação; apenas subconjunto e período curto são anotados. |
| [CVB visual behavior](https://github.com/RenukaSharma/CVB_dataset) | Clipes de pastagem e classes observáveis multi-bovino. | Ajuda a estudar percepção/atividade e localização visual, não estados internos ou efeito de manejo; licenças dos dados devem ser confirmadas no DOI. |
| [Kaggle visits to the trough](https://www.kaggle.com/datasets/juliavalente/data-from-visits-to-the-trough-of-nellore-cattle), [paper socialh](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0271337), [manual R](https://valentebr.r-universe.dev/socialh/doc/manual.html) | 90.211 eventos de machos Nelore em teste de eficiência alimentar; paper associado descreve 37 animais, 21 dias e 5 cochos GrowSafe. Kaggle bruto registra ID animal/equipamento e horários de entrada/saída, úteis para frequência, duração e uso temporal de recurso. | CLI Kaggle marca CC BY-NC-SA 4.0. O paper informa que os sistemas de cocho também mediram consumo; esses valores não estão no CSV Kaggle atual. `socialh` distribui objeto de sete colunas (inclui duração, consumo e `pen`) para os mesmos 90.211 registros, mas lineage/associação linha a linha com o upload Kaggle não está documentada. Duração deriva de `OUT-IN`; consumo tem precedente de medição no estudo mas não é verificável com o upload; `pen` também requer procedência. Não usar as colunas extras até confirmar versão/linhagem. A licença GPL-3 do pacote e a licença do dado são coisas separadas. |

**Verificação do dataset Kaggle de visitas (executada localmente em 2026-10-22):** a API oficial do CLI informa licença `CC-BY-NC-SA-4.0`; o ZIP tem 890.166 bytes e SHA-256 `fd2cfd016503dcff40be2931a6bd04a7185d020c811f6cafa08ec3925d9b5884`; o CSV contém 90.211 registros. O cabeçalho/colunas são `animal_id`, `equip_id`, `IN`, `OUT` (arquivo usa campos separados por espaço com aspas), sem volume ingerido, composição da dieta ou rótulo comportamental. A auditoria do arquivo encontrou 38 IDs (um com apenas um evento), 5 equipamentos, 21 dias (2021-06-25 a 2021-07-15), 17 durações zero e 3.471 durações acima de 300 s. O pipeline executável baixou o arquivo pela URL pública do Kaggle, conferiu tamanho e SHA fixados e repetiu esses números. O artigo PLOS associado descreve 37 machos Nelore em 5 cochos GrowSafe por 21 dias e informa que o sistema também mede consumo. Esses valores, contudo, não estão no CSV Kaggle atual; a diferença 37 vs 38 IDs permanece sem explicação. O pacote R `socialh` descreve uma versão de sete colunas, incluindo duração, consumo e `pen`, e aponta o mesmo Kaggle; a origem/linhagem dos campos adicionais e a correspondência registro a registro não estão estabelecidas no paper/esquema do upload que verifiquei. Usar apenas ID/equipamento/timestamps no baseline; derivar duração de `OUT-IN` após validação. Só incorporar consumo/pen após comparar o objeto do pacote com a versão Kaggle congelada e encontrar documentação da transformação. A licença NC-SA impede presumir uso comercial ou redistribuição irrestrita. A leitura pública e a autenticação para submissão Kaggle funcionaram; a primeira execução de experimento está resumida abaixo.

**Verificação do experimento de sombra (fonte primária):** [Mendeley Data, CC BY 4.0](https://data.mendeley.com/datasets/cctgpznjrc/1) fornece 19.388 eventos de bebida, 3.854 registros diários de matéria seca, 282 pesos em seis tempos e dados meteorológicos de alta frequência, em 47 machos Nelore de ~450 kg por 83 dias. O estudo tem sinal real de tratamento — animais sem sombra beberam em média 8% mais por dia, mas ingestão de matéria seca e ganho médio não diferiram — e inclui temperatura, umidade, vento, radiação e THI. Entretanto, os animais foram mantidos em quatro piquetes coletivos, os próprios autores dizem não excluir efeitos de grupo/pseudorrepetição, e o conjunto não representa pastagem extensiva. Use como calibração/validação de consumo de água e uso de sombra em confinamento tropical; qualquer estimativa de efeito causal deve considerar agrupamento por piquete e o protocolo original. [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0331238).

**Acesso de arquivos de sombra (verificado em 2026-10-22):** a página Mendeley declara CC BY 4.0, mas a listagem pública de arquivos e o endpoint do ZIP retornaram HTTP 401 sem autenticação; DataCite traz simultaneamente direito `embargoedAccess` e CC BY 4.0. Não houve download, inspeção do workbook, hash/esquema de linhas nem comparação row-level com BR-CORTE. O relatório e probe estão em [`training/data/reports/nellore-shade-access-and-validation.md`](../../training/data/reports/nellore-shade-access-and-validation.md) e [`training/data/audit_nellore_shade.py`](../../training/data/audit_nellore_shade.py). Médias de paper permitem apenas cálculo ilustrativo, não bias/MAE de modelo. Além disso, a página informa registros meteorológicos a cada 15 min, enquanto métodos do paper dizem horário; resolução deve ser resolvida a partir do arquivo original antes de alinhar temperatura/umidade diária.

**Primeira prioridade Kaggle, sem confundir fonte com qualidade:** o conjunto bruto de 90.211 eventos foi baixado e inspecionado; é útil para horários/duração de visitas, sem consumo no arquivo, com limite NC-SA. O artigo e o pacote R apontam para relações sociais/dominância em cochos eletrônicos, mas não convertem automaticamente todo acesso em interação observada. A autenticação Kaggle foi configurada durante esta execução; um notebook T4 para o experimento de ganho de peso foi submetido e concluído. A sessão local não tinha PyTorch CUDA, por isso os estimadores sklearn rodaram em CPU apesar do runtime T4; não se deve reportar esse ajuste como treinamento GPU. O conjunto de sombra Mendeley e o conjunto UFMS entram por oferecerem targets diretamente relevantes para fisiologia/ambiente, com licenças abertas explicitadas nos registros.

**Leitura sobre vazamento e generalização.** Estudos de validação de acelerometria animal mostram que separar aleatoriamente janelas correlacionadas infla resultados; o agrupamento por indivíduo reduz esse risco, mas não resolve mudança de sensor/fazenda. Um estudo específico de bovinos compara random holdout com leave-cow-out e observa desempenho menor no segundo. A implementação Campo Vivo já faz um holdout por vaca, que merece crédito, mas com apenas um animal em validação e um em teste a estimativa permanece frágil. Guias mais gerais sobre acelerometria animal também discutem correlação temporal e testes não independentes (a revisão exclui pecuária; serve como referência metodológica, não evidência específica para bovinos). [Ferdinandy et al. (2020)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0236092); [estudo bovino de CV (2025)](https://doi.org/10.1016/j.atech.2025.101483); [Wilson et al. (2025)](https://doi.org/10.1111/1365-2656.70054).

**Modelos existentes a aproveitar como referências:**

- O modelo estocástico individual de vaca-bezerro de Aherin et al. calcula nutrição, crescimento, reprodução e saúde diariamente e foi comparado a dados CHAPS. É uma boa arquitetura e exemplo de validação quantitativa, mas os autores encontraram diferenças de desempenho e parâmetros/gestão subjacentes aos dados de comparação eram desconhecidos. É referência para adaptar e validar, não verdade de ground truth ou modelo pronto para Nelore brasileiro. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9930734/).
- O [NASEM-Model-Python](https://github.com/CNM-University-of-Guelph/NASEM-Model-Python) implementa o modelo de bovino leiteiro NASEM 2021 e é fonte para cálculos de exigências/dieta em seu domínio. O repositório declara MIT para parte do código, mas informa que equações foram copiadas de código R sob termos separados. Não assumir que todo conteúdo é MIT nem que o escopo leiteiro representa bovinos de corte Nelore. Para bovinos de corte, o [NASEM 2016](https://nap.nationalacademies.org/catalog/19014/nutrient-requirements-of-beef-cattle-eighth-revised-edition) é a referência publicada a avaliar, mas os direitos/software disponíveis precisam ser verificados; não equivale a uma implementação aberta pronta.
- [GrazeIn](https://doi.org/10.1111/j.1365-2494.2010.00770.x) e sua validação externa modelam ingestão de forragem e produção em vacas leiteiras a pasto com variáveis de manejo e pastagem. É referência de intake condicionado ao sward, não simulador de decisão individual completo.
- O componente [APSIM/AgPasture](https://apsimdev.apsim.info/ApsimX/Documents/AgPastureScience.pdf) fornece uma referência robusta para acoplar clima, solo e crescimento de pasto; o framework documenta módulos e validação. APSIM tem licença diferenciada: gratuito para educação/pesquisa de bem público, enquanto uso comercial exige licença apropriada; confirmar nos [termos oficiais](https://registration.apsim.info/) antes de reutilizar código/modelos.
- Há precedentes de movimento de bovinos com HMM treinado em GPS/IMU para estados ocultos (forragear, deslocar, deitar) e de ABM espacial calibrado contra GPS em experimentos de pastejo de resíduos. São bem mais próximos da mobilidade real do que o centróide atual, mas requerem observações georreferenciadas da fazenda-alvo. [HMM de movimento](https://doi.org/10.1016/j.ecolmodel.2009.04.047); [ABM espacial com GPS](https://doi.org/10.13031/aim.20162460297).

## 3. O que aprender e o que calcular

### Aprender de dados

- **Percepção observável:** classificar atividade/postura/eventos a partir de IMU, GPS, cocho ou vídeo, com taxonomia alinhada à anotação e confiança calibrada.
- **Duração e transição de bouts:** modelo semi-Markov/HMM/TCN/GRU para duração/ordem de atividade condicionada a hora, ambiente, indivíduo e contexto, se a sequência tiver timestamps e rótulos confiáveis.
- **Movimento e preferência espacial:** distribuição de distância, velocidade, turn angle, área de permanência, seleção de manchas, relação local com biomassa/água/sombra e efeito de gestão, usando trajetória GPS/UWB e medição ambiental co-localizada.
- **Heterogeneidade individual:** efeitos aleatórios/hierárquicos por animal e covariáveis reais (idade, peso, raça, lactação, histórico) depois de haver coortes suficientes; os “traços” não devem ser números escolhidos pelo design.
- **Previsão empírica auxiliar:** duração de visita a cocho, probabilidade de mudança de atividade, temperatura/atividade, desde que o alvo seja descrito como previsão e não causalidade.

### Calcular com modelos mecanísticos/contabilidade

- **Crescimento e nutrientes:** peso, maturidade, consumo de matéria seca, energia/proteína, manutenção, ganho, gestação/lactação quando presentes, usando equações adequadas a tipo racial/fase produtiva e declarando seus domínios e incerteza.
- **Ingestão/ruminação:** intake condicionado por massa/estrutura/qualidade do pasto, disponibilidade, suplemento e capacidade do animal; ruminação vem após ingestão/fibra/digestibilidade, não de um bônus arbitrário ao selecionar `graze`.
- **Clima/água/termorregulação:** balanços de água/energia e índice térmico que inclua temperatura, umidade, vento e radiação; resposta depende de aclimatação e categoria animal. Temperatura isolada não justifica estado de heat load fisiológico.
- **Solo/pastagem:** água no solo, evapotranspiração, crescimento, senescência, composição/qualidade e remoção por pastejo, com unidades e limites de conservação. Separar célula/paddock e passo temporal.
- **Saúde:** risco ou progresso de condição específica com parâmetros publicados/estimados e observáveis pertinentes. Sem dados clínicos, não inventar uma barra `health` como predição veterinária.

### Aprendizado por reforço e sistema multiagente

RL passa a fazer sentido **depois** que observações, transições, ações e resultados da dinâmica do simulador estejam validados e o objetivo seja explicitado. Dados passivos de atividade não identificam o que teria acontecido sob outra ação de manejo; BC/IRL também herdam o viés das ações observadas. Começar por política de referência transparente, comportamento clonado em trajetórias de campo e modelos de duração/transição; validar em domínio independente. Depois usar RL em mecânica calibrada, com sementes repetidas, avaliação de política fora do treino, ablação, restrições de segurança/welfare e relatório de trade-offs. Uma recompensa escalar não deve esconder trocas entre ingestão, desidratação, calor, locomoção e pastagem.

Para comportamento coletivo, treinar/ajustar modelo de interação exige posições sincronizadas de mais de um animal e eventos de interação rotulados. Uma representação inicial adequada é um grafo temporal de vizinhança (distância/orientação/densidade), com efeitos individuais e ambientais, não um `neighbor_presence` escalar. Manter como camadas separadas: mecanismo físico/biológico por animal, influência social observada e modelo de decisão.

## 4. Arquitetura-alvo recomendada

Manter o motor de tempo real no navegador como orquestrador. Dividir a simulação por responsabilidade e taxa temporal:

1. **Ambiente (horas–dias):** clima real/sintético, solo/água, estratos de pastagem e fenologia; entradas e saídas com unidades físicas e balanço de massa. APSIM/AgPasture serve como referência, sujeito à licença.
2. **Fisiologia (horas–dia):** peso/condição corporal, ingestão, energia/proteína, água e carga térmica; parâmetros e intervalos explícitos por categoria. Começar com um tipo produtivo brasileiro (por exemplo, Nelore em recria) e uma tarefa bem definida, não com todas as raças e fases.
3. **Atividade (minutos):** estado oculto/bout semi-Markov condicionado por necessidade, hora, clima e indivíduo; classificador de sensor é evidência de atividade, não motor causal.
4. **Movimento (segundos–minutos):** modelo espacial condicionado a objetivo local, custos de deslocamento, barreiras, gradientes de recurso e trajetória; calibrar contra GPS e validar em pasto/estação reservados.
5. **Social (minutos):** grafo dinâmico por observação e efeitos de aproximação/evitação/sincronia medidos; ablações para testar se a camada explica algo além do ambiente compartilhado.
6. **Política aprendida (opcional, posterior):** recomenda próxima ação/bout sob estado estimado, sem poder violar invariantes do motor; iniciar com BC ou semi-Markov. RL só com simulador calibrado e eval independente.

O motor não precisa ser reescrito em C++ para isso. Definir contratos e um esquema de estado com unidades primeiro. Python é conveniente para análise/treino; inferência pequena pode ser exportada para TypeScript/ONNX quando houver ganho funcional. Se o desempenho futuro exigir compilação, otimizar o núcleo de dinâmica e manter o mesmo contrato e provas de paridade.

## 5. Plano experimental prioritário e portas de qualidade

### P0 — Reprodutibilidade e ontologia (antes de treinar de novo)

- Congelar versão, DOI/ID Kaggle, hash dos arquivos, licença do dado, condições de acesso, dicionário de rótulos, unidade/frequência e código de pré-processamento.
- Resolver por fonte primária a discrepância 3-vs-5 classes do conjunto nose-ring. Preservar `unknown/other` quando a evidência não separar postura/atividade.
- Construir segmentos por animal/sessão e timestamp; validar ordem, frequência, duplicatas, gaps, labels e alinhamento antes de extrair janelas. Não apagar uma amostra e unir as duas vizinhas como se fossem contíguas.
- Criar manifesto executável que reproduza os relatórios com hashes e ambiente travado; armazenar pesos, logs, matriz de confusão, curva e config junto do resultado.

**Porta:** nenhum número de modelo sem versão/hash/licença, protocolo de split e suporte por grupo/classe.

### P1 — Benchmark de atividade, barato e leakage-safe

- Usar o split de autores quando existe; reportar também leave-one-animal-out (todos os folds), agrupamento por sessão e ao menos um teste por domínio (pasto vs instalação, ou dataset independente após harmonizar ontologia).
- Derivar janelas apenas depois do split por animal/sessão. Se houver segmentos no tempo, adicionar margem de exclusão pelo tamanho da janela/horizonte. Fazer scalers, balanceamento e tuning apenas em treino.
- Comparar majoritária, persistência/duração anterior, regressão logística/RF/ExtraTrees, mini-ROCKET+Ridge, TCN/1D-CNN ou GRU compacta. Usar GPU para séries temporais neurais se for mais rápido, não por requisito de qualidade.
- Métricas primárias: macro-F1 e balanced accuracy por animal/farm, recall e precisão por classe, matriz de confusão, duração/bout e calibração; intervalos por bootstrap de grupo (animal/sessão). Publicar média e intervalo entre animais, não tratar janelas como observações independentes.

**Porta:** melhorar baseline de persistência em animais/domínios não vistos; classes raras com suporte suficiente; sem regressão escondida em um animal/farm.

### P2 — Dados brasileiros para resultado produtivo e pastagem

- Auditar e baixar o dataset Mendeley de Nelore; ligar cada registro de peso/suplemento a animal, data, piquete, clima e satélite. Separar animais e blocos temporais/paddocks para calibrar e validar.
- Buscar registros de fazenda com massa/altura/qualidade de forragem, consumo/suplemento, peso/condição corporal, chuva, temperatura/umidade/radiação/vento e protocolo de manejo. Caso inexistam observações co-localizadas, planejar coleta; inferir causalidade de um painel satélite sozinho não é suficiente.
- Reproduzir previsão basal de ganho de peso/visita ao cocho antes de tentar simular mecanismo. Separar erro preditivo temporal de causalidade de intervenção.

### P3 — Núcleo mecanístico mínimo

- Escolher população-alvo e compor um modelo com unidades físicas (kg, kg MS/d, MJ/d, L, °C, ha, mm), fluxos auditáveis e limites de conservação.
- Implementar crescimento/ingestão com fonte adequada a bovino de corte e escopo Nelore; não adaptar automaticamente equações de vaca leiteira. Calibrar parâmetros com uma coorte e avaliar em fazenda/ano independente.
- Implementar crescimento de pastagem a partir de clima/solo/biomassa e remoção por animal; validar dinâmica do pasto sem bovino e com pastejo separadamente.
- Implementar balanço térmico/água com variáveis ambientais; testar consequências plausíveis em faixas de clima reservadas.

**Porta:** erro e viés reportados em medidas reais por animal e ao longo do tempo; balanço de massa/energia fechado; limites de aplicabilidade descritos. Se não há dados para fechar uma equação, marcar explicitamente como hipótese ou desligar o mecanismo.

### P4 — Movimento, grupos e política

- Ajustar emissão de trajetória e seleção de manchas com GPS/UWB, primeiro sem componente social; validar localização/tempo de permanência, distribuição de velocidade, horários e preferência espacial em sessões ou rebanhos não vistos.
- Adicionar rede de vizinhança apenas se o holdout mostrar melhoria identificável sobre recurso compartilhado e ciclo diário; testar ablação social.
- Ajustar duração/transição de atividades em séries completas; depois política behavior-cloning em ações observadas com advertência de confundimento.
- Só então RL sobre simulador validado. Múltiplos seeds de treino, famílias de clima/pasto, distribuição reservada, baselines fortes, testes de stress, restrições biológicas e comparação de trajetórias/estados físicos além do retorno. Manter um conjunto final selado por versão.

## 6. Limites e riscos para alegações futuras

- **Generalização:** uma vaca holdout não representa generalização por raça, sensor, fazenda, clima ou fase produtiva. Cross-validation por animal é necessário, teste por domínio é diferente e ambos são necessários.
- **Confundimento causal:** os conjuntos de atividade são observacionais. Correlação entre atividade, calor, localização ou visita não identifica efeito de fornecer água/suplemento, alterar piquete ou impor uma decisão.
- **Rótulo proxy:** feeder, collar, HMM e detector podem fornecer rótulos indiretos. Manter a origem do rótulo no esquema e não chamá-lo de ground truth humano.
- **Licença:** dado e código têm licenças separadas. MmCows tem card HF CC BY-NC-SA; CIRAD NC-SA; NASEM-Python e APSIM têm termos/partes distintos. Confirmar os arquivos originais antes de distribuir dados, derivados ou pesos em repositório público/produto.
- **Pseudorrealismo:** nomes como “health”, “heat load”, “favorite forage” ou “social stress” não tornam um estado real. Manter metadados `observed | calculated | learned | assumed`, unidade, fonte e incerteza por variável.

## Conclusão

**O que funciona:** pipeline de classificação de atividade previamente executado no Kaggle, com desempenho condicionado ao split documentado; artifacts do DQN são reproduzíveis em termos de código, config, seeds e saída; DQN vence sua heurística no ambiente de treino declarado; o novo pipeline de ingestão Kaggle valida hash, esquema e intervalos e produz splits reproduzíveis sem reconstruir variáveis ausentes. O novo núcleo Rust de água/forragem passa seus testes de contabilidade e roda via WASM; ainda não está calibrado com dados de campo nem conectado à política do rebanho.

**O que é artificial:** estados fisiológicos/ambientais do app TypeScript, taxas de recurso, movimento e efeitos sociais; o ambiente e reward do DQN; a tradução de ações entre DQN e TypeScript; personalidade/raça individuais. O núcleo novo contém uma equação publicada para ingestão hídrica, mas não simula hidratação nem foi validado no Campo Vivo. O CSV de cocho não contém cinco classes de atividade nem ground truth de comportamento.

**Próxima ordem:** (1) resolver proveniência/ontologia e lacunas temporais do dataset; (2) refazer benchmark de atividade com split multi-animal e domínio externo; (3) usar dados Nelore brasileiros e parâmetros/observações de pasto para calibrar o primeiro modelo fisiológico com unidades; (4) fechar balanços e validação externa do núcleo de crescimento/forragem/clima; (5) calibrar movimento com GPS e, depois, relações sociais; (6) retomar RL somente depois das transições e objetivos terem vínculo mensurável com animais e manejo.

### Primeiro experimento empírico integrado no Kaggle (2026-10-22)

O primeiro notebook foi executado em runtime Kaggle com T4 solicitada usando a versão 7 do dataset UFMS/Mendeley (26 animais, 391 observações animal-período). A tarefa foi prever ganho médio diário (ADG, kg/d), que é o target derivável do conjunto; isso não é previsão de peso final nem simulação fisiológica. Split temporal: períodos 1–9 treino, 10–11 validação, 12–15 teste futuro; split adicional por animal reservando IDs 6, 14, 16, 19 e 26. Features futuras e visitas ao cocho após o período-alvo foram excluídas. Nenhum animal/período foi selecionado com base no desempenho do teste.

No split temporal, HistGradientBoosting selecionado na validação teve MAE de teste 0,455 kg/d, RMSE 0,569 kg/d e R² −0,181; baseline pela média de treino teve MAE 0,511 kg/d. O ganho absoluto de MAE é modesto, R² negativo mostra que o modelo não explica bem a variação no futuro, e a validação de apenas dois períodos torna a seleção instável. ExtraTrees obteve MAE 0,406 kg/d no teste, mas esse número não pode ser usado para selecionar o modelo porque o teste é reservado. No holdout por animal, HGB MAE 0,302 vs média 0,363 kg/d; ExtraTrees 0,290, ainda sujeito à regra de não selecionar pelo teste. O Kaggle kernel concluiu no runtime T4 e os resultados foram reproduzidos localmente; os estimadores eram CPU sklearn, então o uso da GPU foi apenas de infraestrutura, não aceleração do treino. Resultado é baseline exploratório e não passa critério de aceitação, nem sustenta contrafactual ou causalidade. A amostra pequena, um único piquete e poucos blocos temporais limitam generalização. Manifesto, hash do CSV e versão da fonte, métricas, log e checkpoint estão registrados em `training/growth/runs/kaggle-v2/`.

### Núcleo científico compartilhado iniciado

O crate `engine/campo-core` implementa balanço diário simples para remoção de forragem, água no solo, escoamento e evapotranspiração, com unidades tipadas e déficits explícitos, além de contratos versionados de observação/ação individual. Foi acrescentada a equação publicada de ingestão diária de água do BR-CORTE 2023 (capítulo 14), preservada como estimativa de domínio tropical com flag quando os preditores saem das faixas de validação. A fonte reporta 14.774 observações diárias em validação independente, erro médio próximo de zero mas concordância 0,574; ela não representa necessidade hídrica, hidratação ou padrão de visitas. `cargo test`, build WASM e smoke test via Node passam. Um wrapper TypeScript verifica ABI e consegue chamar a equação no WASM; o pacote é construído junto com a aplicação. **A dinâmica ainda não está acoplada ao jogo/loop de simulação** porque as escalas ambientais atuais do frontend são arbitrárias e não devem ser silenciosamente convertidas para kg/ha ou mm. Não há equação de crescimento animal ou pasto calibrada neste crate e a equação de água ainda não foi validada contra dados independentes integrados pelo Campo Vivo. [BR-CORTE 2023, capítulo 14](https://brcorte.com.br/assets/book2023/en/14.pdf).

## Critérios de aceitação para a execução incremental

Estes são gates de engenharia definidos para que o trabalho seguinte tenha critérios verificáveis. Eles não representam limites veterinários universais. As métricas devem sempre comparar com os baselines abaixo e manter intervalos por animal/pen/fazenda.

1. **Reprodutibilidade:** cada experimento tem fonte/versão/licença/hash, esquema/unidades, split por grupo salvo, código/config, seeds, dependências fixas, log, checkpoint e relatório. Reexecutar a mesma configuração reproduz splits e métricas (diferença apenas numérica definida por backend, se GPU não for bitwise determinística).
2. **Motor compartilhado:** `cargo test` e build de release `wasm32-unknown-unknown` passam; uma execução de cenário via WASM e Rust nativo com mesmas entradas/seed produz a mesma saída dentro da tolerância especificada. Fisiologia/pasto usam unidades e fecham ledger de massa/água com erro ≤0,1% por passo. O frontend não mantém uma segunda implementação das equações.
3. **Peso/ganho Nelore:** holdout por animal e por tempo (sem linhas futuras no treino) contra baseline persistente e tendência média; alvo de implantação: MAE de peso pelo menos 10% abaixo do baseline mais forte, viés absoluto ≤5 kg em holdout e intervalo de predição com cobertura nominal reportada. Com apenas 26 animais/um piquete, passar este gate valida somente a coorte/dataset, não transferência para fazendas independentes.
4. **Água/sombra:** prever consumo diário/eventos em animal/dia agrupados sem particionar eventos de um mesmo animal/pen entre folds. Reportar erro em L/d, frequência e duração, e diferenças observadas entre grupos. Não usar os 19.388 eventos como 19.388 unidades independentes de tratamento: a intervenção foi disponibilizada a grupos em quatro piquetes. Validar causalidade/generalização exigirá replicação independente por piquete/fazenda.
5. **Classificação de atividade:** comparar majoritária, persistência e modelo; folds leave-one-animal-out e teste externo por fazenda/domínio quando ontologias coincidirem. Macro-F1 e balanced accuracy médios por animal devem superar persistência em ≥5 pontos percentuais, sem colapsar uma classe comum/raras; reportar intervalos e suporte. Se o conjunto não oferecer suporte independente por classe, restringir a ontologia em vez de forçar a aceitação.
6. **Política comportamental/RL:** só entra no runtime após motor mecanístico validado; comparar contra ao menos política fisiológica baseada em limiar e MPC/heurística forte. Avaliar ≥5 seeds de treino e ≥100 cenários selados, mostrar distribuição de retorno e de medidas físicas, superar baseline forte sem violar limites biológicos/de recursos e demonstrar trajetórias plausíveis por análise cega. Ganhar recompensa sintética não satisfaz este critério.
7. **Desempenho populacional:** benchmark headless e versão de hardware/browser/runtime fixada. Meta inicial: processar 10.000 animais por 144 passos (um dia a cada 10 minutos simulados), sem renderização, em ≤1 s no browser WASM e ≤250 ms nativo, p95 em 30 repetições após aquecimento. Medir memória e uso de CPU; interações devem ser locais/sparse para não criar custo O(N²). Se o benchmark real divergir, registrar baseline e mover o gate com evidência, sem apagar a regressão.

Os critérios 3–5 não podem ser declarados atingidos com o classificador atual, nem com um split aleatório de janelas. Para o critério 4, a análise deve preservar o tratamento e agrupamento espacial do estudo. Para todos, “passou” significa métrica, população, desenho e domínio explicitados, sem transformar uma base pequena em alegação de validade universal.

### Arquivos auditados

- [`src/simulation.ts`](../../src/simulation.ts)
- [`src/learned-policy.ts`](../../src/learned-policy.ts)
- [`training/train.py`](../../training/train.py)
- [`training/experiments/kaggle-cuda-run.json`](../../training/experiments/kaggle-cuda-run.json) e log
- [`training/policy/train_policy.py`](../../training/policy/train_policy.py)
- [`training/policy/evaluation-v2.json`](../../training/policy/evaluation-v2.json), log e pesos
- [`training/README.md`](../../training/README.md)
- [`training/policy/README.md`](../../training/policy/README.md)
- Pesquisas anteriores: [`behavior-models.md`](behavior-models.md), [`data-and-models.md`](data-and-models.md), [`landscape.md`](landscape.md)
