# Pesquisa: base científica e arquitetura para um simulador de bovinos

**Escopo da pesquisa:** modelos científicos, código reutilizável, dados públicos e estratégia de aprendizado para uma pequena fazenda interativa no navegador. Pesquisa feita em outubro de 2026. As fontes abaixo priorizam artigos originais, repositórios dos autores, documentação oficial e repositórios de dados. Disponibilidade, licença e termos de uso devem ser verificados novamente antes de incorporar qualquer artefato.

## Síntese e decisão técnica

Não encontrei um simulador aberto único que entregue, ao mesmo tempo, bovinos individuais, ecologia de pastagem, fisiologia validada e apresentação de jogo em tempo real. A solução mais defensável é compor subsistemas com fidelidade declarada:

1. **Núcleo de simulação próprio, determinístico e independente do renderizador.** Estado de cada animal e célula de terreno é explícito; um relógio de passo fixo e semente registrada permite repetir cenários. Separar a simulação da camada visual permite executar no browser hoje e trocar Canvas/WebGL, WebGPU ou WebAssembly depois.
2. **Biologia por regras e equações documentadas primeiro.** Usar modelos publicados de exigências de nutrientes/água e fisiologia térmica como especificação e referência. Não importar silenciosamente um modelo agrícola completo: APSIM é valioso como referência e ferramenta offline, mas sua licença de uso geral limita usos comerciais e impõe obrigações a melhorias.
3. **Bovinos como agentes individuais, com heterogeneidade e interação social.** Cada indivíduo guarda peso, idade, sexo, condição, estado reprodutivo, sede, fome, calor, fadiga, rotina aprendida, memória espacial e relações sociais. A seleção de ação pode começar por utilidade/rule-based; políticas aprendidas substituem apenas essa camada depois de calibradas.
4. **Pasto, solo e água como campos espaciais que reagem ao manejo.** Chuva, irrigação, crescimento, pisoteio e pastejo atualizam biomassa, umidade do solo, lama e qualidade do alimento. O intervalo biológico/ambiental pode ser maior que o intervalo de animação.
5. **Aprendizado com alvos observáveis, não “IA” como atalho biológico.** Datasets públicos ensinam a classificar atividade a partir de acelerômetros ou vídeo. Eles não fornecem, por si só, uma política causal que permita prever o efeito de trocar água, pastagem, clima ou manejo. Começar com modelos compactos de sequência; usar GPU Kaggle para experimentos offline, não para executar a simulação do navegador.

O ganho de realismo virá de comparar saídas observáveis, quantificar erro e registrar a proveniência dos parâmetros. “Validado” abaixo significa validado para o contexto experimental indicado pela fonte, não prova de validade universal nem de validade clínica/produtiva para qualquer raça, região ou sistema.

## O que já tem evidência aproveitável

### Nutrição, consumo e crescimento

O modelo animal **GRAZPLAN** representa ingestão, produção e reprodução de ovinos e bovinos em sistemas de pastejo; a implementação **STOCK** do APSIM Next Generation encapsula esse modelo e contempla classes animais, genótipos, sexo, idade, gestação/lactação, crescimento, reprodução, mortalidade e ações de manejo. A documentação técnica descreve claramente o escopo, mas é um modelo de rebanho/classe, não um “cérebro” espacial por animal com movimento gráfico [artigo fundador de GRAZPLAN](https://doi.org/10.1016/S0308-521X(96)00045-5), [manual técnico do STOCK](https://apsimdev.apsim.info/ApsimX/Releases/2021.04.07.5628/Stock.pdf), [documentação APSIM](https://docs.apsim.info/docs/).

Para a primeira versão, fome e peso não devem ser variáveis independentes arbitrárias: usar necessidade de manutenção + atividade, ingestão de matéria seca limitada pela oferta e digestibilidade, e atualizar reservas/ganho de peso ao longo do tempo. Gestação e lactação elevam demanda e consumo de água. A edição NASEM para leite documenta equações e também marca explicitamente lacunas: por exemplo, não há dados suficientes para validar modificadores de DMI em condições térmicas fora da zona termoneutra; a predição de ingestão de água para bezerros jovens não está disponível. Portanto, lacunas devem virar faixas/hipóteses configuráveis, não certezas escondidas [NASEM, *Nutrient Requirements of Dairy Cattle*, capítulo de DMI](https://www.ncbi.nlm.nih.gov/books/NBK600610/), [capítulo de água](https://www.ncbi.nlm.nih.gov/books/NBK600589/).

Uma equação publicada no NASEM 2016 prediz consumo diário de água para bovinos de corte a partir do peso corporal encolhido e de CETI, índice que combina temperatura, umidade relativa, vento e horas de sol. O resumo de extensão que reproduz a equação informa R²=0,997 para o conjunto usado na publicação, mas também avisa que modelos podem não transferir para pastejo extensivo. Um estudo de bovinos Nelore em condições tropicais reforça que equações têm população/domínio definidos e compara NASEM com equações alternativas; não se deve tratar um único valor previsto como verdade para qualquer animal [equação NASEM transcrita e limitações](https://pubs.nmsu.edu/_b/B231/), [avaliação em Bos indicus tropical](https://pmc.ncbi.nlm.nih.gov/articles/PMC6396249/).

**Uso no jogo:** modelar água ingerida por oportunidade de beber, disponibilidade/qualidade, distância e competição; a equação fornece demanda média diária, não o instante de cada gole. Falta de água pode aumentar sede e reduzir alimentação/desempenho; para magnitude, usar curvas de resposta e incerteza até que sejam calibradas com dados apropriados.

### Pastagem, água do solo e manejo

APSIM é um sistema integrado de modelos de plantas, clima, solo, animais e manejo, com módulo de pastagem AgPasture e exemplos executáveis no repositório oficial. Pode servir como referência de processos, fonte de parâmetros e oráculo offline para cenários selecionados [código APSIM Next Generation](https://github.com/APSIMInitiative/ApsimX), [exemplo oficial AgPasture](https://github.com/APSIMInitiative/ApsimX/blob/master/Examples/AgPasture.apsimx), [visão e documentação oficial](https://docs.apsim.info/docs/).

Não é correto vender uma curva simplificada de “grama cresce após irrigar” como APSIM. Uma avaliação de APSIM tropical em sistema silvipastoril mostrou bom desempenho para crescimento quando a competição considerada era por radiação (R² 0,75–0,86), mas fortes limitações para crescimento sob competição abaixo do solo por água/nutrientes (NSE negativo em algumas configurações), apesar de o solo-água ter sido razoável. Esse resultado é uma demonstração concreta de que até modelos mecanísticos publicados têm validade condicionada ao ambiente [avaliação de APSIM para capim Piatã](https://doi.org/10.1016/j.agsy.2021.103302). Um modelo mecanístico diário simples de clima/solo/pasto também foi validado para efeitos relativos de clima; é uma opção didática para versão inicial, com menor escopo que APSIM [modelo simples de crescimento de pasto e validação](https://pmc.ncbi.nlm.nih.gov/articles/PMC8464747/).

Para o protótipo, uma grade de células pode manter **biomassa verde, biomassa seca, umidade do solo, qualidade/digestibilidade e dano/compactação**. A chuva e irrigação alteram umidade; temperatura, luz, solo e umidade dirigem o crescimento; pastejo remove biomassa disponível e pode reduzir a área foliar remanescente; pisoteio e saturação produzem lama/compactação; descanso permite rebrote. A versão inicial pode usar balanços e curvas locais de resposta calibráveis, mas deve ser rotulada como aproximação até ser comparada com séries de crescimento e água de solo. Clima e pasto usam passo diário/horário; animal usa passos menores; renderer interpola estados sem dirigir a física.

Há evidência observacional para decisões espaciais: em experimento de pasto, o rebanho não ocupava uniformemente toda a área; padrão intra-rebanho variava entre pastejo e descanso [Shiyomi 1995](https://doi.org/10.1303/aez.30.259). Um modelo individual de movimento baseado em GPS/IMU propôs reconhecer indivíduo como unidade central e ligar escolhas de alimentação à posição e estrutura do pasto [modelagem de movimento e comportamento bovino](https://doi.org/10.1016/j.ecolmodel.2009.04.047). Outro ABM de resíduo de milho modelou fome/hidratação, distribuição de forragem, água, suplementação, ciclo dia/noite e manejo espacial, com calibração a dados de experimento de três anos [registro dos autores na University of Illinois](https://experts.illinois.edu/en/publications/a-spatially-explicit-agent-based-model-for-simulating-movements-o/). É uma referência próxima para arquitetura de agentes, mas não deve ser confundida com código livre nem com fisiologia integral.

Distância/qualidade da água também importam na distribuição. Um experimento de bovinos em pastagem reporta desempenho melhor com bebedouro que com água de lagoa e cita 250 m como distância máxima recomendada em sistemas pasture-based; isso não é regra universal de movimento [estudo de fonte de água](https://doi.org/10.3389/fvets.2021.616904). Rotação de piquetes também muda velocidade, caminho e uso espacial, com efeitos diferentes em anos úmidos e secos [estudo ranch-scale de pastejo rotacionado](https://doi.org/10.1016/j.agee.2023.108521). Esses resultados apoiam uma ligação causal simples e visível: mover bebedouro, cerca ou sombra muda trajetos, concentração do pastejo, consumo de água e exposição, mas parâmetros espaciais devem permanecer dependentes do contexto.

### Calor, sombra e comportamento

THI é útil, porém omite radiação solar e movimento do ar. Estudos com vacas leiteiras e bovinos de corte a pasto observaram que maior sombra foi associada a menor resposta comportamental/fisiológica ao calor, incluindo frequência respiratória e permanência perto da água; o uso de sombra cresceu com carga térmica [experimento de sombra em vacas leiteiras](https://doi.org/10.3168/jds.2009-2416), [estudo de calor/sombra em leite e corte](https://doi.org/10.1017/S1751731115000804). Um estudo cruzado mais recente com vacas Holandesas mostrou que DHLI e THI têm desempenhos diferentes segundo sombra e variável comportamental; recomenda incorporar fatores animais e de manejo para tornar índices mais úteis [comparação DHLI/THI](https://doi.org/10.1007/s00484-021-02186-x).

**Implicação:** separar “índice meteorológico”, “carga térmica individual” e “ação observada”. Um protótipo pode calcular THI/HLI, elevar desconforto sob calor e aumentar busca por sombra/água, reduzindo atividade ou ingestão. Esses efeitos são uma aproximação, a menos que raça, pelagem, aclimatação, radiação, vento e disponibilidade de sombra sejam parametrizados e comparados contra uma coorte experimental definida. Valores de corte não devem ser universais.

### Coesão do grupo, aprendizagem social e memória espacial

Há suporte para representar coesão e sincronização como tendências sociais com variação individual, não como uma massa que anda em formação perfeita: em gado Highland, transferência entre grupos foi estudada por métricas de rede social [estudo de redes sociais](https://doi.org/10.3389/fvets.2019.00183). Experimento de busca alimentar indicou que novilhos ingênuos localizaram alimento com maior eficiência na presença de um indivíduo experiente em determinadas distribuições, consistente com influência social e memória espacial [busca por alimento e modelo social](https://era.dpi.qld.gov.au/id/eprint/9739/). Há um modelo simples de deslocamento “parar e mover” aplicado a trajetórias de bovinos em alta frequência [padrões espaço-temporais](https://doi.org/10.1038/srep31967).

O primeiro agente pode ter motivação individual (fome/sede/calor/fadiga), atração por recursos, aversão a distância/risco e alinhamento/cohesão social com limites suaves. Memória de locais de forragem/água pode ser uma pequena lista de utilidades decaídas no tempo. Isso é mais explicável que uma política caixa-preta e dá alvos para calibração. A função exata de utilidade e os pesos de socialidade são **hipóteses do simulador**, a menos que ajustados contra trilhas identificadas.

## Código e projetos próximos

| Recurso | O que entrega | Reutilização e limites |
|---|---|---|
| [APSIM Next Generation](https://github.com/APSIMInitiative/ApsimX) | Plataforma agrícola modular, pasto, solo, clima, manejo e STOCK/GRAZPLAN | Melhor referência de processos e datasets de validação. A licença geral atual prevê uso gratuito para atividades educacionais/públicas não comerciais; consulting, pesquisa comercial ou geração de receita requer licença especial. A licença também atribui IP de melhorias à iniciativa. Não incorporar código antes de revisão jurídica/licença compatível com o produto. [Licença oficial](https://github.com/APSIMInitiative/ApsimX/blob/master/LICENSE.md), [termos e registro](https://registration.apsim.info/). |
| [Livestock Interaction Simulator (LIS)](https://github.com/CiaranWang/LIS) | Simulador ABM C++ para comportamento e interações dirigidas em animais alojados | Código C++ e arquitetura de fenótipos/motivação social útil como referência. Foi desenvolvido para perus alojados, não bovinos a pasto; adaptar exige nova parametrização/validação. Confirmar licença do repositório antes de incorporar. |
| [ALMaSS](https://projects.au.dk/almass) | Plataforma aberta de simulação espacial ABM para animais, paisagens e manejo | Referência para separação entre paisagem, agentes individuais e processos sociais/ecológicos. Foco é biodiversidade/paisagem e outras espécies; não é um simulador de pecuária pronto. |
| [Cattle grazing distribution ABM](https://mountainscholar.org/handle/10217/208505) e [ABM de resíduo de milho](https://experts.illinois.edu/en/publications/a-spatially-explicit-agent-based-model-for-simulating-movements-o/) | Modelos espaciais individuais de pastejo e busca de recursos | Evidência de desenho de modelos/agentes e processos a incluir; artigo/tese não implica fonte pública/licença de código disponível. |

Decisão de implementação: **não usar APSIM em runtime do jogo**. Reproduzir, em módulos próprios e com parâmetros/documentação próprios, um recorte de processos que caiba no produto. Comparar lotes de cenários contra exemplos APSIM onde a licença permite uso como ferramenta externa; para qualquer redistribuição, derivação ou uso comercial, obter licença apropriada. Usar C++/WASM só se profiling mostrar que o motor de simulação é gargalo: renderização pixel art 2.5D e UI permanecem no navegador, enquanto dados e regras biológicas podem começar em TypeScript e mover para WASM atrás de interface estável.

## Dados públicos para treinamento e validação

**Nenhum dataset aberto observado cobre tudo:** identidade persistente + vídeo/GPS + ingestão + pasto/sensoriamento + clima + intervenções controladas. Os conjuntos abaixo são bons para tarefas específicas, com animais/raças, locais, dispositivos, rótulos e licenças diferentes.

| Dataset | Conteúdo e utilidade | Riscos/termos |
|---|---|---|
| [Precision Beef, Zenodo 4064802](https://zenodo.org/records/4064802) | 18 bovinos, três ensaios; acelerômetro de coleira 10 Hz e halter Rumiwatch com rótulos de comer, ruminar, outro. Ótimo primeiro benchmark multissensor para classificação de atividade. | 8,2 GB no total; muitos exemplos da mesma fazenda/dispositivo podem induzir vazamento se janelas aleatórias forem separadas. Fazer split por animal/ensaio; conferir Readme e licença antes de download/uso. |
| [Ito et al., Japanese Black Beef Cow Behavior, Zenodo 5849025](https://zenodo.org/records/5849025) | Acelerômetro triaxial a 25 Hz de seis vacas, 13 classes anotadas a partir de vídeo; inclui campo e curral. Útil para classificação de sequências e comparar domínio a pasto/curral. | Uma fazenda, dois dias; licença CC BY-NC-ND 4.0, que limita uso comercial e obras derivadas. Não utilizar como material de produto/treinamento comercial sem esclarecer direitos. |
| [CVB: Cattle Visual Behaviors](https://doi.org/10.25919/3g3t-p068), [paper](https://arxiv.org/abs/2305.16555) | 502 clipes de 15 s a campo, 11 comportamentos visualmente observáveis, anotações em AVA; serve para detector/ação visual. | Não garante ID longitudinal nem estados fisiológicos internos. Conferir licença e composição do split. |
| [Open pasture cow dataset (WASP Lab)](https://github.com/WASP-lab/db-cow-walking) | 10 vacas em pastejo aberto por aproximadamente cinco meses; classes walking, grazing e resting em sinais de colar IMU. Fonte de um paper de 2025; promissor para transferência para pasto. | Confirmar condições de acesso, licença, rótulos/sincronização e protocolo; não assume ingestão, saúde ou decisão causal. |
| [Versluijs et al., DataverseNO](https://doi.org/10.18710/ND4CLL) | Dados limpos e scripts de classificação de comportamento de bovinos em pastejo extensivo, com acelerometria/GPS virtual-fence; importante para deslocamento e avaliação em cenário real. | Dataverse lista ZIP ~736 MB; verificar licença e classes observadas. Separar validação por animal e cenário. |
| [Cattle GPS, locations/activity/temperature — Essex](https://researchdata.essex.ac.uk/215/) | GPS, atividade e temperatura, ligado a observações de pastejo; útil para calibrar atividade e associação entre posição/uso do ambiente. | Ler documentação ligada, cobertura, frequência, licença e unidade antes de usar. |
| [ActBeCalf, dataset de bezerros](https://research.wur.nl/en/datasets/accelerometer-based-multivariate-time-series-dataset-for-calf-beh/) | 30 bezerros Holstein-Friesian/Jersey monitorados por 13 semanas desde uma semana de idade; bom para desenvolvimento/comportamento de bezerros e atividade temporal. | Não transferir diretamente para adulto a pasto; separar idade, raça e curral. |
| [Bolus sensor acceleration, Strathclyde](https://doi.org/10.15129/5e1dfedc-87c7-4af0-bb20-3f3e1388d9d9) | 107 horas de aceleração de bolus ruminal com rótulos de ruminação/comer/outro derivados de coleira; CC BY 4.0. | Uma configuração e rótulos herdados de classificador/collar, não observação manual independente. |
| [CattleEyeView](https://arxiv.org/abs/2312.08764) | Dataset top-down multi-tarefa para detecção, tracking, segmentação, pose; apropriado para técnicas visuais vistas de cima. | Acesso/redistribuição pode ser controlado; não é base para comportamento causal. |

**Kaggle e Hugging Face:** busca direcionada encontrou alguns uploads comunitários (incluindo dataset de brincadeira de bezerros e um conjunto de atividade de cuidado de bovinos) e referências a “cow nose-ring dataset” no Kaggle, mas não um benchmark canônico/validado comparável em documentação/licença àqueles acima. Plataformas são canais de espelho/distribuição, não garantia de procedência, licença ou ausência de duplicação. Priorizar DOI e repositório original; tratar cópias Kaggle/HF como conveniência e validar checksum, versão, origem e licença. [Exemplo de publicação HF de bezerros](https://huggingface.co/datasets/Sonam5/Calf-Play-Behavior-Dataset), [exemplo multimodal HF com conteúdo de interação humano-bovino](https://huggingface.co/datasets/deepannotate-ai/cattle-farming-multimodal-v1), [menção de dataset Kaggle de nose-ring em paper](https://doi.org/10.3389/fanim.2026.1676504). Nenhum desses exemplos substitui observações ecológicas com bovinos identificados em pasto.

## Estratégia de modelos e uso de GPU

### Componentes de aprendizado recomendados

1. **Classificador de atividade:** acelerômetro → grazing/walking/resting/ruminating/drinking/other. Começar com Random Forest ou XGBoost sobre janelas de 1–5 s como baseline reproduzível; em seguida TCN/GRU pequena para sequência. Reportar macro-F1, matriz de confusão, calibração de probabilidade e métricas por animal, cenário e classe. Usar split por animal/fazenda, não split aleatório de janelas.
2. **Reconhecimento visual opcional:** detector+tracking para bovinos em câmera top-down (dataset CattleEyeView) e modelo temporal de ação CVB. O jogo já sabe IDs e estados; não precisa usar visão computacional para “enxergar” seus próprios sprites. A utilidade é avaliar animações/gestos ou treinar modelos separadamente, não mover a lógica de jogo para visão.
3. **Política de comportamento:** começar por utilidade explicável como referência. A partir de ações anotadas, pode-se aprender uma política por imitação/behavior cloning; depois avaliar aprendizado por reforço em cenários sintéticos controlados. Política escolhe entre ações válidas (pastar, beber, ruminar, descansar, caminhar, buscar sombra, seguir grupo); motor biológico conserva limites de demanda, tempo e energia.
4. **Modelo de pasto:** não usar rede neural para substituir dinâmica sem dados. Equações/balanços alimentam primeiro o ciclo; árvore/ML pode corrigir erro local de crescimento após haver séries de biomassa, precipitação e solo com splits temporais e por fazenda.

Aprender “o que o animal faz ao ver aceleração/vídeo” é tarefa supervisionada suportada por dados. Aprender “o que faria uma vaca se a fazenda mudasse X” é tarefa de política/causalidade e não está identificada por datasets passivos comuns. RL sobre mundos sintéticos aprende as regras do simulador e pode explorar bugs/recompensas; precisa de randomização de ambiente, restrições, testes fora de distribuição e comparação com comportamento medido. Não anunciar agentes como comportamentos científicos só por terem sido treinados em GPU.

Kaggle documenta notebooks com aceleradores NVIDIA e quota semanal variável; a própria documentação avisa que GPU acelera workloads compatíveis, não bibliotecas como pandas/scikit-learn sem backend específico [documentação Kaggle Notebooks](https://www.kaggle.com/docs/notebooks), [uso eficiente de GPU/quota](https://www.kaggle.com/docs/efficient-gpu-usage). Usar Kaggle para treinamento/reprodução offline de classificadores, varredura de parâmetros e exportar pesos/configuração/versionamento; conjunto de treino e notebook precisam registrar fonte, DOI, licença, transformações e semente. A simulação jogável deve rodar localmente no navegador com centenas de agentes, sem depender de Kaggle ou chamada de inferência remota. WebGPU pode ajudar em gráficos e computação em GPU em alguns navegadores, mas a compatibilidade ainda não é universal; manter fallback WebGL/CPU [MDN WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API). Neste jogo, modelos compactos ou uma política pequena em CPU devem ser suficientes; GPU local é recurso opcional, não premissa do produto.

## Proposta de arquitetura do produto

```text
Entradas do jogador / roteiro climático
           │ comandos versionados
           ▼
Simulação determinística (passo fixo, seed, relógio de jogo)
  ├─ clima e relógio sazonal
  ├─ grade de solo/pasto/água e manejo
  ├─ estado fisiológico por indivíduo
  ├─ decisão + memória + relações do agente
  ├─ movimento espacial / colisão / cercas
  └─ eventos, métricas e proveniência
           │ snapshot de renderização
           ▼
Renderer pixel art 2.5D + UI/inspector + controles de replay
```

O renderizador só lê snapshots; não calcula ingestão, sede ou regrowth. Registrar comandos e semente permite replay determinístico e comparar políticas com mesmos cenários. Decompor frequência: desenho a 60 FPS; locomoção/animação a passo fixo visual; decisões comportamentais a cada dezenas de segundos simulados; fisiologia e recursos em minutos/horas; crescimento de pasto/solo em escala horária/diária. Tais escalas iniciais são **decisões de engenharia**, não frequências biologicamente validadas.

### Estado mínimo por bovino

`id, raça/genótipo, sexo, nascimento/idade, peso, escore de condição, estado reprodutivo, reservas energéticas, fome/demanda de matéria seca, sede/demanda de água, carga térmica, fadiga, localização/velocidade, atividade atual, saúde/locomoção, preferência individual, memória espacial, parceiros sociais, histórico resumido de ingestão/atividade`.

### Eventos causais que a simulação deve expor

| Intervenção | Resposta mecânica inicial | Status epistemológico |
|---|---|---|
| Mais irrigação/chuva | solo fica mais úmido; crescimento aumenta dentro dos limites configurados; risco de solo saturado/lama | Mecanismo fundamentado, curva local ainda aproximada até validar espécie/solo/clima. |
| Menos biomassa / sobrepastejo | menor forragem acessível e possivelmente pior qualidade; o agente caminha mais/seleciona patch diferente; ingestão cai se oferta for limitante | Forrageamento/seleção possuem evidência; função exata local é aproximação calibrável. |
| Mover bebedouro | altera distância de viagem, concentração espacial e competição; muda oportunidade de beber e gasto locomotor | Efeito qualitativo suportado por estudos de distribuição e água; magnitude depende do terreno/rebanho. |
| Calor e umidade/radiação | aumenta carga térmica; cresce motivação para sombra/água; atividade e consumo podem cair | Respostas a calor/sombra observadas; converter em equação individual universal é extrapolação. |
| Mudar cerca/piquete | restringe grafo de navegação e área de pastejo; muda trilhas e pressão local | Manejo espacial é evidenciado em estudos de pastejo rotacionado; a política de navegação é nossa. |
| Bomba de água desligada | consumo de água indisponível, sede sobe; priorização de deslocamento para outro recurso; condições prolongadas comprometem estado | Consequência física é certa dentro do modelo; limiares de doença/dano precisam fonte específica e não devem ser inventados. |

## Plano de calibração e afirmações científicas

1. **Versão 0, sem ML na política:** determinismo, metabolismo simplificado, clima, água e pasto configuráveis; inspector mostra valores e causa da última decisão. Desenvolver cenários “sem intervenção” e “mudou um fator”.
2. **Validação por camada:** comparar consumo/água/peso contra equações/tabelas e limites publicados; comparar proporções diárias de atividade contra datasets com classes; comparar distribuição espacial com trajetórias GPS; comparar biomassa/solo contra experimentos de pastagem. Anotar escopo de raça, região, estação e sistema em cada gráfico.
3. **Ajuste e incerteza:** apresentar faixa ou nível de confiança quando parâmetros/transferência não têm suporte. Separar dados de calibração e teste por animal/fazenda/ano; publicar erros e cenários em que o modelo falha.
4. **Modelo aprendido, versionado:** executar baseline e política treinada nos mesmos cenários/seed. Só promover política se melhora aderência em dados retidos sem quebrar invariantes biológicos nem gerar estratégias artificiais.
5. **Ficha de evidência por parâmetro:** `parâmetro`, unidade, equação/fonte, população de origem, faixa publicada, valor escolhido, ajuste realizado, motivo e categoria (`validado no escopo`, `extrapolado`, `aproximação de jogo`).

Invariantes importantes: nenhum animal ganha massa sem energia ingerida; não bebe sem fonte disponível; atividade soma dentro do dia; ingestão não ultrapassa a oferta nem capacidade parametrizada; pasto consumido não rebrotará no mesmo passo sem processo definido; chuva/irrigação/manejo produzem eventos auditáveis; dados e decisões são reproduzíveis quando a semente e sequência de comandos são idênticas.

## Lista de fontes-chave

- [Freer, Moore & Donnelly (1997), GRAZPLAN animal biology model](https://doi.org/10.1016/S0308-521X(96)00045-5).
- [APSIM official docs](https://docs.apsim.info/docs/), [APSIM code](https://github.com/APSIMInitiative/ApsimX), [APSIM current license](https://github.com/APSIMInitiative/ApsimX/blob/master/LICENSE.md), [registration/licensing page](https://registration.apsim.info/).
- [NASEM 2021 Dairy Cattle, DMI](https://www.ncbi.nlm.nih.gov/books/NBK600610/) and [water](https://www.ncbi.nlm.nih.gov/books/NBK600589/).
- [NMSU, NASEM water-intake equation and caveats](https://pubs.nmsu.edu/_b/B231/); [Bos indicus tropical evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6396249/).
- [APSIM-Piatã pasture validation](https://doi.org/10.1016/j.agsy.2021.103302); [simple mechanistic pasture model](https://pmc.ncbi.nlm.nih.gov/articles/PMC8464747/).
- [Grazing cattle behavior ABM/trajectory modeling](https://doi.org/10.1016/j.ecolmodel.2009.04.047), [corn-residue spatial ABM](https://experts.illinois.edu/en/publications/a-spatially-explicit-agent-based-model-for-simulating-movements-o/), [herd spatial pattern](https://doi.org/10.1303/aez.30.259).
- [Shade and dairy behavior](https://doi.org/10.3168/jds.2009-2416), [heat index comparison](https://doi.org/10.1007/s00484-021-02186-x), [shade and beef/dairy cattle](https://doi.org/10.1017/S1751731115000804).
- [Precision Beef dataset](https://zenodo.org/records/4064802), [Japanese Black Cow behavior](https://zenodo.org/records/5849025), [CVB](https://doi.org/10.25919/3g3t-p068), [free-grazing cow IMU](https://github.com/WASP-lab/db-cow-walking), [DataverseNO cattle data](https://doi.org/10.18710/ND4CLL), [Essex GPS/activity/temperature](https://researchdata.essex.ac.uk/215/), [ActBeCalf](https://research.wur.nl/en/datasets/accelerometer-based-multivariate-time-series-dataset-for-calf-beh/), [Strathclyde bolus data](https://doi.org/10.15129/5e1dfedc-87c7-4af0-bb20-3f3e1388d9d9).
- [Kaggle official notebook docs](https://www.kaggle.com/docs/notebooks), [Kaggle GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage), [MDN WebGPU status](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API).
