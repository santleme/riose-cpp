# campo-core

Primeiro módulo do motor científico Campo Vivo, em Rust. A interface mantém quantidades com unidades explícitas e executa uma transição diária determinística de balanço para um talhão. Também contém uma estimativa empírica publicada de ingestão diária de água de bovinos de corte em clima tropical. Ainda não é um modelo validado de solo, pastagem ou comportamento individual.

## Interface

- `PatchState`: estoque de forragem (`kg DM/ha`), água no solo (`mm`) e capacidade hídrica (`mm`).
- `DailyForcing`: chuva, irrigação e evapotranspiração potencial (`mm`), crescimento líquido da forragem e remoção solicitada (`kg DM/ha`). Crescimento e remoção vêm do chamador; o módulo não os prevê. Remoção/desaparecimento não equivale a ingestão animal.
- `step_patch(state, forcing) -> Result<StepResult, InputError>`: retorna novo estado, evapotranspiração real, escoamento e déficits de ET/remoção.
- `estimate_water_intake_brcorte_2023_v1(...)`: calcula a equação publicada no BR-CORTE 2023, capítulo 14: `WI = 9.449 + 0.190 × BW^0.75 + 0.271 × Tmax − 0.259 × RH + 0.489 × DMI`. Entrada: peso vivo (kg), DMI (kg MS/d), temperatura máxima diária (°C), umidade relativa (%); saída: L/d assumindo densidade da água ~1 kg/L. A saída traz `within_validation_domain`; fora das faixas publicadas, o valor é explicitamente extrapolação.
- ABI C/WASM v1: `campo_step_patch_v1` e `campo_core_abi_version`; declaração em `include/campo_core.h`.
- ABI C/WASM para a equação: `campo_estimate_water_intake_brcorte_2023_v1`.
- Contrato individual versionado `ObservationV1 -> ActionV1`: identifica o bovino, recebe peso observado (kg), comportamento/bout atual, tempo de bout e memória explícita dos oito bouts completos mais recentes; ação declara tipo, destino opcional e duração pretendida. A validação checa versão, ordenação temporal, duração e destino. Isso é somente um formato de entrada/saída para um futuro modelo; não há função que escolha ação.

Balanço hídrico: `S1 = min(capacidade, S0 + chuva + irrigação) - min(ETpot, min(capacidade, S0 + chuva + irrigação))`; o excedente sobre capacidade é reportado como runoff e ET não atendida como déficit. Balanço de forragem: `B1 = B0 + crescimento - remoção_realizada`, com remoção limitada ao estoque disponível e demanda excedente reportada.

## Escopo científico da estimativa de água

A equação BR-CORTE é uma estimativa empírica de **ingestão diária de água** e não uma exigência mínima, balanço hídrico do organismo, desidratação ou previsão do horário de beber. O capítulo reporta validação independente com 14.774 observações diárias de cinco experimentos. Relata intercepto/slope sem diferença estatística de 0/1 e média estimada de 16,3 contra observada de 16,1 L/d, porém coeficiente de concordância de 0,574 e faixa individual observada até 49,9 L/d (máximo previsto 32,3 L/d). Assim, bom ajuste médio não implica precisão no consumo de cada animal/dia. As faixas usadas para sinalizar extrapolação são BW^0.75 46,1–119,4 kg^0.75; Tmax 16,2–36,6 °C; RH 42,5–95,3%; DMI 0,3–15,7 kg/d, conforme tabelas do capítulo. A equação deriva do trabalho de Zanetti et al. e não substitui validação numa coorte independente do Campo Vivo.

Fontes primárias: [BR-CORTE 2023, capítulo 14 (PDF)](https://brcorte.com.br/assets/book2023/en/14.pdf); [BR-CORTE 2023, capítulo 2 (DMI)](https://brcorte.com.br/assets/book2023/en/02.pdf). A implementação preserva a fórmula, a unidade e a indicação de domínio de validade; os coeficientes não foram ajustados ao conjunto de sombra Mendeley nem a dados sintéticos.

O smoke test também reproduz o exemplo trabalhado no capítulo 14: BW=520 kg, DMI=10,62 kg MS/d, Tmax=29 °C e RH=79% resultam em 22,73 L/d (22,729971 L/d antes do arredondamento). Isso confere a transcrição da equação, não valida o uso fora da população/modelo publicado.

Os cálculos são determinísticos para entradas iguais no mesmo alvo. IEEE-754 não promete resultados bit a bit idênticos entre arquiteturas/compiladores distintos. Não há RNG nem parâmetros ajustados com dados.

## Build local

```sh
cargo test --manifest-path engine/campo-core/Cargo.toml
cargo build --release --manifest-path engine/campo-core/Cargo.toml --target wasm32-unknown-unknown
node engine/campo-core/examples/wasm-smoke.mjs
```

O artefato WASM fica em `engine/campo-core/target/wasm32-unknown-unknown/release/campo_core.wasm`. O crate ainda não está conectado ao loop visual/browser; o chamador deve fornecer forçantes diárias e carregar/instanciar os exports da ABI. Um adapter JS/WASM e compatibilidade de unidades são o próximo seam de integração.

## Decisões e limites

1. Sem constantes agronômicas inventadas: a primeira versão só fecha estoques/fluxos com os valores fornecidos.
2. O contrato de agente não é um cérebro nem codifica preferência, percepção, ingestão de matéria seca ou aprendizado. A equação BR-CORTE estima ingestão observada de água, não necessidade fisiológica, estado de hidratação ou comportamento. Sem crescimento animal, balanço energético, clima ou crescimento vegetal previsto. A fórmula foi publicada/validada na fonte, mas não foi recalibrada ou validada de forma independente contra os dados reais do Campo Vivo.
3. `field capacity` é parâmetro fornecido; não representa água disponível por camada, infiltração, drenagem, escoamento por relevo ou estresse vegetal. O bucket é uma simplificação de ledger, não FAO-56/APSIM.
4. A próxima evolução científica deve introduzir cada processo com unidade, fonte, parâmetro calibrável, validação por estudo e teste de balanço antes de expor comportamento aprendido.
