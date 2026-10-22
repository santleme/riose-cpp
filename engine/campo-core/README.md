# campo-core

Primeiro módulo do motor científico Campo Vivo, em Rust. A interface mantém quantidades com unidades explícitas e executa uma transição diária determinística de balanço para um talhão. É uma fundação contábil de estado, ainda não um modelo validado de solo, pastagem ou bovino.

## Interface

- `PatchState`: estoque de forragem (`kg DM/ha`), água no solo (`mm`) e capacidade hídrica (`mm`).
- `DailyForcing`: chuva, irrigação e evapotranspiração potencial (`mm`), crescimento líquido da forragem e remoção solicitada (`kg DM/ha`). Crescimento e remoção vêm do chamador; o módulo não os prevê. Remoção/desaparecimento não equivale a ingestão animal.
- `step_patch(state, forcing) -> Result<StepResult, InputError>`: retorna novo estado, evapotranspiração real, escoamento e déficits de ET/remoção.
- ABI C/WASM v1: `campo_step_patch_v1` e `campo_core_abi_version`; declaração em `include/campo_core.h`.
- Contrato individual versionado `ObservationV1 -> ActionV1`: identifica o bovino, recebe peso observado (kg), comportamento/bout atual, tempo de bout e memória explícita dos oito bouts completos mais recentes; ação declara tipo, destino opcional e duração pretendida. A validação checa versão, ordenação temporal, duração e destino. Isso é somente um formato de entrada/saída para um futuro modelo; não há função que escolha ação.

Balanço hídrico: `S1 = min(capacidade, S0 + chuva + irrigação) - min(ETpot, min(capacidade, S0 + chuva + irrigação))`; o excedente sobre capacidade é reportado como runoff e ET não atendida como déficit. Balanço de forragem: `B1 = B0 + crescimento - remoção_realizada`, com remoção limitada ao estoque disponível e demanda excedente reportada.

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
2. O contrato de agente não é um cérebro nem codifica preferência, percepção, ingestão ou aprendizado. Sem crescimento animal, balanço energético, clima ou crescimento vegetal previsto. Nada aqui foi treinado ou validado contra dados reais.
3. `field capacity` é parâmetro fornecido; não representa água disponível por camada, infiltração, drenagem, escoamento por relevo ou estresse vegetal. O bucket é uma simplificação de ledger, não FAO-56/APSIM.
4. A próxima evolução científica deve introduzir cada processo com unidade, fonte, parâmetro calibrável, validação por estudo e teste de balanço antes de expor comportamento aprendido.
