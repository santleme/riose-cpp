# Campo Vivo

Pequena fazenda bovina em pixel art 2.5D, executada diretamente no navegador. A tela inteira é o mundo: sem painéis de gestão ou dashboards. Bovinos individuais percorrem o pasto e reagem a fome, sede, calor, sombra, descanso, água e forragem.

> **Estado científico:** a simulação atual é um protótipo experimental, não um modelo zootécnico validado. Os parâmetros de fome, sede, ingestão, termorregulação, pastagem e decisão são aproximações do jogo. Os sprites e o cenário são ilustrações geradas para este protótipo; não constituem referência anatômica validada.

## Executar

```sh
npm install
npm run dev
```

`npm run build` gera a versão estática em `dist/`.

## Explorar

- Arraste o mouse ou toque para deslocar a câmera.
- Use a roda do mouse ou `+` / `-` para aproximar e afastar.
- Use `W A S D` ou as setas para navegar; `0` restaura a câmera.
- Clique em um bovino para acompanhá-lo.

## Estrutura

- `src/world.ts`: canvas em tela cheia, projeção 2.5D, sprites, cenário, animações e câmera.
- `src/simulation.ts`: estado ecológico e agentes bovinos individuais, independente do DOM.
- `src/behavior-policy.ts`: política interpretável de escolha de atividade, mantida como baseline até que uma alternativa aprendida passe validação.
- `public/assets/`: atlas de bovinos e ilustração de cenário.
- `docs/research/`: pesquisa de evidências, projetos, artigos, dados e limites de validade.

## Modelos e evidência

Os datasets públicos encontrados ensinam principalmente a **reconhecer atividade observada** (como pastejo ou ruminação). Um classificador desse tipo não descobre sozinho a intenção do animal nem o efeito causal de mudar água, sombra ou pastagem. A política de atividade do jogo permanece explícita e experimental; resultados de classificadores serão documentados separadamente e não tratados como fisiologia. Para avaliar decisões aprendidas, precisamos de trajetórias individuais sincronizadas com estado ambiental e resultados de intervenções.

Consulte [`docs/research/behavior-models.md`](docs/research/behavior-models.md), [`docs/research/data-and-models.md`](docs/research/data-and-models.md) e [`docs/research/landscape.md`](docs/research/landscape.md) para fontes, licenças, limitações e protocolo de avaliação.
