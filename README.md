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
- `src/behavior-policy.ts`: baseline interpretável de escolha de atividade.
- `src/learned-policy.ts`: inferência em navegador da política DQN treinada em ambiente sintético Kaggle; usada como política de jogo, sem alegação de validade biológica.
- `public/assets/`: atlas de bovinos e ilustração de cenário.
- `docs/research/`: pesquisa de evidências, projetos, artigos, dados e limites de validade.

## Modelos e evidência

Os datasets públicos encontrados ensinam principalmente a **reconhecer atividade observada** (como pastejo ou ruminação). Um classificador desse tipo não descobre sozinho a intenção do animal nem o efeito causal de mudar água, sombra ou pastagem. O DQN integrado aprende apenas as regras do ambiente sintético implementado; suas observações de fadiga, sombra e vizinhança no jogo incluem aproximações. Nenhum desses modelos é um simulador zootécnico validado. Para avaliar decisões aprendidas biologicamente, precisamos de trajetórias individuais sincronizadas com estado ambiental e resultados de intervenções.

Consulte [`docs/research/behavior-models.md`](docs/research/behavior-models.md), [`docs/research/data-and-models.md`](docs/research/data-and-models.md) e [`docs/research/landscape.md`](docs/research/landscape.md) para fontes, licenças, limitações e protocolo de avaliação.
