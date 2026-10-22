# Campo Vivo

Simulador experimental de uma pequena fazenda de bovinos, executado no navegador. O projeto começa com uma cena 2.5D em pixel art e um ecossistema simples no qual ações de manejo alteram recursos e necessidades dos animais.

> **Estado científico:** a versão atual é um protótipo de interação, não um modelo zootécnico validado. Taxas de fome, sede, pastejo, umidade e conforto são aproximações heurísticas. A interface explicita esse status; valores do protótipo não devem orientar decisões reais de manejo.

## Executar

```sh
npm install
npm run dev
```

Para gerar a versão estática:

```sh
npm run build
```

## O que está implementado

- Mapa isométrico desenhado em canvas, com piquete, cercas, curral, açude, cocho, árvores e bovinos.
- Bovinos com identidade, raça ilustrativa, idade, fome, sede, estresse, saúde, ruminação, movimento e estado atual.
- Pastagem distribuída em canteiros com biomassa e umidade próprias; o pastejo consome biomassa local e o clima afeta água, solo e crescimento.
- Ações de ração, irrigação, rotação, clima, lotação, seleção e deslocamento de animais, pausa e velocidade.
- Observação contextual para expor consequências na própria interface.

## Arquitetura inicial

`src/simulation.ts` contém estado e atualização da simulação sem dependência do DOM. `src/world.ts` projeta o estado em canvas e recebe interação espacial. `src/main.ts` monta a interface e conecta comandos à simulação. Essa fronteira permite substituir gradualmente as regras heurísticas por processos calibrados ou componentes aprendidos sem atrelar a biologia ao renderizador.

## Próximos passos

1. Construir a matriz de evidência e inventário de datasets em [`docs/research/landscape.md`](docs/research/landscape.md).
2. Definir unidades, escalas de tempo, balanços de massa e proveniência por variável antes de calibrar.
3. Criar cenários reproduzíveis e dados sintéticos com rótulos de origem; reservar dados reais para validação externa.
4. Implementar modelos de forrageamento, ingestão e conforto térmico com equações/limites documentados.
5. Avaliar modelos aprendidos em GPU Kaggle apenas para componentes com dados e métricas adequados; comparar sempre com uma linha de base mecanística.

## Fontes e evidência

O levantamento técnico e científico em [`docs/research/landscape.md`](docs/research/landscape.md) registra fontes, licenças e limitações dos dados. Até que essa avaliação esteja concluída, os comportamentos do protótipo são explicitamente aproximações.
