import './style.css';
import { FarmSimulation, type Animal, type Weather } from './simulation';
import { FarmRenderer } from './world';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <main class="shell">
    <header class="topbar">
      <a class="brand" href="#" aria-label="Campo Vivo início"><span class="brand-mark">cv</span><span>campo<span class="brand-light">vivo</span><small>FAZENDA EXPERIMENTAL</small></span></a>
      <div class="top-meta"><span class="live-dot"></span><span>SIMULAÇÃO LOCAL</span><span class="meta-divider"></span><span class="save-status">● salvo agora</span></div>
      <button class="icon-button" aria-label="Configurações">☼</button>
    </header>
    <section class="dashboard">
      <aside class="sidebar">
        <div class="farm-title"><span class="eyebrow">PROPRIEDADE · 01</span><h1>Fazenda<br><em>Ipê Roxo</em></h1><div class="location">⌖ &nbsp; Cerrado · Goiás, BR</div></div>
        <div class="season-card"><div class="season-icon">☀</div><div><span class="eyebrow">ESTAÇÃO SECA</span><strong>Dia <span id="day-number">01</span> <span class="season-year">· Ano 1</span></strong></div><span class="weather-pill" id="weather-pill">☀ 27°</span></div>
        <div class="section-label"><span>REBANHO</span><span class="count-pill" id="herd-count">04</span></div>
        <div class="herd-list" id="herd-list"></div>
        <div class="resource-block">
          <div class="section-label"><span>RECURSOS DA FAZENDA</span><span class="tiny-info">i</span></div>
          <div class="resource-row"><span class="resource-icon grass-icon">▧</span><div class="resource-copy"><div><span>Pastagem</span><b id="grass-value">68%</b></div><div class="meter"><i id="grass-meter" style="width:68%"></i></div></div></div>
          <div class="resource-row"><span class="resource-icon water-icon">◉</span><div class="resource-copy"><div><span>Reservatório</span><b id="water-value">74%</b></div><div class="meter blue"><i id="water-meter" style="width:74%"></i></div></div></div>
          <div class="resource-row"><span class="resource-icon soil-icon">⌁</span><div class="resource-copy"><div><span>Umidade do solo</span><b id="soil-value">42%</b></div><div class="meter amber"><i id="soil-meter" style="width:42%"></i></div></div></div>
        </div>
        <div class="sidebar-foot"><span class="foot-leaf">✳</span><span>Ecossistema em equilíbrio</span><span class="status-led"></span></div>
      </aside>
      <section class="world-column">
        <div class="world-toolbar"><div><span class="eyebrow">VISÃO GERAL</span><h2>O campo, agora</h2></div><div class="world-actions"><button class="tool-button active" data-tool="inspect" title="Inspecionar">⌖</button><button class="tool-button" data-tool="move" title="Mover um bovino">↗</button><button class="tool-button" data-tool="water" title="Irrigar um canteiro">⌁</button><span class="toolbar-divider"></span><button class="tool-button" id="zoom-out" title="Diminuir zoom">−</button><span class="zoom-level">100%</span><button class="tool-button" id="zoom-in" title="Aumentar zoom">＋</button></div></div>
        <div class="world-frame"><canvas id="farm-canvas" aria-label="Mapa interativo da fazenda"></canvas><div class="map-label label-pasture"><span></span> PIQUETE 01</div><div class="map-label label-water">AÇUDE</div><div class="map-label label-barn">CURRAL</div><div class="weather-overlay" id="weather-overlay"></div><div class="world-hint" id="world-hint">↖ Selecione um bovino para acompanhar</div><div class="canvas-controls"><button id="toggle-grid" title="Alternar grade">▦</button><button id="center-map" title="Centralizar mapa">⌖</button></div><div class="map-scale"><span></span><small>10 m</small></div></div>
        <div class="world-bottom"><div class="legend"><span><i class="legend-dot cow-dot"></i>Bovinos <b id="legend-cows">4</b></span><span><i class="legend-dot grass-dot"></i>Pastagem</span><span><i class="legend-dot water-dot"></i>Água</span></div><div class="sim-clock"><span class="clock-dot"></span><span id="sim-time">08:00</span><span class="clock-sep">·</span><span id="sim-speed">1×</span></div></div>
      </section>
      <aside class="control-panel">
        <div class="panel-head"><div><span class="eyebrow">CENTRO DE MANEJO</span><h2>Ações da fazenda</h2></div><span class="panel-more">···</span></div>
        <div class="action-group"><span class="group-label">NUTRIÇÃO & ÁGUA</span><button class="action-card" id="feed-action"><span class="action-symbol feed-symbol">⌁</span><span class="action-copy"><strong>Distribuir ração</strong><small>Suplemento no cocho</small></span><span class="action-arrow">↗</span></button><button class="action-card" id="irrigate-action"><span class="action-symbol irrigate-symbol">⌇</span><span class="action-copy"><strong>Irrigar pasto</strong><small>Usa 12% do reservatório</small></span><span class="action-arrow">↗</span></button></div>
        <div class="action-group weather-group"><span class="group-label">CLIMA & AMBIENTE</span><div class="weather-control"><div class="weather-current"><span class="sun-symbol" id="weather-icon">☀</span><div><strong id="weather-name">Sol entre nuvens</strong><small id="weather-effect">Evaporação moderada</small></div></div><div class="weather-options"><button class="weather-option selected" data-weather="sunny" title="Sol">☀</button><button class="weather-option" data-weather="cloudy" title="Nublado">☁</button><button class="weather-option" data-weather="rain" title="Chuva">☂</button><button class="weather-option" data-weather="heat" title="Calor intenso">♨</button></div></div></div>
        <div class="action-group management-group"><span class="group-label">MANEJO</span><button class="action-card" id="rotate-action"><span class="action-symbol rotate-symbol">⟳</span><span class="action-copy"><strong>Rotacionar piquete</strong><small>Recupera a pastagem atual</small></span><span class="action-arrow">↗</span></button><div class="range-row"><div><span>Lotação do pasto</span><b id="stocking-value">4 animais</b></div><input id="stocking" type="range" min="2" max="8" value="4"/><div class="range-labels"><span>Baixa</span><span>Alta</span></div></div></div>
        <div class="insight-card"><span class="insight-spark">✳</span><div><span class="eyebrow">OBSERVAÇÃO DO CAMPO</span><p id="insight-text">Capim-marandu em boas condições. A umidade começa a cair com a estação seca.</p><small id="insight-source">DADOS DA SIMULAÇÃO · APROXIMAÇÃO</small></div></div>
        <div class="panel-footer"><button class="play-button" id="play-button"><span id="play-icon">Ⅱ</span><span id="play-label">Pausar simulação</span></button><button class="speed-button" id="speed-button">1× <span>⌄</span></button></div>
      </aside>
    </section>
    <div class="toast" id="toast" role="status"></div>
    <footer class="app-footer"><span>CAMPO VIVO <i>·</i> SIMULADOR ECOLÓGICO v0.1</span><span>MODELO BIOLÓGICO <b>EXPERIMENTAL</b></span><span>FEITO PARA APRENDER, COM A TERRA</span></footer>
  </main>`;

const canvas = document.querySelector<HTMLCanvasElement>('#farm-canvas')!;
const sim = new FarmSimulation();
const renderer = new FarmRenderer(canvas, sim);
let selectedId = sim.animals[0].id;
let speed = 1;
let toastTimer = 0;

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const toast = (message: string) => { const el = $('#toast'); el.textContent = message; el.classList.add('show'); clearTimeout(toastTimer); toastTimer = window.setTimeout(() => el.classList.remove('show'), 2400); };
const pct = (v: number) => `${Math.round(v)}%`;
function renderUI() {
  const herd = $('#herd-list');
  herd.innerHTML = sim.animals.map((a) => `<button class="herd-animal ${a.id === selectedId ? 'selected' : ''}" data-animal="${a.id}"><span class="animal-avatar ${a.coat}"><span></span></span><span class="animal-info"><strong>${a.name}</strong><small>${a.breed} · ${a.age} anos</small></span><span class="animal-health"><i class="health-led ${a.health < 45 ? 'warning' : ''}"></i>${a.health < 45 ? 'Atenção' : 'Bem'}</span></button>`).join('');
  herd.querySelectorAll<HTMLButtonElement>('[data-animal]').forEach((button) => button.onclick = () => { selectedId = button.dataset.animal!; renderer.selectedId = selectedId; renderUI(); toast(`${sim.animals.find(a => a.id === selectedId)?.name} selecionada no piquete.`); });
  $('#herd-count').textContent = String(sim.animals.length).padStart(2, '0'); $('#legend-cows').textContent = String(sim.animals.length);
  $('#grass-value').textContent = pct(sim.grass); $('#grass-meter').style.width = pct(sim.grass); $('#water-value').textContent = pct(sim.water); $('#water-meter').style.width = pct(sim.water); $('#soil-value').textContent = pct(sim.soil); $('#soil-meter').style.width = pct(sim.soil);
  $('#day-number').textContent = String(Math.floor(sim.hour / 24) + 1).padStart(2, '0'); $('#sim-time').textContent = `${String(Math.floor(sim.hour % 24)).padStart(2, '0')}:${String(Math.floor((sim.hour % 1) * 60 / 10) * 10).padStart(2, '0')}`;
  $('#weather-pill').textContent = `${sim.weather === 'rain' ? '☂' : sim.weather === 'cloudy' ? '☁' : '☀'} ${Math.round(sim.temperature)}°`;
  $('#weather-icon').textContent = sim.weather === 'rain' ? '☂' : sim.weather === 'cloudy' ? '☁' : sim.weather === 'heat' ? '♨' : '☀';
  $('#weather-name').textContent = ({sunny:'Sol entre nuvens',cloudy:'Céu nublado',rain:'Chuva leve',heat:'Onda de calor'})[sim.weather];
  $('#weather-effect').textContent = ({sunny:'Evaporação moderada',cloudy:'Menos evaporação',rain:'Solo recebe água',heat:'Estresse térmico elevado'})[sim.weather];
  const notes = sim.lastEvent || (sim.soil < 25 ? 'Solo seco: o capim cresce mais devagar. Irrigue para recuperar a umidade.' : sim.grass < 35 ? 'Pouco capim disponível. Considere rotacionar o piquete ou suplementar o rebanho.' : sim.weather === 'heat' ? 'Calor intenso aumenta a necessidade de água e o estresse térmico dos animais.' : 'Capim-marandu em boas condições. A umidade começa a cair com a estação seca.');
  $('#insight-text').textContent = notes;
}
renderUI();

$('#feed-action').onclick = () => { sim.feed(); toast('Suplemento distribuído: saciedade e ruminação foram atualizadas.'); renderUI(); };
$('#irrigate-action').onclick = () => { if (sim.irrigate()) toast('Pasto irrigado. A umidade do solo e o crescimento do capim aumentaram.'); else toast('Reservatório baixo: não há água suficiente para irrigar.'); renderUI(); };
$('#rotate-action').onclick = () => { sim.rotate(); toast('Rebanho movido ao piquete em descanso. A área anterior começa a se recuperar.'); renderUI(); };
document.querySelectorAll<HTMLButtonElement>('[data-weather]').forEach((button) => button.onclick = () => { sim.setWeather(button.dataset.weather as Weather); document.querySelectorAll('.weather-option').forEach((b) => b.classList.remove('selected')); button.classList.add('selected'); renderer.weather = sim.weather; toast(`Clima alterado: ${$('#weather-name').textContent}. Efeitos aplicados ao campo.`); renderUI(); });
document.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach((button) => button.onclick = () => { document.querySelectorAll('.tool-button[data-tool]').forEach(b => b.classList.remove('active')); button.classList.add('active'); renderer.tool = button.dataset.tool as 'inspect'|'move'|'water'; $('#world-hint').textContent = renderer.tool === 'move' ? '↖ Clique no campo para mover o animal selecionado' : renderer.tool === 'water' ? '↖ Clique num canteiro para irrigar' : '↖ Selecione um bovino para acompanhar'; });
$('#stocking').oninput = (e) => { sim.stocking = Number((e.target as HTMLInputElement).value); $('#stocking-value').textContent = `${sim.stocking} animais`; };
$('#play-button').onclick = () => { sim.paused = !sim.paused; $('#play-icon').textContent = sim.paused ? '▶' : 'Ⅱ'; $('#play-label').textContent = sim.paused ? 'Retomar simulação' : 'Pausar simulação'; };
$('#speed-button').onclick = () => { speed = speed === 1 ? 2 : speed === 2 ? 4 : 1; sim.speed = speed; $('#speed-button').innerHTML = `${speed}× <span>⌄</span>`; $('#sim-speed').textContent = `${speed}×`; };
$('#zoom-in').onclick = () => renderer.zoom = Math.min(1.35, renderer.zoom + .1); $('#zoom-out').onclick = () => renderer.zoom = Math.max(.8, renderer.zoom - .1); $('#toggle-grid').onclick = () => { renderer.showGrid = !renderer.showGrid; $('#toggle-grid').classList.toggle('on', renderer.showGrid); }; $('#center-map').onclick = () => renderer.center();
canvas.addEventListener('click', (e) => { const r = canvas.getBoundingClientRect(); const hit = renderer.handleClick((e.clientX-r.left)*canvas.width/r.width, (e.clientY-r.top)*canvas.height/r.height); if (hit) { if (renderer.tool === 'inspect') { selectedId = hit; renderUI(); } else if (renderer.tool === 'move') { sim.moveAnimal(hit, selectedId); toast(`${sim.animals.find(a => a.id === selectedId)?.name} mudou de posição.`); } else { sim.irrigatePatch(hit); toast('Canteiro irrigado: o capim local começa a se recuperar.'); } renderUI(); } });
let last = performance.now(); let accumulator = 0;
function frame(now: number) { const dt = Math.min(.1, (now-last)/1000); last = now; accumulator += dt; if (accumulator > .45) { accumulator = 0; if (!sim.paused) { sim.step(.25 * sim.speed); renderUI(); } } renderer.draw(now); requestAnimationFrame(frame); }
requestAnimationFrame(frame);
