export type Weather = 'sunny' | 'cloudy' | 'rain' | 'heat';
export interface Animal { id: string; name: string; breed: string; age: number; coat: string; x: number; y: number; hunger: number; thirst: number; stress: number; health: number; rumination: number; targetX: number; targetY: number; state: 'grazing'|'drinking'|'resting'|'walking'; }
export interface Patch { x: number; y: number; grass: number; moisture: number; }
const clamp = (n:number, min=0, max=100) => Math.max(min, Math.min(max,n));
export class FarmSimulation {
  animals: Animal[] = [
    this.makeAnimal('a1','Jurema','Nelore',4,'coat-white',.31,.38), this.makeAnimal('a2','Cacau','Angus',3,'coat-brown',.48,.33), this.makeAnimal('a3','Canela','Girolando',5,'coat-spotted',.43,.56), this.makeAnimal('a4','Trovão','Nelore',2,'coat-gray',.61,.47),
  ];
  patches: Patch[] = Array.from({length: 32}, (_, i) => ({x:.08+(i%8)*.1, y:.19+Math.floor(i/8)*.13, grass:52+((i*17)%46), moisture:30+((i*23)%45)}));
  grass = 68; water = 74; soil = 42; hour = 8; temperature = 27; weather: Weather = 'sunny'; paused = false; speed = 1; stocking = 4; lastEvent = ''; rotation = 0;
  private tick = 0; private randomState = 20261022;
  private random() { this.randomState ^= this.randomState << 13; this.randomState ^= this.randomState >>> 17; this.randomState ^= this.randomState << 5; return (this.randomState >>> 0) / 4294967296; }
  private makeAnimal(id:string,name:string,breed:string,age:number,coat:string,x:number,y:number):Animal { return {id,name,breed,age,coat,x,y,hunger:20+(age*7)%35,thirst:17+(age*11)%30,stress:10+(age*9)%24,health:88+(age*3)%12,rumination:58+(age*6)%30,targetX:x,targetY:y,state:'grazing'}; }
  feed() { for (const a of this.animals) { a.hunger=clamp(a.hunger-26); a.rumination=clamp(a.rumination+10); a.stress=clamp(a.stress-4); } this.lastEvent='Suplementação reduziu a fome e favoreceu a ruminação do rebanho.'; }
  irrigate() { if (this.water < 12) return false; this.water-=12; this.soil=clamp(this.soil+26); for(const p of this.patches) p.moisture=clamp(p.moisture+18); this.lastEvent='Irrigação elevou a umidade do solo; o capim responde gradualmente.'; return true; }
  irrigatePatch(id:string) { const p=this.patches[Number(id.replace('p',''))]; if (p && this.water>=3) { this.water-=3; p.moisture=clamp(p.moisture+32); this.soil=clamp(this.soil+4); this.lastEvent='Irrigação localizada aplicada ao canteiro selecionado.'; } }
  rotate() { this.rotation++; for(const p of this.patches) p.grass=clamp(p.grass+8); for(const a of this.animals) { a.stress=clamp(a.stress+3); a.x=.12+this.random()*.72; a.y=.22+this.random()*.5; a.targetX=a.x; a.targetY=a.y; } this.lastEvent='Rotação reduziu a pressão de pastejo e iniciou o descanso da área anterior.'; }
  setWeather(weather:Weather) { this.weather=weather; this.temperature=weather==='heat'?36:weather==='rain'?23:weather==='cloudy'?25:29; this.lastEvent=weather==='rain'?'Chuva adiciona umidade e favorece a recuperação do capim.':weather==='heat'?'Calor aumenta a carga térmica e a necessidade de água.':'O clima altera evaporação e conforto térmico.'; }
  moveAnimal(id:string, animalId:string) { const p=this.patches[Number(id.replace('p',''))]; if(p){const a=this.animals.find(v=>v.id===animalId); if(a){a.targetX=p.x;a.targetY=p.y;a.state='walking';}} }
  step(hours:number) {
    this.tick++; this.hour+=hours; const heat=this.temperature>30? (this.temperature-30)*.8:0;
    this.soil=clamp(this.soil+(this.weather==='rain'?1.1:-.16*hours)-(this.weather==='heat'?.4*hours:0));
    this.water=clamp(this.water-(this.weather==='heat'?.12:.04)*hours);
    for(const p of this.patches){p.moisture=clamp(p.moisture+(this.weather==='rain'?1.3:-.22*hours));p.grass=clamp(p.grass+(p.moisture/100)*.55*hours-(this.stocking/4)*.095*hours);}
    this.grass=this.patches.reduce((sum,p)=>sum+p.grass,0)/this.patches.length;
    for(const a of this.animals){
      a.hunger=clamp(a.hunger+(.32+(this.stocking/4)*.08)*hours); a.thirst=clamp(a.thirst+(.37+heat*.04)*hours); a.stress=clamp(a.stress+(heat*.11+(this.grass<25?.13:0)-.035)*hours); a.health=clamp(a.health+((a.hunger<65&&a.thirst<65&&a.stress<60)?.025:-.055)*hours); a.rumination=clamp(a.rumination+((a.state==='grazing'&&this.grass>25)?.3:-.12)*hours);
      const waterDistance=Math.hypot(.84-a.x,.76-a.y);
      if(a.thirst>68&&waterDistance>.075){a.targetX=.84;a.targetY=.76;}
      const dx=a.targetX-a.x,dy=a.targetY-a.y,dist=Math.hypot(dx,dy); if(dist>.015){a.x+=dx/dist*.012*hours;a.y+=dy/dist*.012*hours;a.state='walking';} else {
        const wantWater=a.thirst>68, wantFood=a.hunger>44;
        if(wantWater&&waterDistance<.1&&this.water>1){a.state='drinking';a.thirst=clamp(a.thirst-1.1*hours);this.water=clamp(this.water-.12*hours);}
        else if(wantFood&&this.grass>8){a.state='grazing';a.hunger=clamp(a.hunger-.8*hours);a.rumination=clamp(a.rumination+.24*hours);const nearest=this.patches.reduce((best,p)=>Math.hypot(p.x-a.x,p.y-a.y)<Math.hypot(best.x-a.x,best.y-a.y)?p:best,this.patches[0]);nearest.grass=clamp(nearest.grass-.22*hours);}
        else {a.state='resting';a.stress=clamp(a.stress-.08*hours);}
      }
    }
  }
}
