import { chooseBehavior, type AnimalAction, type AnimalTraits } from './behavior-policy';
import { chooseLearnedGameAction, type CowPolicyObservation } from './learned-policy';

export type Weather = 'sunny' | 'cloudy' | 'rain' | 'heat';
export type AnimalState = 'grazing' | 'drinking' | 'resting' | 'walking';
export interface Animal {
  id:string; name:string; breed:string; age:number; coat:string; x:number; y:number;
  hunger:number; thirst:number; stress:number; health:number; rumination:number;
  targetX:number; targetY:number; state:AnimalState; action:AnimalAction;
  actionHours:number; fatigue:number; favoritePatch:number; traits:AnimalTraits;
}
export interface Patch { x:number; y:number; grass:number; moisture:number; }
const clamp=(x:number,min=0,max=100)=>Math.max(min,Math.min(max,x));
const distance=(x:number,y:number,a:number,b:number)=>Math.hypot(x-a,y-b);

/**
 * Lightweight spatial agent simulation. State and utility values are explicit
 * game-model approximations; only future calibration against field data can
 * establish biological validity.
 */
export class FarmSimulation {
  animals:Animal[]=[
    this.makeAnimal('a1','Jurema','Nelore',4,'coat-white',.31,.38,{sociality:.54,heatTolerance:.88,exploration:.38}),
    this.makeAnimal('a2','Cacau','Angus',3,'coat-brown',.48,.33,{sociality:.68,heatTolerance:.48,exploration:.62}),
    this.makeAnimal('a3','Canela','Girolando',5,'coat-spotted',.43,.56,{sociality:.76,heatTolerance:.69,exploration:.44}),
    this.makeAnimal('a4','Trovão','Nelore',2,'coat-gray',.61,.47,{sociality:.48,heatTolerance:.91,exploration:.77}),
  ];
  patches:Patch[]=Array.from({length:32},(_,i)=>({x:.08+(i%8)*.1,y:.19+Math.floor(i/8)*.13,grass:52+((i*17)%46),moisture:30+((i*23)%45)}));
  grass=68; water=74; soil=42; hour=8; temperature=27; weather:Weather='sunny';
  paused=false; speed=1; stocking=4; lastEvent=''; rotation=0;
  private randomState=20261022;
  private random(){this.randomState^=this.randomState<<13;this.randomState^=this.randomState>>>17;this.randomState^=this.randomState<<5;return(this.randomState>>>0)/4294967296;}
  private makeAnimal(id:string,name:string,breed:string,age:number,coat:string,x:number,y:number,traits:AnimalTraits):Animal{
    return{id,name,breed,age,coat,x,y,hunger:20+(age*7)%35,thirst:17+(age*11)%30,stress:10+(age*9)%24,health:88+(age*3)%12,rumination:58+(age*6)%30,targetX:x,targetY:y,state:'grazing',action:'graze',actionHours:0,fatigue:22+(age*7)%25,favoritePatch:-1,traits};
  }
  feed(){for(const a of this.animals){a.hunger=clamp(a.hunger-26);a.rumination=clamp(a.rumination+10);a.stress=clamp(a.stress-4);}this.lastEvent='Suplementação reduziu a fome e favoreceu a ruminação do rebanho.';}
  irrigate(){if(this.water<12)return false;this.water-=12;this.soil=clamp(this.soil+26);for(const p of this.patches)p.moisture=clamp(p.moisture+18);this.lastEvent='Irrigação elevou a umidade do solo; o capim responde gradualmente.';return true;}
  irrigatePatch(id:string){const p=this.patches[Number(id.replace('p',''))];if(p&&this.water>=3){this.water-=3;p.moisture=clamp(p.moisture+32);this.soil=clamp(this.soil+4);this.lastEvent='Irrigação localizada aplicada ao canteiro selecionado.';}}
  rotate(){this.rotation++;for(const p of this.patches)p.grass=clamp(p.grass+8);for(const a of this.animals){a.stress=clamp(a.stress+3);a.targetX=.12+this.random()*.72;a.targetY=.22+this.random()*.5;a.action='followHerd';a.actionHours=0;}this.lastEvent='Rotação reduziu a pressão de pastejo e iniciou o descanso da área anterior.';}
  setWeather(weather:Weather){this.weather=weather;this.temperature=weather==='heat'?36:weather==='rain'?23:weather==='cloudy'?25:29;this.lastEvent=weather==='rain'?'Chuva adiciona umidade e favorece a recuperação do capim.':weather==='heat'?'Calor aumenta a carga térmica e a necessidade de água.':'O clima altera evaporação e conforto térmico.';}
  moveAnimal(id:string,animalId:string){const p=this.patches[Number(id.replace('p',''))],a=this.animals.find(v=>v.id===animalId);if(p&&a){a.targetX=p.x;a.targetY=p.y;a.action='graze';a.actionHours=0;}}

  step(hours:number){
    this.hour=(this.hour+hours)%24;
    const raining=this.weather==='rain',heat=Math.max(0,this.temperature-27);
    this.soil=clamp(this.soil+(raining?1.1:-.16*hours)-(this.weather==='heat'?.4*hours:0));
    this.water=clamp(this.water-(this.weather==='heat'?.12:.04)*hours);
    for(const p of this.patches){p.moisture=clamp(p.moisture+(raining?1.3:-.22*hours));p.grass=clamp(p.grass+(p.moisture/100)*.55*hours-(this.stocking/4)*.095*hours);}
    this.grass=this.patches.reduce((sum,p)=>sum+p.grass,0)/this.patches.length;
    const herdX=this.animals.reduce((s,a)=>s+a.x,0)/this.animals.length,herdY=this.animals.reduce((s,a)=>s+a.y,0)/this.animals.length;
    for(const a of this.animals){
      a.hunger=clamp(a.hunger+(.32+(this.stocking/4)*.08)*hours);
      a.thirst=clamp(a.thirst+(.37+heat*.04*(1-a.traits.heatTolerance))*hours);
      const heatLoad=Math.max(0,this.temperature-(27+a.traits.heatTolerance*8));
      a.stress=clamp(a.stress+(heatLoad*.11+(this.grass<25?.13:0)-.035)*hours);
      a.fatigue=clamp(a.fatigue+((a.state==='walking'?.6:-.15))*hours);
      const patch=this.nearestPatch(a.x,a.y),shade=this.nearestShade(a.x,a.y);
      const behaviorObservation={hunger:a.hunger,thirst:a.thirst,heatLoad,fatigue:a.fatigue,forage:patch.grass,waterAvailable:this.water>1,shadeAvailable:true,distanceToWater:distance(a.x,a.y,.84,.76),distanceToShade:distance(a.x,a.y,shade.x,shade.y),distanceToHerd:distance(a.x,a.y,herdX,herdY),favoriteForage:a.favoritePatch===this.patches.indexOf(patch)?.28:0,currentAction:a.action,actionHours:a.actionHours,daylight:this.daylight(),traits:a.traits};
      const learnedObservation:CowPolicyObservation=[a.hunger/100,a.thirst/100,a.fatigue/100,Math.min(1,heatLoad/10),patch.grass/100,this.water/100,Math.max(.08,1-distance(a.x,a.y,shade.x,shade.y)/.6),Math.max(0,1-distance(a.x,a.y,herdX,herdY)/.55),a.stress/100];
      // The learned game policy is a candidate trained only on authored synthetic dynamics.
      // Keep the explicit utility rule available as a transparent baseline.
      const baselineAction=chooseBehavior(behaviorObservation).action;
      let action:AnimalAction;
      try{action=chooseLearnedGameAction(learnedObservation);}catch{action=baselineAction;}
      if(action!==a.action){a.action=action;a.actionHours=0;}
      a.actionHours+=hours;
      let tx=a.x,ty=a.y;
      if(action==='drink'){tx=.84;ty=.76;}
      if(action==='graze'){
        const idx=this.patches.indexOf(patch);a.favoritePatch=idx;tx=patch.x;ty=patch.y;
      }
      if(action==='seekShade'){tx=shade.x;ty=shade.y;}
      if(action==='followHerd'){tx=herdX;ty=herdY;}
      a.targetX=tx;a.targetY=ty;
      const dist=distance(a.x,a.y,tx,ty),move=Math.min(dist,.085*hours);
      if(dist>.025){a.x+=(tx-a.x)/dist*move;a.y+=(ty-a.y)/dist*move;a.state='walking';}
      else if(action==='drink'&&this.water>0){a.state='drinking';a.thirst=clamp(a.thirst-2.1*hours);this.water=clamp(this.water-.18*hours);a.stress=clamp(a.stress-.04*hours);}
      else if(action==='graze'&&patch.grass>1){a.state='grazing';const intake=Math.min(patch.grass,.38*hours);patch.grass=clamp(patch.grass-intake);a.hunger=clamp(a.hunger-.95*hours);a.rumination=clamp(a.rumination+.46*hours);}
      else {a.state='resting';a.fatigue=clamp(a.fatigue-.7*hours);a.rumination=clamp(a.rumination+.18*hours);a.stress=clamp(a.stress-(action==='seekShade'?.3:.12)*hours);}
      a.health=clamp(a.health+((a.hunger<65&&a.thirst<65&&a.stress<60)?.025:-.055)*hours);
    }
  }
  private nearestPatch(x:number,y:number){return this.patches.reduce((best,p)=>{const score=p.grass*.015-distance(x,y,p.x,p.y);return score>(best.grass*.015-distance(x,y,best.x,best.y))?p:best;},this.patches[0]);}
  private nearestShade(x:number,y:number){const trees:[[number,number],[number,number],[number,number],[number,number],[number,number],[number,number]]=[[.12,.24],[.2,.13],[.77,.15],[.9,.27],[.1,.8],[.92,.57]];return trees.reduce((best,[tx,ty])=>distance(x,y,tx,ty)<distance(x,y,best.x,best.y)?{x:tx,y:ty}:best,{x:trees[0][0],y:trees[0][1]});}
  private daylight(){return Math.max(0,Math.min(1,Math.sin((this.hour-6)/12*Math.PI)));}
}
