/**
 * Interpretable behavior policy used until a learned policy passes validation.
 * Utilities are game-model parameters, not published physiological constants.
 */
export type AnimalAction='graze'|'drink'|'seekShade'|'followHerd'|'rest';
export interface AnimalTraits { sociality:number; heatTolerance:number; exploration:number; }
export interface BehaviorObservation {
  hunger:number; thirst:number; heatLoad:number; fatigue:number;
  forage:number; waterAvailable:boolean; shadeAvailable:boolean;
  distanceToWater:number; distanceToShade:number; distanceToHerd:number;
  favoriteForage:number; currentAction:AnimalAction; actionHours:number;
  daylight:number; traits:AnimalTraits;
}
export interface BehaviorDecision { action:AnimalAction; utility:number; }

const clamp=(x:number,min=0,max=1)=>Math.max(min,Math.min(max,x));
const smooth=(x:number,edge0:number,edge1:number)=>{const t=clamp((x-edge0)/(edge1-edge0));return t*t*(3-2*t);};
const duration:Record<AnimalAction,number>={graze:.35,drink:.08,seekShade:.3,followHerd:.18,rest:.45};

export function chooseBehavior(o:BehaviorObservation):BehaviorDecision{
  const h=clamp(o.hunger/100),th=clamp(o.thirst/100),heat=clamp(o.heatLoad/18),fatigue=clamp(o.fatigue/100);
  const separated=smooth(o.distanceToHerd,.06,.3);
  const closeToWater=1-smooth(o.distanceToWater,.02,.5);
  const closeToShade=1-smooth(o.distanceToShade,.02,.32);
  const memory=clamp(o.favoriteForage);
  const scores:Record<AnimalAction,number>={
    drink:o.waterAvailable?(th*th*1.12 + closeToWater*.06):0,
    graze:o.forage>4?(h*(.56+o.forage/190)+memory*.08+o.traits.exploration*.015):0,
    seekShade:o.shadeAvailable?(heat*(1-o.traits.heatTolerance)*.8+closeToShade*.04):0,
    followHerd:separated*o.traits.sociality*.32,
    rest:fatigue*.48+(1-o.daylight)*.11+(1-h)*.08,
  };
  // Keep an activity bout coherent. A strong unmet need can still interrupt it.
  const current=o.currentAction,currentScore=scores[current];
  const urgent=Math.max(th,h,heat*(1-o.traits.heatTolerance));
  if(o.actionHours<duration[current]&&urgent<.72)return{action:current,utility:currentScore};
  const actions=Object.keys(scores) as AnimalAction[];
  let best:AnimalAction='rest';
  for(const action of actions){if(scores[action]>scores[best])best=action;}
  return{action:best,utility:scores[best]};
}
