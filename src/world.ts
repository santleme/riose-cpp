import type { Animal, FarmSimulation } from './simulation';

type Point = { x:number; y:number };
type Color = { coat:string; shade:string; light:string; spots:string };

const COW_COLORS:Record<string,Color> = {
  'coat-white': {coat:'#eee9d4',shade:'#bcb9a6',light:'#fff8df',spots:'#766e5d'},
  'coat-brown': {coat:'#98603d',shade:'#68432f',light:'#bd8051',spots:'#553729'},
  'coat-spotted': {coat:'#e8dfc6',shade:'#bcb5a0',light:'#fff4d9',spots:'#4e4a40'},
  'coat-gray': {coat:'#b8b6a6',shade:'#828475',light:'#d4d1bd',spots:'#5f6259'},
};

export class FarmRenderer {
  private ctx:CanvasRenderingContext2D;
  private width=0; private height=0; private pixelRatio=1;
  private zoom=1; private panX=0; private panY=0; private selectedId:string|null=null;
  private drag:{x:number;y:number;panX:number;panY:number}|null=null;
  private keys=new Set<string>();
  private lastTime=0; private pointer={x:0,y:0};
  private cattleAtlas:HTMLImageElement|null=null;
  private farmLandscape:HTMLImageElement|null=null;
  private get visualScale(){return this.pixelRatio*Math.min(1,this.width/900)*this.zoom;}

  constructor(private canvas:HTMLCanvasElement, private sim:FarmSimulation) {
    this.ctx=canvas.getContext('2d',{alpha:false})!;
    const atlas=new Image();atlas.onload=()=>{this.cattleAtlas=atlas;};atlas.src='/assets/cattle-atlas.png';
    const landscape=new Image();landscape.onload=()=>{this.farmLandscape=landscape;};landscape.src='/assets/farm-landscape.png';
    this.resize();
    window.addEventListener('resize',()=>this.resize());
    window.addEventListener('keydown',e=>this.onKeyDown(e));
    window.addEventListener('keyup',e=>this.keys.delete(e.key.toLowerCase()));
    canvas.addEventListener('pointerdown',e=>this.onPointerDown(e));
    canvas.addEventListener('pointermove',e=>this.onPointerMove(e));
    canvas.addEventListener('pointerup',e=>this.onPointerUp(e));
    canvas.addEventListener('pointercancel',()=>this.finishDrag());
    canvas.addEventListener('wheel',e=>this.onWheel(e),{passive:false});
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
  }

  private resize(){
    this.pixelRatio=Math.min(2,window.devicePixelRatio||1);
    this.width=Math.max(1,Math.round(this.canvas.clientWidth*this.pixelRatio));
    this.height=Math.max(1,Math.round(this.canvas.clientHeight*this.pixelRatio));
    this.canvas.width=this.width;this.canvas.height=this.height;
    this.ctx.imageSmoothingEnabled=false;
    if(this.width&&this.width<this.height&&this.zoom===1)this.zoom=2;
  }

  private onKeyDown(e:KeyboardEvent){
    const key=e.key.toLowerCase();
    if(['arrowup','arrowdown','arrowleft','arrowright','w','a','s','d','+','=','-','0'].includes(key))e.preventDefault();
    this.keys.add(key);
    if(key==='0'){this.zoom=1;this.panX=0;this.panY=0;this.selectedId=null;}
    if(key==='+'||key==='=')this.zoomAt(this.width/2,this.height/2,1.12);
    if(key==='-')this.zoomAt(this.width/2,this.height/2,1/1.12);
  }

  private onPointerDown(e:PointerEvent){
    const p=this.localPointer(e);this.pointer=p;
    if(e.button===0){
      const hit=this.hitAnimal(p.x,p.y);
      if(hit){this.selectedId=hit.id;const center=this.project(hit.x,hit.y);this.panX+=this.width/2-center.x;this.panY+=this.height*.50-center.y;return;}
    }
    this.drag={x:p.x,y:p.y,panX:this.panX,panY:this.panY};this.canvas.classList.add('dragging');this.canvas.setPointerCapture(e.pointerId);
  }

  private onPointerMove(e:PointerEvent){
    const p=this.localPointer(e);this.pointer=p;
    if(this.drag){this.panX=this.drag.panX+p.x-this.drag.x;this.panY=this.drag.panY+p.y-this.drag.y;}
  }

  private onPointerUp(e:PointerEvent){
    if(this.drag){this.finishDrag();if(this.canvas.hasPointerCapture(e.pointerId))this.canvas.releasePointerCapture(e.pointerId);}
  }

  private finishDrag(){this.drag=null;this.canvas.classList.remove('dragging');}

  private onWheel(e:WheelEvent){e.preventDefault();const p=this.localPointer(e);this.zoomAt(p.x,p.y,e.deltaY<0?1.11:1/1.11);}

  private zoomAt(x:number,y:number,factor:number){
    const next=Math.max(.72,Math.min(2.15,this.zoom*factor));
    const cx=this.width/2,cy=this.height/2;
    const baseX=cx+(x-cx-this.panX)/this.zoom,baseY=cy+(y-cy-this.panY)/this.zoom;
    this.zoom=next;this.panX=x-cx-(baseX-cx)*next;this.panY=y-cy-(baseY-cy)*next;
  }

  private localPointer(e:PointerEvent|WheelEvent){const r=this.canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*this.pixelRatio,y:(e.clientY-r.top)*this.pixelRatio};}

  private project(x:number,y:number):Point{
    const cx=this.width/2,cy=this.height/2;
    const unitX=Math.min(this.width*.47,this.height*.42*1.79),unitY=unitX/1.79;
    const baseX=cx+(x-y)*unitX;
    const baseY=cy+(x+y-1)*unitY;
    return{x:cx+(baseX-cx)*this.zoom+this.panX,y:cy+(baseY-cy)*this.zoom+this.panY};
  }

  private polygon(points:Point[],fill:string,stroke?:string,lineWidth=1){
    const c=this.ctx;c.beginPath();points.forEach((p,i)=>i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y));c.closePath();c.fillStyle=fill;c.fill();
    if(stroke){c.strokeStyle=stroke;c.lineWidth=lineWidth*this.visualScale;c.stroke();}
  }

  private box(x:number,y:number,w:number,h:number,color:string){this.ctx.fillStyle=color;this.ctx.fillRect(Math.round(x),Math.round(y),Math.max(1,Math.round(w)),Math.max(1,Math.round(h)));}

  draw(now:number){
    this.lastTime=now;const c=this.ctx,w=this.width,h=this.height,t=now/1000;
    c.clearRect(0,0,w,h);this.drawHorizon(t);
    if(this.farmLandscape){
      this.drawFarmLandscape();
      for(const animal of [...this.sim.animals].sort((a,b)=>a.x+a.y-(b.x+b.y)))this.drawAnimal(animal,t);
      this.drawRain(t);this.stepCameraKeys();return;
    }
    this.drawIsland();this.drawGround();this.drawPaths();this.drawPond(t);this.drawFence();
    const actors:Array<{depth:number;draw:()=>void}>=[];
    for(const [x,y,s] of [[.12,.24,.85],[.20,.13,.75],[.77,.15,.86],[.90,.27,.92],[.10,.80,.76],[.92,.57,.76],[.70,.87,.78],[.28,.86,.74],[.62,.25,.58],[.17,.55,.61]] as number[][]){actors.push({depth:x+y,draw:()=>this.drawTree(x,y,s,t)});}
    actors.push({depth:.42,draw:()=>this.drawWindmill(.20,.34,t)});
    actors.push({depth:.54,draw:()=>this.drawBarn(.43,.20,t)});
    actors.push({depth:.77,draw:()=>this.drawTrough(.27,.67)});
    actors.push({depth:.43,draw:()=>this.drawHay(.64,.38)});
    for(const animal of this.sim.animals)actors.push({depth:animal.x+animal.y,draw:()=>this.drawAnimal(animal,t)});
    actors.sort((a,b)=>a.depth-b.depth).forEach(actor=>actor.draw());
    this.drawForegroundFence();this.drawPlants(t);this.drawRain(t);
    this.stepCameraKeys();
  }

  private drawFarmLandscape(){
    if(!this.farmLandscape)return;
    const cx=this.width/2,cy=this.height/2;
    const unitX=Math.min(this.width*.47,this.height*.42*1.79),unitY=unitX/1.79;
    const x=cx-unitX*this.zoom+this.panX,y=cy-unitY*this.zoom+this.panY;
    this.ctx.drawImage(this.farmLandscape,x,y,unitX*2*this.zoom,unitY*2*this.zoom);
  }

  private drawHorizon(t:number){
    const c=this.ctx,w=this.width,h=this.height;
    const sky=c.createLinearGradient(0,0,0,h);sky.addColorStop(0,'#b8d3c2');sky.addColorStop(.38,'#d7dfbd');sky.addColorStop(.78,'#b5c992');sky.addColorStop(1,'#769c69');c.fillStyle=sky;c.fillRect(0,0,w,h);
    c.fillStyle='rgba(248,225,150,.72)';c.beginPath();c.arc(w*.83,h*.13,Math.min(w,h)*.055,0,Math.PI*2);c.fill();
    const ridge=(base:number,amp:number,color:string,offset:number)=>{c.fillStyle=color;c.beginPath();c.moveTo(0,h);for(let x=0;x<=w+16;x+=16){const y=h*base+Math.sin(x/w*7+offset)*h*amp+Math.sin(x/w*17+offset*2)*h*amp*.25;c.lineTo(x,y);}c.lineTo(w,h);c.closePath();c.fill();};
    ridge(.40,.09,'#90b289',.3);ridge(.48,.075,'#7fa675',1.2);ridge(.57,.06,'#6f9868',2.4);
    this.box(0,h*.59,w,h*.41,'#b6c98c');
    for(let i=0;i<35;i++){const x=(i*151%997)/997*w,y=h*(.42+(i*71%140)/1000),sz=(2+i%4)*this.visualScale;this.box(x,y,sz,sz,'#587d59');}
  }

  private drawIsland(){
    const pA=this.project(0,0),pB=this.project(1,0),pC=this.project(1,1),pD=this.project(0,1);
    this.polygon([{x:pA.x,y:pA.y+12*this.visualScale},{x:pB.x,y:pB.y+12*this.visualScale},{x:pC.x,y:pC.y+17*this.visualScale},{x:pD.x,y:pD.y+17*this.visualScale}],'#354f3a');
    this.polygon([pA,pB,pC,pD],'#718e56','#657b4b',2);
    this.polygon([this.project(.018,.018),this.project(.982,.018),this.project(.982,.982),this.project(.018,.982)],'#789a5d');
    const bank=[this.project(.04,.04),this.project(.96,.04),this.project(.96,.96),this.project(.04,.96)];
    const ctx=this.ctx;ctx.strokeStyle='#a5ad68';ctx.lineWidth=4*this.visualScale;ctx.beginPath();bank.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.stroke();
  }

  private drawGround(){
    const c=this.ctx;
    for(let i=0;i<this.sim.patches.length;i++){
      const patch=this.sim.patches[i],p=this.project(patch.x,patch.y);
      const green=Math.round(95+patch.grass*.35),red=Math.round(78+patch.grass*.10);
      const bladeCount=Math.round(12+patch.grass*.25);
      for(let n=0;n<bladeCount;n++){
        const u=((i*47+n*29)%101)/101-.5,v=((i*61+n*43)%101)/101-.5;
        const q=this.project(Math.max(.02,Math.min(.98,patch.x+u*.075)),Math.max(.02,Math.min(.98,patch.y+v*.075)));
        const color=n%5===0?`rgba(221,214,137,${.25+patch.moisture/500})`:n%3===0?`rgba(46,91,55,${.15+patch.grass/700})`:`rgba(${red},${green},74,.48)`;
        this.box(q.x,q.y,2*this.visualScale,2*this.visualScale,color);
      }
    }
    for(let i=0;i<125;i++){
      const x=((i*179)%991)/991,y=((i*313+71)%983)/983,p=this.project(.06+x*.88,.06+y*.88);
      if(i%9===0){this.box(p.x,p.y,3*this.visualScale,2*this.visualScale,'rgba(233,224,157,.72)');this.box(p.x+2*this.visualScale,p.y-2*this.visualScale,2*this.visualScale,2*this.visualScale,'#d8c76d');}
    }
  }

  private drawPaths(){
    const c=this.ctx;const paths=[[[.16,.76],[.42,.54],[.44,.29]],[[.42,.54],[.76,.62],[.83,.76]],[[.16,.76],[.27,.68]]];
    for(const path of paths){const pts=path.map(([x,y])=>this.project(x,y));c.lineCap='square';c.lineJoin='miter';c.strokeStyle='#a98e5d';c.lineWidth=15*this.visualScale;c.beginPath();pts.forEach((p,i)=>i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y));c.stroke();c.strokeStyle='#d0b57c';c.lineWidth=9*this.visualScale;c.beginPath();pts.forEach((p,i)=>i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y));c.stroke();
      for(let i=0;i<pts.length-1;i++){const a=pts[i],b=pts[i+1];for(let k=0;k<16;k++){const t=k/16,x=a.x+(b.x-a.x)*t,y=a.y+(b.y-a.y)*t;this.box(x,y,3*this.visualScale,2*this.visualScale,'rgba(126,96,54,.38)');}}
    }
  }

  private drawPond(t:number){
    const c=this.ctx,p=this.project(.84,.76);c.fillStyle='rgba(46,69,49,.25)';c.beginPath();c.ellipse(p.x+8*this.visualScale,p.y+11*this.visualScale,88*this.visualScale,45*this.visualScale,0,0,Math.PI*2);c.fill();
    c.fillStyle='#436f5b';c.beginPath();c.ellipse(p.x,p.y,83*this.visualScale,47*this.visualScale,0,0,Math.PI*2);c.fill();
    c.fillStyle='#397f83';c.beginPath();c.ellipse(p.x,p.y-5*this.visualScale,75*this.visualScale,38*this.visualScale,0,0,Math.PI*2);c.fill();
    c.fillStyle='#59a4a0';c.beginPath();c.ellipse(p.x-10*this.visualScale,p.y-13*this.visualScale,39*this.visualScale,10*this.visualScale,-.18,0,Math.PI*2);c.fill();
    c.fillStyle='#79b9a6';c.beginPath();c.ellipse(p.x+24*this.visualScale,p.y+4*this.visualScale,21*this.visualScale,6*this.visualScale,.12,0,Math.PI*2);c.fill();
    const ripple=(t*.026)%1;c.strokeStyle=`rgba(183,218,180,${.46*(1-ripple)})`;c.lineWidth=2*this.visualScale;c.beginPath();c.ellipse(p.x,p.y,15*this.visualScale+ripple*38*this.visualScale,5*this.visualScale+ripple*18*this.visualScale,0,0,Math.PI*2);c.stroke();
    for(const [x,y] of [[-70,-1],[68,0],[0,-41],[0,39]] as number[][]){this.box(p.x+x*this.visualScale,p.y+y*this.visualScale,9*this.visualScale,4*this.visualScale,'#668a5b');}
  }

  private drawFence(){
    const edges=[[[.04,.04],[.96,.04]],[[.96,.04],[.96,.96]],[[.04,.04],[.04,.96]]];
    for(const [[ax,ay],[bx,by]] of edges as number[][][]){const a=this.project(ax,ay),b=this.project(bx,by);this.drawFenceLine(a,b,ax<.05||bx<.05);}
    // A small stockyard pen tucked beside the barn.
    const pen=[[.54,.16],[.66,.16],[.66,.27],[.54,.27],[.54,.16]] as Array<[number,number]>;
    for(let i=0;i<pen.length-1;i++)this.drawFenceLine(this.project(...pen[i]),this.project(...pen[i+1]),false,.62);
  }

  private drawFenceLine(a:Point,b:Point,left=false,scale=1){
    const c=this.ctx;const color='#684f36';c.strokeStyle='#75593c';c.lineWidth=3*this.visualScale*scale;c.beginPath();c.moveTo(a.x,a.y-5*this.visualScale);c.lineTo(b.x,b.y-5*this.visualScale);c.stroke();
    for(let t=0;t<=1.001;t+=.075){if(!left&&t>.44&&t<.54)continue;const x=a.x+(b.x-a.x)*t,y=a.y+(b.y-a.y)*t;
      this.box(x-2*this.visualScale,y-8*this.visualScale,4*this.visualScale,14*this.visualScale,color);this.box(x-3*this.visualScale,y-9*this.visualScale,6*this.visualScale,3*this.visualScale,'#9a7650');
      this.box(x-2*this.visualScale,y-4*this.visualScale,4*this.visualScale,2*this.visualScale,'#98744c');
    }
  }

  private drawForegroundFence(){this.drawFenceLine(this.project(.04,.96),this.project(.96,.96),false);}

  private drawTree(x:number,y:number,scale:number,t:number){
    const c=this.ctx,p=this.project(x,y),s=scale*this.visualScale;const sway=Math.sin(t*.55+x*13)*.7*s;
    c.fillStyle='rgba(39,58,37,.24)';c.beginPath();c.ellipse(p.x+12*s,p.y+7*s,27*s,10*s,0,0,Math.PI*2);c.fill();
    this.box(p.x-4*s,p.y-28*s,9*s,32*s,'#705037');this.box(p.x-2*s,p.y-26*s,4*s,28*s,'#a1784e');
    this.box(p.x-18*s+sway,p.y-48*s,34*s,22*s,'#31563c');this.box(p.x-25*s+sway,p.y-42*s,34*s,20*s,'#3c6944');this.box(p.x-9*s+sway,p.y-56*s,29*s,19*s,'#4c7b4a');
    this.box(p.x-3*s+sway,p.y-51*s,18*s,12*s,'#61894e');this.box(p.x-17*s+sway,p.y-40*s,10*s,7*s,'#719253');
    this.box(p.x-14*s+sway,p.y-45*s,5*s,3*s,'#d29c4f');if((Math.floor(x*100+y*50)%3)===0)this.box(p.x+8*s+sway,p.y-49*s,4*s,4*s,'#d5af63');
  }

  private drawBarn(x:number,y:number,t:number){
    const c=this.ctx,p=this.project(x,y),s=this.visualScale;
    this.polygon([{x:p.x-47*s,y:p.y-3*s},{x:p.x+4*s,y:p.y+16*s},{x:p.x+61*s,y:p.y-4*s},{x:p.x+10*s,y:p.y-24*s}],'#9b8057');
    this.polygon([{x:p.x-33*s,y:p.y-47*s},{x:p.x+13*s,y:p.y-33*s},{x:p.x+13*s,y:p.y+11*s},{x:p.x-33*s,y:p.y-4*s}],'#d2bd8e');
    this.polygon([{x:p.x+13*s,y:p.y-33*s},{x:p.x+48*s,y:p.y-47*s},{x:p.x+48*s,y:p.y-4*s},{x:p.x+13*s,y:p.y+11*s}],'#b9a477');
    this.polygon([{x:p.x-42*s,y:p.y-47*s},{x:p.x+9*s,y:p.y-70*s},{x:p.x+55*s,y:p.y-49*s},{x:p.x+13*s,y:p.y-29*s}],'#773d31');
    this.polygon([{x:p.x-42*s,y:p.y-47*s},{x:p.x+9*s,y:p.y-70*s},{x:p.x+9*s,y:p.y-62*s},{x:p.x-35*s,y:p.y-42*s}],'#a6533c');
    this.polygon([{x:p.x+9*s,y:p.y-70*s},{x:p.x+55*s,y:p.y-49*s},{x:p.x+13*s,y:p.y-29*s},{x:p.x+9*s,y:p.y-35*s}],'#8d4936');
    for(let k=0;k<5;k++)this.box(p.x+(-29+k*9)*s,p.y+(-48+k*3)*s,26*s,2*s,'rgba(224,156,93,.62)');
    this.box(p.x-5*s,p.y-28*s,18*s,37*s,'#654536');this.box(p.x-2*s,p.y-23*s,12*s,32*s,'#78513b');this.box(p.x-27*s,p.y-26*s,9*s,9*s,'#8eb2a0');this.box(p.x-25*s,p.y-24*s,5*s,5*s,'#b3d1b4');
    this.box(p.x+33*s,p.y-40*s,7*s,23*s,'#d9c88e');this.box(p.x+35*s,p.y-38*s,3*s,16*s,'#f0d99d');
    if(Math.sin(t*.3)>.97)this.box(p.x+17*s,p.y-19*s,3*s,3*s,'#ffdc83');
  }

  private drawWindmill(x:number,y:number,t:number){
    const p=this.project(x,y),s=this.visualScale;this.box(p.x-6*s,p.y-43*s,12*s,43*s,'#e0d2ac');this.box(p.x-4*s,p.y-43*s,3*s,43*s,'#fff0c7');this.box(p.x-8*s,p.y-46*s,16*s,5*s,'#855a3e');
    const angle=t*.22,c=this.ctx;c.save();c.translate(p.x,p.y-45*s);c.rotate(angle);for(let i=0;i<4;i++){c.rotate(Math.PI/2);c.fillStyle='#d7c49a';c.fillRect(-2*s,-28*s,4*s,26*s);c.fillStyle='#f0e0b9';c.fillRect(-1*s,-27*s,2*s,23*s);}c.restore();
    c.fillStyle='#79573e';c.beginPath();c.arc(p.x,p.y-45*s,4*s,0,Math.PI*2);c.fill();
  }

  private drawTrough(x:number,y:number){
    const p=this.project(x,y),s=this.visualScale;
    this.polygon([{x:p.x-24*s,y:p.y-7*s},{x:p.x+11*s,y:p.y-7*s},{x:p.x+20*s,y:p.y-1*s},{x:p.x-15*s,y:p.y}],'#75513a');
    this.polygon([{x:p.x-24*s,y:p.y-9*s},{x:p.x+10*s,y:p.y-9*s},{x:p.x+19*s,y:p.y-3*s},{x:p.x-15*s,y:p.y-3*s}],'#65969a');
    this.box(p.x-21*s,p.y-7*s,32*s,2*s,'#8cc4ba');this.box(p.x-22*s,p.y,p.x?4*s:4*s,10*s,'#6e523a');this.box(p.x+13*s,p.y-1*s,4*s,10*s,'#6e523a');
  }

  private drawHay(x:number,y:number){
    const p=this.project(x,y),s=this.visualScale;
    this.polygon([{x:p.x-15*s,y:p.y-1*s},{x:p.x,y:p.y-8*s},{x:p.x+15*s,y:p.y-1*s},{x:p.x,y:p.y+6*s}],'#c9a956');
    this.polygon([{x:p.x-15*s,y:p.y-1*s},{x:p.x,y:p.y+6*s},{x:p.x,y:p.y+15*s},{x:p.x-15*s,y:p.y+8*s}],'#a77f43');
    this.polygon([{x:p.x,y:p.y+6*s},{x:p.x+15*s,y:p.y-1*s},{x:p.x+15*s,y:p.y+8*s},{x:p.x,y:p.y+15*s}],'#bd954c');
    this.box(p.x-10*s,p.y+1*s,12*s,2*s,'#ead076');
  }

  private drawAnimal(a:Animal,t:number){
    const c=this.ctx,s=this.visualScale*(a.id===this.selectedId?1.12:1),p=this.project(a.x,a.y),coat=COW_COLORS[a.coat]||COW_COLORS['coat-white'];
    const dx=a.targetX-a.x,dy=a.targetY-a.y,angle=Math.abs(dx)+Math.abs(dy)>.01?(Math.abs(dx)>Math.abs(dy)?(dx>0?0:Math.PI):dy>0?.22:Math.PI-.22):0;
    const walking=a.state==='walking',grazing=a.state==='grazing',bob=walking?Math.sin(t*12+a.x*40)*1.3*s:grazing?Math.sin(t*3+a.y*30)*1.1*s:0;
    if(this.cattleAtlas){
      const variant=a.breed==='Angus'?1:a.breed==='Girolando'?2:a.age<=2?3:0;
      const cellW=this.cattleAtlas.width/2,cellH=this.cattleAtlas.height/2,sx=(variant%2)*cellW,sy=Math.floor(variant/2)*cellH;
      c.save();c.translate(p.x,p.y+bob);if(angle>Math.PI/2)c.scale(-1,1);
      c.fillStyle='rgba(37,56,37,.2)';c.beginPath();c.ellipse(0,8*s,31*s,10*s,0,0,Math.PI*2);c.fill();
      if(a.id===this.selectedId){c.strokeStyle='rgba(244,218,133,.82)';c.lineWidth=1.5*this.visualScale;c.beginPath();c.ellipse(0,8*s,34*s,11*s,0,0,Math.PI*2);c.stroke();}
      c.drawImage(this.cattleAtlas,sx,sy,cellW,cellH,-47*s,-58*s,94*s,73*s);
      c.restore();return;
    }
    c.save();c.translate(p.x,p.y+bob);if(angle>Math.PI/2)c.scale(-1,1);
    c.fillStyle='rgba(34,53,36,.27)';c.beginPath();c.ellipse(3*s,8*s,23*s,9*s,0,0,Math.PI*2);c.fill();
    const tail=Math.sin(t*5+a.y*21)*2.8*s;this.box(-23*s,0+tail,9*s,2*s,coat.shade);this.box(-29*s,tail+1*s,6*s,2*s,coat.spots);
    // Four short legs; the paired stride changes while the animal walks.
    const stride=walking?Math.sin(t*12+a.x*22)*3*s:0;
    for(const [lx,off] of [[-14,stride],[ -5,-stride],[7,-stride],[15,stride]] as number[][]){
      this.box(lx*s,4*s+off,4*s,10*s,coat.shade);this.box((lx+1)*s,12*s+off,4*s,2*s,'#544638');
    }
    this.box(-18*s,-9*s,37*s,19*s,coat.shade);this.box(-16*s,-12*s,33*s,18*s,coat.coat);this.box(-13*s,-14*s,26*s,5*s,coat.light);this.box(4*s,-11*s,10*s,13*s,coat.coat);
    if(a.coat==='coat-spotted'||a.coat==='coat-white'){
      this.box(-10*s,-8*s,7*s,7*s,coat.spots);this.box(4*s,0*s,9*s,6*s,coat.spots);this.box(-2*s,-12*s,5*s,3*s,coat.shade);
    }
    // Zebu shoulder hump and dewlap read clearly at the game's camera angle.
    if(a.breed==='Nelore'){this.box(5*s,-16*s,10*s,7*s,coat.light);this.box(11*s,-8*s,3*s,7*s,coat.shade);this.box(-17*s,-11*s,4*s,4*s,coat.light);}
    const headDrop=grazing?4*s:0;this.box(14*s,-12*s+headDrop,13*s,12*s,coat.coat);this.box(22*s,-8*s+headDrop,8*s,7*s,coat.light);this.box(27*s,-6*s+headDrop,5*s,4*s,a.coat==='coat-brown'?'#684131':'#9b7764');
    this.box(18*s,-16*s+headDrop,4*s,5*s,coat.shade);this.box(24*s,-16*s+headDrop,4*s,5*s,coat.shade);this.box(23*s,-10*s+headDrop,2*s,2*s,'#2a3027');
    if(a.breed==='Nelore'){this.box(17*s,-18*s+headDrop,3*s,3*s,'#ead8b4');this.box(27*s,-17*s+headDrop,3*s,3*s,'#ead8b4');}
    if(a.state==='drinking'){this.box(28*s,-3*s,5*s,2*s,'#779fa0');}
    c.restore();
    if(a.id===this.selectedId){c.strokeStyle='rgba(244,218,133,.8)';c.lineWidth=1.5*this.visualScale;c.beginPath();c.ellipse(p.x,p.y+13*s,25*s,8*s,0,0,Math.PI*2);c.stroke();}
  }

  private drawPlants(t:number){
    const c=this.ctx;for(let i=0;i<48;i++){
      const x=.08+((i*193)%827)/1000,y=.1+((i*307+29)%781)/1000,p=this.project(x,y),wave=Math.sin(t*1.8+i*4)*this.visualScale;
      const h=(3+i%4)*this.visualScale;this.box(p.x,p.y,2*this.visualScale,h,'#3e7046');this.box(p.x+wave,p.y-h,2*this.visualScale,h*.65,i%5===0?'#d5bd64':'#88a95b');
      if(i%7===0){this.box(p.x+3*this.visualScale,p.y-h,3*this.visualScale,3*this.visualScale,'#e5d98e');this.box(p.x+4*this.visualScale,p.y-h+1,1*this.visualScale,1*this.visualScale,'#b26d48');}
    }
  }

  private drawRain(t:number){
    if(this.sim.weather!=='rain')return;const c=this.ctx;c.strokeStyle='rgba(205,232,219,.48)';c.lineWidth=this.visualScale;
    for(let i=0;i<42;i++){const x=(i*173+t*44*this.visualScale)%this.width,y=(i*97+t*83*this.visualScale)%this.height;c.beginPath();c.moveTo(x,y);c.lineTo(x-4*this.visualScale,y+9*this.visualScale);c.stroke();}
  }

  private hitAnimal(x:number,y:number):Animal|null{
    let best:Animal|null=null,bestDistance=31*this.visualScale;
    for(const a of this.sim.animals){const p=this.project(a.x,a.y),d=Math.hypot(p.x-x,p.y-y);if(d<bestDistance){bestDistance=d;best=a;}}
    return best;
  }

  private stepCameraKeys(){
    const speed=7*this.pixelRatio;let dx=0,dy=0;
    if(this.keys.has('arrowleft')||this.keys.has('a'))dx+=speed;if(this.keys.has('arrowright')||this.keys.has('d'))dx-=speed;
    if(this.keys.has('arrowup')||this.keys.has('w'))dy+=speed;if(this.keys.has('arrowdown')||this.keys.has('s'))dy-=speed;
    this.panX+=dx;this.panY+=dy;
  }
}
