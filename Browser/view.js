/* Canvas renderer for the Swift/WASM folding solver. */
(() => {
  const ink='#101110',paper='#f2f0e9',red='#e61c1b',muted='#93958a';
  const rng=seed=>()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
  class Orbit {
    constructor(){this.reset();}
    reset(){this.yaw=-.35;this.pitch=.18;this.zoom=1;}
    project(x,y,z,cx,cy,scale=1){const a=x*Math.cos(this.yaw)+z*Math.sin(this.yaw),b=z*Math.cos(this.yaw)-x*Math.sin(this.yaw),v=y*Math.cos(this.pitch)-b*Math.sin(this.pitch),depth=y*Math.sin(this.pitch)+b*Math.cos(this.pitch),p=700/(700+depth*scale);return{x:cx+a*scale*p*this.zoom,y:cy+v*scale*p*this.zoom,z:depth,p};}
  }
  function stage(ctx){const {width:w,height:h}=ctx.canvas;ctx.fillStyle=ink;ctx.fillRect(0,0,w,h);ctx.save();const s=Math.min(w/900,h/540);ctx.translate((w-900*s)/2,(h-540*s)/2);ctx.scale(s,s);return()=>ctx.restore();}
  function text(c,s,x,y,size=12,color=muted){c.fillStyle=color;c.font=`${size}px monospace`;c.fillText(s,x,y);}
  function trace(c,values,x,y,w,h){if(values.length<2)return;const low=Math.min(...values),span=Math.max(.01,Math.max(...values)-low);c.strokeStyle=red;c.lineWidth=2;c.beginPath();values.forEach((v,i)=>{const px=x+i*w/(values.length-1),py=y+h-(v-low)/span*h;i?c.lineTo(px,py):c.moveTo(px,py);});c.stroke();}
  class Fold {
    constructor(){this.random=rng(813);this.points=Array.from({length:28},(_,i)=>[i*.85,Math.sin(i*.9)*.3,Math.cos(i*.9)*.3]);this.hydro=Array.from({length:28},(_,i)=>i%5!==1&&i%5!==4);this.steps=0;this.accepted=0;this.energy=-2.5685235108062714;this.best=this.energy;this.history=[this.energy];this.time=0;}
    step(){throw Error('Swift solver is not initialized');}
    draw(ctx,dt,intensity){if(dt){this.time+=dt;this.step(20*intensity);}this.orbit??=new Orbit();const end=stage(ctx),center=[0,1,2].map(k=>this.points.reduce((s,p)=>s+p[k],0)/28),targetScale=Math.min(58,155/Math.max(...this.points.map(p=>Math.hypot(...p.map((v,k)=>v-center[k])))));this.zoom=(this.zoom||19)+(targetScale-(this.zoom||19))*.025;const scale=this.zoom;const pts=this.points.map((p,i)=>({...this.orbit.project(p[0]-center[0],p[1]-center[1],p[2]-center[2],310,285,scale),i}));
      
      ctx.strokeStyle='#f2f0e970';ctx.lineWidth=2;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke();
      for(let i=0;i<28;i++)for(let j=i+3;j<28;j++)if(this.hydro[i]&&this.hydro[j]&&Math.hypot(...this.points[i].map((v,k)=>v-this.points[j][k]))<1.6){ctx.strokeStyle='#e61c1b50';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.stroke();}
      pts.sort((a,b)=>a.z-b.z).forEach(p=>{ctx.fillStyle=this.hydro[p.i]?red:paper;ctx.strokeStyle=ink;ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,5.5,0,Math.PI*2);ctx.fill();ctx.stroke();});
      text(ctx,'LOWER ENERGY',625,146,12);text(ctx,this.energy.toFixed(2),625,192,35,paper);text(ctx,'MODEL UNITS',625,216,10);text(ctx,`${this.steps.toLocaleString()} PROPOSALS`,625,261);text(ctx,`${(100*this.accepted/Math.max(1,this.steps)).toFixed(1)}% ACCEPTED`,625,284);text(ctx,'BEST ENERGY / HISTORY',625,339,11);trace(ctx,this.history,625,355,240,85);
      text(ctx,'RED / HYDROPHOBIC     CREAM / POLAR',28,476,12);text(ctx,'FOLD / RELAX / FIND LOWER ENERGY',28,502,11);end();return `${this.steps} folding proposals / energy ${this.energy.toFixed(2)} / best ${this.best.toFixed(2)}`;}
  }
  globalThis.ProteinView={Fold,Orbit};
})();
