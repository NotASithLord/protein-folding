// Four independent Metropolis chains. Pair energies are reduced on the GPU.
(() => {
function gpuWork(current, elapsed, frameMS, timestamp, maximum) {
 if(!Number.isFinite(elapsed)||elapsed<=0)return current;
 const target=timestamp?frameMS*.875:Math.min(8,frameMS*.5);
 const ratio=elapsed/target;
 if(ratio>=.9714&&ratio<=1.0286)return current;
 const next=current*Math.max(.5,Math.min(1.25,1/ratio));
 return Math.max(1,Math.min(maximum,ratio<1?Math.ceil(next):Math.floor(next)));
}

const pairs=[];for(let i=0;i<26;i++)for(let j=i+2;j<28;j++)pairs.push(`vec2u(${i}u,${j}u)`);
const shader=`
struct Config { iterations:u32, round:u32, progress:f32, pad:u32 }
@group(0) @binding(0) var<uniform> config:Config;
@group(0) @binding(1) var<storage,read_write> data:array<vec4f>;
const pairs=array<vec2u,351>(${pairs.join(',')});
var<workgroup> current:array<vec3f,28>;
var<workgroup> candidate:array<vec3f,28>;
var<workgroup> sums:array<f32,64>;
var<workgroup> params:vec4f;
var<workgroup> stats:vec4f;
var<workgroup> accepted:u32;
var<workgroup> improved:u32;
fn hash(v:u32)->f32 {var x=v;x=((x>>16u)^x)*73244475u;x=((x>>16u)^x)*73244475u;return f32((x>>16u)^x)/4294967295.;}
fn hydro(i:u32)->bool{return i%5u!=1u&&i%5u!=4u;}
@compute @workgroup_size(64) fn fold(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 let chain=group.x;let base=chain*28u;
 if(lane<28u){current[lane]=data[base+lane].xyz;}
 if(lane==0u){stats=data[224u+chain];}workgroupBarrier();
 for(var step=0u;step<config.iterations;step++){
  if(lane==0u){let seed=u32(stats.z)*13u+chain*100003u+config.round*7919u;params=vec4f(floor(hash(seed+1u)*25.)+1.,floor(hash(seed+2u)*3.),(hash(seed+3u)-.5)*1.8,hash(seed+4u));}
  workgroupBarrier();
  if(lane<28u){var p=current[lane];let pivot=u32(params.x);if(lane>pivot){let axis=u32(params.y);let a=(axis+1u)%3u;let b=(axis+2u)%3u;let x=p[a]-current[pivot][a];let y=p[b]-current[pivot][b];p[a]=current[pivot][a]+x*cos(params.z)-y*sin(params.z);p[b]=current[pivot][b]+x*sin(params.z)+y*cos(params.z);}candidate[lane]=p;}
  workgroupBarrier();var energy=0.;
  for(var k=lane;k<351u;k+=64u){let pair=pairs[k];let d=distance(candidate[pair.x],candidate[pair.y]);if(d<.7){energy+=180.*(.7-d)*(.7-d);}if(hydro(pair.x)&&hydro(pair.y)){let z=(d-1.1)/.55;energy-=exp(-z*z);}}
  sums[lane]=energy;workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){sums[lane]+=sums[lane+stride];}workgroupBarrier();}
  if(lane==0u){let next=sums[0];let temperature=.07+.9*exp(-5.*config.progress);accepted=select(0u,1u,next<stats.x||params.w<exp((stats.x-next)/temperature));improved=0u;if(accepted==1u){stats.x=next;stats.w+=1.;if(next<stats.y){stats.y=next;improved=1u;}}stats.z+=1.;}
  workgroupBarrier();if(lane<28u&&accepted==1u){current[lane]=candidate[lane];if(improved==1u){data[112u+base+lane]=vec4f(current[lane],0.);}}workgroupBarrier();
 }
 if(lane<28u){data[base+lane]=vec4f(current[lane],0.);}if(lane==0u){data[224u+chain]=stats;}
}`;
class FoldingRace {
 constructor(){this.iterations=64;this.frameMS=1000/60;this.round=1;this.time=0;this.hold=0;this.phase='ready';this.busy=false;this.disposed=false;this.gpuMS=0;this.wins=[0,0,0,0];this.orbit=new ProteinView.Orbit();this.chains=Array.from({length:4},()=>({points:Array.from({length:28},(_,i)=>[i*.85,Math.sin(i*.9)*.3,Math.cos(i*.9)*.3]),bestPoints:null,energy:-2.5685235,best:-2.5685235,steps:0,zoom:8}));}
 async init(){if(!navigator.gpu)throw Error('WebGPU is required for parallel folding');const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('No WebGPU adapter');this.timestamp=adapter.features.has('timestamp-query');this.device=await adapter.requestDevice({requiredFeatures:this.timestamp?['timestamp-query']:[]});try{
 this.core=await SwiftCore.load();const d=this.device,module=d.createShaderModule({code:shader}),info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>m.message).join('\n'));
 this.pipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'fold'}});this.config=d.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.data=d.createBuffer({size:3648,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});this.read=d.createBuffer({size:3648,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});this.group=d.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.config}},{binding:1,resource:{buffer:this.data}}]});
 if(this.timestamp){this.query=d.createQuerySet({type:'timestamp',count:2});this.resolve=d.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});this.timeRead=d.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});}
 d.lost.then(()=>{if(!this.disposed)this.error='GPU connection lost. Reload to restart.';});this.resetRound();return this;
 }catch(e){this.device.destroy();throw e;}}
 resetRound(){this.core.fold_init();const points=new Float64Array(this.core.memory.buffer,this.core.fold_positions(),84),packed=new Float32Array(912),energy=this.core.fold_energy();for(let n=0;n<4;n++){for(let i=0;i<28;i++)for(let a=0;a<3;a++)packed[(n*28+i)*4+a]=packed[(112+n*28+i)*4+a]=points[i*3+a];packed.set([energy,energy,0,0],896+n*4);Object.assign(this.chains[n],{points:Array.from({length:28},(_,i)=>Array.from(points.subarray(i*3,i*3+3))),bestPoints:null,energy,best:energy,steps:0});}this.device.queue.writeBuffer(this.data,0,packed);this.time=0;this.phase='racing';this.winner=-1;}
 async step(intensity){if(this.busy||this.disposed||this.error||this.phase!=='racing')return;this.busy=true;const d=this.device,start=performance.now();try{const config=new ArrayBuffer(16);new Uint32Array(config).set([this.iterations,this.round,0,0]);new Float32Array(config)[2]=this.time/30;d.queue.writeBuffer(this.config,0,config);const encoder=d.createCommandEncoder(),pass=encoder.beginComputePass(this.timestamp?{timestampWrites:{querySet:this.query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.group);pass.dispatchWorkgroups(4);pass.end();encoder.copyBufferToBuffer(this.data,0,this.read,0,3648);if(this.timestamp){encoder.resolveQuerySet(this.query,0,2,this.resolve,0);encoder.copyBufferToBuffer(this.resolve,0,this.timeRead,0,16);}d.queue.submit([encoder.finish()]);await this.read.mapAsync(GPUMapMode.READ);const values=new Float32Array(this.read.getMappedRange().slice(0));this.read.unmap();if(this.disposed)return;if(!values.every(Number.isFinite))throw Error('Folding state became unstable');
 for(let n=0;n<4;n++){const c=this.chains[n],s=896+n*4;c.points=Array.from({length:28},(_,i)=>Array.from(values.subarray((n*28+i)*4,(n*28+i)*4+3)));c.bestPoints=Array.from({length:28},(_,i)=>Array.from(values.subarray((112+n*28+i)*4,(112+n*28+i)*4+3)));[c.energy,c.best,c.steps,c.accepted]=values.subarray(s,s+4);}
 if(this.timestamp){await this.timeRead.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(this.timeRead.getMappedRange());this.gpuMS=Number(times[1]-times[0])/1e6;this.timeRead.unmap();const at=performance.now();const gap=this.sampleAt?at-this.sampleAt:this.frameMS;if(gap>0&&gap<this.frameMS*3)this.cadenceMS=(this.cadenceMS||this.frameMS)*.8+gap*.2;window.dispatchEvent(new CustomEvent('protein-gpu-sample',{detail:{mode:'molecule',busy:this.gpuMS,interval:this.sampleAt?at-this.sampleAt:16.7}}));this.sampleAt=at;}
 this.iterations=gpuWork(this.iterations,this.timestamp?this.gpuMS:performance.now()-start,Math.max(this.frameMS,Math.min(this.frameMS*2,this.cadenceMS||this.frameMS)),this.timestamp,8192);
 }catch(e){if(!this.disposed){this.error=e.message;console.error('Protein folding:',e);}}finally{this.busy=false;if(!this.disposed&&!this.error&&!document.hidden&&performance.now()<this.activeUntil&&this.phase==='racing'&&this.time<30){clearTimeout(this.nextWork);this.nextWork=setTimeout(()=>{if(performance.now()<this.activeUntil&&!document.hidden&&this.time<30)this.step(intensity);},Math.max(0,start+this.frameMS-performance.now()));}}}
 advance(dt,intensity){if(!this.core||!dt||this.error)return;if(dt>=.004&&dt<.025)this.frameMS=Math.min(this.frameMS,dt*1000);if(this.phase==='racing'){this.time=Math.min(30,this.time+dt);if(this.time<30)this.step(intensity);else if(!this.busy){for(const c of this.chains){if(c.bestPoints){new Float64Array(this.core.memory.buffer,this.core.fold_positions(),84).set(c.bestPoints.flat());c.best=this.core.fold_score();}}const best=Math.min(...this.chains.map(c=>c.best));this.winners=this.chains.map((c,i)=>Math.abs(c.best-best)<.0001?i:-1).filter(i=>i>=0);this.winner=this.winners[0];this.winners.forEach(i=>this.wins[i]++);this.phase='result';this.hold=0;}}
 else if(this.phase==='result'){this.hold+=dt;if(this.hold>=4){this.round++;this.resetRound();}}}
 draw(ctx,dt,intensity=1){this.activeUntil=dt?performance.now()+100:0;this.advance(dt,intensity);const w=ctx.canvas.width,h=ctx.canvas.height;ctx.fillStyle='#101110';ctx.fillRect(0,0,w,h);ctx.save();const portrait=w/h<1.25,vw=portrait?460:900,vh=portrait?800:540,scale=Math.min(w/vw,h/vh);ctx.translate((w-vw*scale)/2,(h-vh*scale)/2);ctx.scale(scale,scale);const text=(s,x,y,size=12,color='#a6a69d')=>{ctx.fillStyle=color;ctx.font=`${size}px monospace`;ctx.fillText(s,x,y);};
 const stateKey=this.round+this.phase;if(this.announced!==stateKey){this.announced=stateKey;ctx.canvas.setAttribute('aria-label',this.phase==='result'?`Round ${this.round} complete. Fold ${this.winner+1} has the lowest energy. Drag to rotate.`:`Round ${this.round}. Four GPU protein folds. Drag to rotate, scroll to zoom.`);}
 const leader=this.chains.reduce((a,c,i)=>c.best<this.chains[a].best?i:a,0),result=this.phase==='result';
 text(`ROUND ${String(this.round).padStart(2,'0')}`,28,portrait?155:116,13,'#f2f0e9');text(result?(this.winners.length>1?'TIED ROUND':`FOLD 0${this.winner+1} WINS`):`${Math.max(0,30-this.time).toFixed(1)}s REMAINING`,portrait?230:570,portrait?155:116,16,result?'#e61c1b':'#f2f0e9');
 ctx.fillStyle='#f2f0e925';ctx.fillRect(28,portrait?170:130,vw-56,2);ctx.fillStyle='#e61c1b';ctx.fillRect(28,portrait?170:130,(vw-56)*(this.time/30),2);
 this.chains.forEach((c,n)=>{const x=28+(n%(portrait?2:4))*(portrait?208:214),top=(portrait?205:151)+(portrait?Math.floor(n/2)*251:0),cx=x+100,cy=top+(portrait?105:137),highlight=result?this.winners.includes(n):n===leader;ctx.strokeStyle=highlight?'#e61c1b':'#f2f0e930';ctx.lineWidth=highlight?1.5:.7;ctx.strokeRect(x,top,202,portrait?217:276);text(`FOLD 0${n+1}`,x+12,top+23,12,'#f2f0e9');if(highlight)text(result?'WINNER':'LEADING',x+12,top+41,9,'#e61c1b');
 const source=result&&c.bestPoints?c.bestPoints:c.points,center=[0,1,2].map(a=>source.reduce((s,p)=>s+p[a],0)/28),radius=Math.max(...source.map(p=>Math.hypot(...p.map((v,a)=>v-center[a])))),target=Math.min(30,(portrait?49:68)/Math.max(1,radius));c.zoom+=(target-c.zoom)*.09;
 const pts=source.map((p,i)=>({...this.orbit.project(p[0]-center[0],p[1]-center[1],p[2]-center[2],cx,cy,c.zoom),i}));ctx.save();ctx.beginPath();ctx.rect(x+2,top+47,198,portrait?111:171);ctx.clip();ctx.strokeStyle='#f2f0e9a0';ctx.lineWidth=1.5;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke();pts.sort((a,b)=>b.z-a.z).forEach(p=>{ctx.fillStyle=p.i%5!==1&&p.i%5!==4?'#e61c1b':'#f2f0e9';ctx.beginPath();ctx.arc(p.x,p.y,3.3*p.p,0,Math.PI*2);ctx.fill();});ctx.restore();text(c.best.toFixed(2),x+12,top+(portrait?176:240),22,'#f2f0e9');text('BEST ENERGY',x+12,top+(portrait?196:260),9);text(`${c.steps.toLocaleString()} MOVES`,x+12,top+(portrait?231:296),9);text(`${this.wins[n]} WINS`,portrait?x+134:x+12,top+(portrait?231:313),9);});
 text(this.error|| (result?'BEST CONFORMATIONS / NEXT ROUND IN '+Math.max(0,4-this.hold).toFixed(1)+'s':'FOUR INDEPENDENT SEARCHES / LOWEST ENERGY WINS'),28,portrait?763:502,portrait?8:11,this.error?'#e61c1b':'#a6a69d');if(this.timestamp)text(`${this.gpuMS.toFixed(2)} ms GPU`,portrait?28:710,portrait?784:502,11);ctx.restore();return `Round ${this.round} / ${this.phase} / ${this.time.toFixed(1)}s / best energy ${Math.min(...this.chains.map(c=>c.best)).toFixed(2)}`;}
 destroy(){this.disposed=true;this.device?.destroy();}
}
globalThis.ProteinRace=FoldingRace;
})();
