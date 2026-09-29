// JavaScript is the browser/Canvas bridge; numerical kernels live in Swift/WASM.
(() => {
let pending;
async function load(){
  if(!pending)pending=(async()=>{
    const response=await fetch(globalThis.PROTEIN_WASM_URL||'./core.wasm');
    if(!response.ok)throw Error('Swift core could not be loaded');
    let memory;
    const imports={wasi_snapshot_preview1:{random_get:(ptr,length)=>{
      if(!memory)return 52;
      const bytes=new Uint8Array(memory.buffer,ptr,length);
      for(let i=0;i<length;i+=65536)crypto.getRandomValues(bytes.subarray(i,i+65536));
      return 0;
    }}};
    const {instance}=await WebAssembly.instantiate(await response.arrayBuffer(),imports);
    memory=instance.exports.memory;
    instance.exports._initialize?.();
    return instance.exports;
  })().catch(error=>{pending=null;throw error;});
  return pending;
}
async function folding(fold){
  const core=await load();core.fold_init();
  fold.step=function(n){
    core.fold_step(n);
    const values=new Float64Array(core.memory.buffer,core.fold_positions(),84);
    this.points=Array.from({length:28},(_,i)=>Array.from(values.subarray(i*3,i*3+3)));
    this.energy=core.fold_energy();this.best=core.fold_best();this.steps=core.fold_steps();this.accepted=core.fold_accepted();
    if(this.steps-(this.historyAt||0)>=100){this.history.push(this.best);this.historyAt=this.steps;if(this.history.length>150)this.history.shift();}
  };
  fold.step(0);fold.swift=true;return fold;
}
globalThis.SwiftCore={load,folding};
})();
