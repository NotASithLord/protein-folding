import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const bytes=readFileSync(new URL('core.wasm',import.meta.url));
let memory;
const {instance}=await WebAssembly.instantiate(bytes,{wasi_snapshot_preview1:{random_get:(p,n)=>{if(!memory)return 52;crypto.getRandomValues(new Uint8Array(memory.buffer,p,n));return 0;}}});
const c=instance.exports;memory=c.memory;
c.fold_init();const initial=c.fold_energy();
const before=new Float64Array(memory.buffer,c.fold_positions(),84).slice();
for(let i=0;i<24;i++)c.fold_step(500);
const after=new Float64Array(memory.buffer,c.fold_positions(),84);
assert(c.fold_best()<initial-20,'Folding must lower energy substantially');
for(let i=1;i<28;i++){
 const length=p=>Math.hypot(...[0,1,2].map(a=>p[i*3+a]-p[(i-1)*3+a]));
 assert(Math.abs(length(before)-length(after))<1e-8,'Bond length must be conserved');
}
console.log('Folding:',initial,'→',c.fold_best(),'/ 12,000 proposals; bond lengths conserved');
assert(Math.abs(c.fold_score()-c.fold_energy())<1e-10,'Independent scoring must agree with the solver');
