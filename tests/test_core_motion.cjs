const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let now=0,next=0;const frames=new Map(),events={},mediaEvents={},documentEvents={};
const context2d=new Proxy({}, {get:(o,k)=>o[k]??(k==='createRadialGradient'?()=>({addColorStop(){}}):()=>{}),set:(o,k,v)=>(o[k]=v,true)});
const bounds=()=>({left:0,top:0,width:600,height:480});
const canvas={getContext:()=>context2d,getBoundingClientRect:bounds};
const figure={getBoundingClientRect:bounds,addEventListener:(name,callback)=>events[name]=callback};
const reduced={matches:false,addEventListener:(name,callback)=>mediaEvents[name]=callback};
const document={hidden:false,getElementById:id=>id==='acousticCore'?canvas:id==='coreFigure'?figure:{textContent:''},addEventListener:(name,callback)=>documentEvents[name]=callback};
const window={addEventListener(){}};
const source=fs.readFileSync('core-visual.js','utf8').replaceAll(';size();start();',';window.inspectCore=()=>({rotation,speed,px,py});size();start();');
vm.runInNewContext(source,{document,window,performance:{now:()=>now},navigator:{hardwareConcurrency:8,deviceMemory:8},devicePixelRatio:1,matchMedia:()=>reduced,requestAnimationFrame:callback=>{frames.set(++next,callback);return next},cancelAnimationFrame:id=>frames.delete(id),ResizeObserver:class{observe(){}disconnect(){}},IntersectionObserver:class{observe(){}}});
function step(ms){for(let elapsed=0;elapsed<ms;elapsed+=50){now+=50;const callbacks=[...frames.values()];frames.clear();callbacks.forEach(f=>f(now));}}
step(5000);const a=window.inspectCore();assert(a.rotation>.5,'idle should rotate continuously');
events.pointermove({clientX:550,clientY:60});step(5000);const b=window.inspectCore();
assert(b.rotation>a.rotation,'hover still rotates');assert(b.speed<a.speed/3,'hover slows smoothly');assert(b.px>.5&&b.py<-.5,'pointer followed');
events.pointerleave();step(5000);const c=window.inspectCore();assert(c.speed>b.speed*3,'leave restores speed');
reduced.matches=true;mediaEvents.change();step(1000);const d=window.inspectCore();step(1000);assert.equal(window.inspectCore().rotation,d.rotation,'reduced motion remains still');
console.log('PASS: continuous idle rotation, hover slowdown, cursor following, leave recovery and reduced motion.');
