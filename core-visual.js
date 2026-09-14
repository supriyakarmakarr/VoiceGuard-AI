/* Perspective-projected 3D geometry; decorative structure is explicitly labeled.
   Only the center waveform uses microphone samples / measured waveform peaks. */
(()=>{
 const canvas=document.getElementById('acousticCore'),figure=document.getElementById('coreFigure');if(!canvas)return;
 const ctx=canvas.getContext('2d'),reduce=matchMedia('(prefers-reduced-motion: reduce)');
 let hovered=false,speed=.16,visible=true,active=false,signal=null,rotation=0,last=0,frame=0,width=1,height=1,px=0,py=0,tx=0,ty=0;
 const weak=(navigator.hardwareConcurrency||4)<=4||(navigator.deviceMemory||8)<=4,points=[],count=weak?170:330;
 for(let i=0;i<count;i++){const y=1-2*(i+.5)/count,a=i*2.399963229728653,r=Math.sqrt(1-y*y);points.push([r*Math.cos(a),y,r*Math.sin(a)]);}
 function size(){const rect=canvas.getBoundingClientRect();width=rect.width;height=rect.height;const ratio=Math.min(2,devicePixelRatio||1);canvas.width=width*ratio;canvas.height=height*ratio;ctx.setTransform(ratio,0,0,ratio,0,0);draw(performance.now(),true);}
 function project(x,y,z,r){const yaw=rotation+px*.2,c=Math.cos(yaw),s=Math.sin(yaw),xx=x*c-z*s,zz=x*s+z*c;const pitch=-.18+py*.13,cy=Math.cos(pitch),sy=Math.sin(pitch),yy=y*cy-zz*sy,depth=y*sy+zz*cy;const scale=3.7/(3.7-depth);return{x:width*.51+xx*r*scale,y:height*.49+yy*r*scale,z:depth};}
 function line(points3,r,alpha=.2,lineWidth=.7){ctx.beginPath();points3.forEach((p,i)=>{const q=project(...p,r);i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y);});ctx.strokeStyle=`rgba(89,88,151,${alpha*1.3})`;ctx.lineWidth=lineWidth;ctx.stroke();}
 function draw(now,once=false){if(!once&&(!visible||document.hidden))return;ctx.clearRect(0,0,width,height);const r=Math.min(width*.30,height*.33);px+=(tx-px)*.045;py+=(ty-py)*.045;
  const glow=ctx.createRadialGradient(width*.50,height*.47,1,width*.50,height*.47,r*1.35);glow.addColorStop(0,'rgba(190,215,246,.12)');glow.addColorStop(.65,'rgba(210,229,251,.13)');glow.addColorStop(1,'rgba(240,247,255,0)');ctx.fillStyle=glow;ctx.fillRect(0,0,width,height);
  // Outer elliptical orbits have actual depth and follow the same camera transform.
  for(let orbit=0;orbit<3;orbit++){const coords=[];for(let j=0;j<=128;j++){const a=j/128*Math.PI*2,tilt=.45+orbit*.65;coords.push([Math.cos(a)*1.25,Math.sin(a)*Math.cos(tilt)*1.25,Math.sin(a)*Math.sin(tilt)*1.25]);}line(coords,r,.10,.6);}
  for(let lat=-3;lat<=3;lat++){const y=lat*.24,radius=Math.sqrt(1-y*y),coords=[];for(let j=0;j<=100;j++){const a=j/100*Math.PI*2;coords.push([radius*Math.cos(a),y,radius*Math.sin(a)]);}line(coords,r,lat===0?.28:.14,.65);}
  for(let longitude=0;longitude<9;longitude++){const a=longitude/9*Math.PI,coords=[];for(let j=0;j<=100;j++){const theta=j/100*Math.PI*2;coords.push([Math.cos(theta)*Math.cos(a),Math.sin(theta),Math.cos(theta)*Math.sin(a)]);}line(coords,r,.12,.5);}
  const sorted=points.map(p=>project(...p,r)).sort((a,b)=>a.z-b.z);for(const p of sorted){ctx.fillStyle=`rgba(88,85,147,${.16+(p.z+1)*.2})`;ctx.beginPath();ctx.arc(p.x,p.y,.65+(p.z+1)*.5,0,Math.PI*2);ctx.fill();}
  // Idle signal is illustrative. Analysis and microphone modes replace it with measured values.
  ctx.lineWidth=1.15;for(let echo=-1;echo<=1;echo++){ctx.beginPath();for(let i=0;i<=110;i++){const x=(i/110-.5)*1.65,envelope=Math.exp(-x*x*4);let value=signal?.length?signal[Math.min(signal.length-1,Math.floor(i/111*signal.length))]:Math.sin(i*.46+rotation*3)*Math.cos(i*.115)*.24;value*=signal? .55:1;const p=project(x,value*envelope+echo*.045,0,r);i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y);}ctx.strokeStyle=echo===0?'rgba(82,75,142,.85)':'rgba(134,120,175,.3)';ctx.stroke();}
  for(let i=0;i<6;i++){const a=i/6*Math.PI*2+rotation*.7,p=project(Math.cos(a)*1.25,Math.sin(a)*.67,Math.sin(a)*1.05,r);ctx.fillStyle='#f8fbff';ctx.strokeStyle='#98b6da';ctx.lineWidth=.8;ctx.beginPath();ctx.arc(p.x,p.y,i%2===0?3:2,0,Math.PI*2);ctx.fill();ctx.stroke();}
  ctx.strokeStyle='#dce7f4';ctx.lineWidth=1;for(const [x,y]of [[width*.16,height*.21],[width*.85,height*.76]]){ctx.beginPath();ctx.moveTo(x-4,y);ctx.lineTo(x+4,y);ctx.moveTo(x,y-4);ctx.lineTo(x,y+4);ctx.stroke();}
 }
 function loop(now){frame=0;if(!visible||document.hidden)return;if(now-last>(weak?60:40)){const dt=Math.min((now-last)/1000,.08);const target=hovered?.025:active?.23:.16;speed+=(target-speed)*.08;if(!reduce.matches)rotation+=speed*dt;draw(now);last=now;}if(active||!reduce.matches)frame=requestAnimationFrame(loop);}
 function start(){if(!frame&&visible&&!document.hidden){last=performance.now();frame=requestAnimationFrame(loop);}}
 figure.addEventListener('pointermove',e=>{if(reduce.matches)return;const b=figure.getBoundingClientRect();tx=((e.clientX-b.left)/b.width-.5)*2;ty=((e.clientY-b.top)/b.height-.5)*2;hovered=true;start();},{passive:true});figure.addEventListener('pointerleave',()=>{tx=ty=0;hovered=false;start();});
 if('IntersectionObserver'in window)new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)start();else{cancelAnimationFrame(frame);frame=0;}},{threshold:0}).observe(figure);
 document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frame);frame=0;}else start();});reduce.addEventListener('change',()=>{cancelAnimationFrame(frame);frame=0;draw(performance.now(),true);start();});
 const resizeObserver=new ResizeObserver(size);resizeObserver.observe(figure);
 window.addEventListener('pagehide',()=>{cancelAnimationFrame(frame);frame=0;resizeObserver.disconnect();});
 window.addEventListener('pageshow',()=>{resizeObserver.observe(figure);size();start();});
 window.voiceCore={setActive(v){active=v;document.getElementById('coreState').textContent=v?'PROCESSING SIGNAL':'STANDBY';start();},setSignal(values){signal=values;document.getElementById('coreCaption').textContent=values?'Measured waveform · geometry is illustrative':'Illustrative acoustic geometry · move to explore';if(visible&&!document.hidden&&(reduce.matches||!active))draw(performance.now(),true);}};size();start();
})();
