#!/usr/bin/env python3
"""Build the Cube Sumo HTML arena with embedded AI policy. v3."""

import json, os

with open("models/policy.json") as f:
    policy_json = f.read()

html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cube Sumo — RL Arena v3</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;overflow:hidden;font-family:'Rajdhani',sans-serif;color:#fff;user-select:none}
canvas{display:block}
#hud{position:fixed;top:0;left:0;right:0;z-index:10;display:flex;justify-content:center;padding:20px 30px;pointer-events:none}
.sp{display:flex;align-items:center;gap:40px;background:rgba(0,0,0,0.6);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:14px 36px}
.ps{display:flex;flex-direction:column;align-items:center;gap:2px}
.pl{font-family:'Orbitron',monospace;font-size:11px;letter-spacing:3px;text-transform:uppercase;opacity:0.6}
.pl.r{color:#ff4060}.pl.b{color:#40a0ff}
.pp{font-family:'Orbitron',monospace;font-size:42px;font-weight:900;line-height:1}
.pp.r{color:#ff4060}.pp.b{color:#40a0ff}
.vs{font-family:'Orbitron',monospace;font-size:14px;font-weight:700;opacity:0.25;letter-spacing:2px}
#ann{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:20;font-family:'Orbitron',monospace;font-size:48px;font-weight:900;letter-spacing:6px;text-transform:uppercase;opacity:0;transition:opacity 0.3s;text-shadow:0 0 40px currentColor,0 0 80px currentColor;pointer-events:none}
#rl{font-family:'Orbitron',monospace;font-size:10px;letter-spacing:3px;opacity:0.3;text-align:center;margin-bottom:-4px}
#mp{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:10;display:flex;gap:8px;background:rgba(0,0,0,0.6);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:10px 16px;pointer-events:all}
.mb{font-family:'Orbitron',monospace;font-size:10px;letter-spacing:1px;padding:8px 16px;border:1px solid rgba(255,255,255,0.15);border-radius:8px;background:rgba(255,255,255,0.03);color:#fff;cursor:pointer;transition:all 0.2s;text-transform:uppercase}
.mb:hover{background:rgba(255,255,255,0.08)}.mb.ac{background:rgba(255,200,68,0.15);border-color:rgba(255,200,68,0.5);color:#ffcc44}
#sl{font-family:'Orbitron',monospace;font-size:9px;letter-spacing:2px;opacity:0.4;display:flex;align-items:center;padding:0 8px}
#badge{position:fixed;top:80px;left:50%;transform:translateX(-50%);z-index:10;font-family:'Orbitron',monospace;font-size:9px;letter-spacing:3px;opacity:0;transition:opacity 0.5s;color:#ffcc44;text-shadow:0 0 20px rgba(255,200,68,0.5);pointer-events:none;text-transform:uppercase}
</style>
</head>
<body>
<div id="hud"><div class="sp">
  <div class="ps"><div class="pl r">Crimson</div><div class="pp r" id="sr">0</div></div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:4px"><div id="rl">ROUND 1</div><div class="vs">VS</div></div>
  <div class="ps"><div class="pl b">Azure</div><div class="pp b" id="sb">0</div></div>
</div></div>
<div id="ann"></div>
<div id="badge">PPO Neural Network Active</div>
<div id="mp">
  <button class="mb ac" id="btn-aivai" onclick="SM('aivai')">AI vs AI</button>
  <button class="mb" id="btn-pvai" onclick="SM('pvai')">Player vs AI</button>
  <button class="mb" id="btn-pvp" onclick="SM('pvp')">P1 vs P2</button>
  <span id="sl">1×</span>
  <button class="mb" onclick="CS()">Speed</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ═══════════════════════════════════════════════════════════
// TRAINED POLICY
// ═══════════════════════════════════════════════════════════
const POLICY = ''' + policy_json + r''';

function aiPredict(obs) {
  let x = new Float64Array(obs);
  for (let i = 0; i < POLICY.layers.length; i++) {
    const W = POLICY.layers[i].weight, B = POLICY.layers[i].bias;
    const y = new Float64Array(W.length);
    for (let j = 0; j < W.length; j++) {
      let s = B[j];
      for (let k = 0; k < x.length; k++) s += W[j][k] * x[k];
      y[j] = i < POLICY.layers.length - 1 ? Math.tanh(s) : s;
    }
    x = y;
  }
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

// ═══════════════════════════════════════════════════════════
// OBSERVATION — must exactly match Python sumo_env._get_obs()
// ═══════════════════════════════════════════════════════════
const R = 5.5, ROUT = 5.3, MV = 15.0;

function buildObs(me, opp) {
  const dx = opp.x - me.x, dz = opp.z - me.z;
  const distBetween = Math.sqrt(dx*dx + dz*dz) || 1e-8;
  const ownDC = Math.sqrt(me.x*me.x + me.z*me.z) || 1e-8;
  const oppDC = Math.sqrt(opp.x*opp.x + opp.z*opp.z) || 1e-8;

  const mySpeed = Math.sqrt(me.vx*me.vx + me.vz*me.vz);
  let dotVel = 0;
  if (mySpeed > 0.01 && distBetween > 0.01) {
    dotVel = (me.vx*(dx/distBetween) + me.vz*(dz/distBetween)) / Math.max(mySpeed, 0.01);
  }

  const oppSpeed = Math.sqrt(opp.vx*opp.vx + opp.vz*opp.vz);
  let dotOpp = 0;
  if (oppSpeed > 0.01 && distBetween > 0.01) {
    dotOpp = (opp.vx*(-dx/distBetween) + opp.vz*(-dz/distBetween)) / Math.max(oppSpeed, 0.01);
  }

  const cross = me.vx * dz - me.vz * dx;
  const crossSign = Math.max(-1, Math.min(1, cross / Math.max(mySpeed * distBetween, 0.01)));

  return [
    me.x / R,              // 0
    me.z / R,              // 1
    me.vx / MV,            // 2
    me.vz / MV,            // 3
    opp.x / R,             // 4
    opp.z / R,             // 5
    opp.vx / MV,           // 6
    opp.vz / MV,           // 7
    distBetween / (2*R),   // 8
    ownDC / R,             // 9
    oppDC / R,             // 10
    dotVel,                // 11
    dotOpp,                // 12
    crossSign,             // 13
  ];
}

// ═══════════════════════════════════════════════════════════
// GAME ENGINE
// ═══════════════════════════════════════════════════════════
(function(){
const RH=0.4, CS=0.9, PF=28, FR=0.92, BN=0.3, GR=-25, RD=1800, WS=5, DT=1/60, SS=2;
const DF=PF/Math.SQRT2;
const AF=[[0,0],[PF,0],[-PF,0],[0,PF],[0,-PF],[DF,DF],[DF,-DF],[-DF,DF],[-DF,-DF]];

let scores={r:0,b:0}, round=1, frozen=false, slowMo=1, slowMoT=0, mode='aivai', spd=1;
const K={}, P=[], TP=[];
let camShk=0, camAng=0, camTY=9;

window.SM=function(m){mode=m;document.querySelectorAll('.mb').forEach(b=>b.classList.remove('ac'));
  document.getElementById('btn-'+m).classList.add('ac');
  const bg=document.getElementById('badge');bg.style.opacity=m==='pvp'?'0':'1';
  bg.textContent=m==='aivai'?'PPO Neural Network — Both Agents':m==='pvai'?'PPO Neural Network — Azure (WASD to play)':'';
  scores={r:0,b:0};round=1;$('sr').textContent='0';$('sb').textContent='0';$('rl').textContent='ROUND 1';reset();};
window.CS=function(){const s=[1,2,4];spd=s[(s.indexOf(spd)+1)%s.length];$('sl').textContent=spd+'×';};
function $(id){return document.getElementById(id);}

const scene=new THREE.Scene();scene.fog=new THREE.FogExp2(0x0a0a1a,0.035);
const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.1,100);cam.position.set(0,9,12);cam.lookAt(0,0,0);
const ren=new THREE.WebGLRenderer({antialias:true});ren.setSize(innerWidth,innerHeight);
ren.setPixelRatio(Math.min(devicePixelRatio,2));ren.shadowMap.enabled=true;ren.shadowMap.type=THREE.PCFSoftShadowMap;
ren.toneMapping=THREE.ACESFilmicToneMapping;ren.toneMappingExposure=1.1;ren.outputEncoding=THREE.sRGBEncoding;
document.body.appendChild(ren.domElement);

scene.add(new THREE.AmbientLight(0x1a1a3a,0.5));
const ml=new THREE.DirectionalLight(0xffeedd,1.2);ml.position.set(8,15,5);ml.castShadow=true;
ml.shadow.mapSize.set(2048,2048);ml.shadow.camera.near=1;ml.shadow.camera.far=40;
ml.shadow.camera.left=-10;ml.shadow.camera.right=10;ml.shadow.camera.top=10;ml.shadow.camera.bottom=-10;ml.shadow.bias=-0.001;
scene.add(ml);
const rl=new THREE.DirectionalLight(0x4060ff,0.6);rl.position.set(-6,8,-8);scene.add(rl);
const rsp=new THREE.PointLight(0xff2040,2,15);rsp.position.set(-4,5,0);scene.add(rsp);
const bsp=new THREE.PointLight(0x2060ff,2,15);bsp.position.set(4,5,0);scene.add(bsp);
const ug=new THREE.PointLight(0xff8800,0.8,8);ug.position.set(0,-0.5,0);scene.add(ug);

const gnd=new THREE.Mesh(new THREE.PlaneGeometry(60,60),new THREE.MeshStandardMaterial({color:0x0c0c18,roughness:0.85,metalness:0.1}));
gnd.rotation.x=-Math.PI/2;gnd.position.y=-0.02;gnd.receiveShadow=true;scene.add(gnd);

const rng=new THREE.Mesh(new THREE.CylinderGeometry(R,R+0.3,RH,64),new THREE.MeshStandardMaterial({color:0x1a1520,roughness:0.4,metalness:0.3}));
rng.position.y=RH/2;rng.receiveShadow=true;rng.castShadow=true;scene.add(rng);
const srf=new THREE.Mesh(new THREE.CylinderGeometry(R-0.05,R-0.05,0.02,64),new THREE.MeshStandardMaterial({color:0x2a2030,roughness:0.3,metalness:0.5}));
srf.position.y=RH+0.01;srf.receiveShadow=true;scene.add(srf);
const bndMat=new THREE.MeshStandardMaterial({color:0xffcc44,emissive:0xffaa00,emissiveIntensity:0.4,roughness:0.3,metalness:0.7});
const bnd=new THREE.Mesh(new THREE.TorusGeometry(R-0.1,0.08,16,64),bndMat);
bnd.rotation.x=-Math.PI/2;bnd.position.y=RH+0.08;scene.add(bnd);
const ib=new THREE.Mesh(new THREE.TorusGeometry(2.5,0.04,12,48),new THREE.MeshStandardMaterial({color:0x665544,emissive:0x332211,emissiveIntensity:0.3}));
ib.rotation.x=-Math.PI/2;ib.position.y=RH+0.04;scene.add(ib);
for(let s of[-1,1]){const l=new THREE.Mesh(new THREE.BoxGeometry(1.2,0.02,0.06),new THREE.MeshStandardMaterial({color:0xeee,emissive:0xaaa,emissiveIntensity:0.2}));l.position.set(s*1.5,RH+0.02,0);scene.add(l);}

function mkCube(col,emC,px){
  const g=new THREE.Group();
  const geo=new THREE.BoxGeometry(CS,CS,CS);
  const mat=new THREE.MeshStandardMaterial({color:col,roughness:0.2,metalness:0.6,emissive:emC,emissiveIntensity:0.15});
  const m=new THREE.Mesh(geo,mat);m.castShadow=true;m.receiveShadow=true;g.add(m);
  const ed=new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({color:emC,transparent:true,opacity:0.6}));g.add(ed);
  for(let ey of[-0.15,0.15]){const e=new THREE.Mesh(new THREE.SphereGeometry(0.07,12,12),new THREE.MeshStandardMaterial({color:0xfff,emissive:0xfff,emissiveIntensity:1}));e.position.set(ey,0.1,CS/2+0.01);g.add(e);}
  g.position.set(px,RH+CS/2,0);scene.add(g);
  return{mesh:g,body:m,edges:ed,vx:0,vy:0,vz:0,sx:px,flash:0,x:px,z:0};
}
const cR=mkCube(0xcc1030,0xff2050,-2.5), cB=mkCube(0x1040cc,0x2060ff,2.5);
cR.mesh.rotation.y=Math.PI/2;cB.mesh.rotation.y=-Math.PI/2;
function sync(c){c.x=c.mesh.position.x;c.z=c.mesh.position.z;}

const pGeo=new THREE.SphereGeometry(0.06,6,6), tGeo=new THREE.SphereGeometry(0.04,4,4);
function spawnP(x,y,z,col,n=20,f=6){for(let i=0;i<n;i++){const mt=new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:1});const p=new THREE.Mesh(pGeo,mt);p.position.set(x,y,z);scene.add(p);const a=Math.random()*Math.PI*2,e=(Math.random()-0.3)*Math.PI,s=(Math.random()*0.7+0.3)*f;P.push({mesh:p,vx:Math.cos(a)*Math.cos(e)*s,vy:Math.sin(e)*s+3,vz:Math.sin(a)*Math.cos(e)*s,life:1,decay:0.015+Math.random()*0.025});}}
function spawnT(c,col){const sp=Math.sqrt(c.vx*c.vx+c.vz*c.vz);if(sp<1.5)return;const mt=new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.5});const p=new THREE.Mesh(tGeo,mt);p.position.set(c.mesh.position.x+(Math.random()-0.5)*0.3,c.mesh.position.y-0.3,c.mesh.position.z+(Math.random()-0.5)*0.3);scene.add(p);TP.push({mesh:p,life:1,decay:0.04});}
let sw=null;
function spawnSW(x,z){if(sw)scene.remove(sw);sw=new THREE.Mesh(new THREE.RingGeometry(0.1,0.3,32),new THREE.MeshBasicMaterial({color:0xffcc44,transparent:true,opacity:1,side:THREE.DoubleSide}));sw.rotation.x=-Math.PI/2;sw.position.set(x,RH+0.1,z);scene.add(sw);sw._s=0.5;}

function applyA(c,a){const[fx,fz]=AF[a]||[0,0];c.vx+=fx*DT;c.vz+=fz*DT;}
function collide(a,b){return Math.abs(a.mesh.position.x-b.mesh.position.x)<CS&&Math.abs(a.mesh.position.z-b.mesh.position.z)<CS;}
function resolve(a,b){const dx=b.mesh.position.x-a.mesh.position.x,dz=b.mesh.position.z-a.mesh.position.z;const d=Math.sqrt(dx*dx+dz*dz)||0.01;const nx=dx/d,nz=dz/d;const ol=CS-d;if(ol>0){a.mesh.position.x-=nx*ol*0.5;a.mesh.position.z-=nz*ol*0.5;b.mesh.position.x+=nx*ol*0.5;b.mesh.position.z+=nz*ol*0.5;}const rv=(a.vx-b.vx)*nx+(a.vz-b.vz)*nz;if(rv<=0)return;const im=rv*(1+BN);a.vx-=im*nx*0.5;a.vz-=im*nz*0.5;b.vx+=im*nx*0.5;b.vz+=im*nz*0.5;const mx=(a.mesh.position.x+b.mesh.position.x)/2,mz=(a.mesh.position.z+b.mesh.position.z)/2,f=Math.abs(rv);if(f>2){spawnP(mx,RH+CS/2,mz,0xffcc44,Math.min(30,f*4|0),f*0.8);a.flash=0.4;b.flash=0.4;camShk=Math.min(0.3,f*0.03);}}
function outR(c){const x=c.mesh.position.x,z=c.mesh.position.z;return Math.sqrt(x*x+z*z)>ROUT;}

function reset(){cR.mesh.position.set(cR.sx,RH+CS/2,0);cB.mesh.position.set(cB.sx,RH+CS/2,0);cR.vx=cR.vz=cR.vy=0;cB.vx=cB.vz=cB.vy=0;cR.mesh.rotation.y=Math.PI/2;cB.mesh.rotation.y=-Math.PI/2;sync(cR);sync(cB);frozen=false;slowMo=1;}
function ann(t,c,d=1500){const e=$('ann');e.textContent=t;e.style.color=c;e.style.opacity='1';setTimeout(()=>e.style.opacity='0',d);}
function ringOut(loser){if(frozen)return;frozen=true;spawnSW(loser.mesh.position.x,loser.mesh.position.z);spawnP(loser.mesh.position.x,RH+CS/2,loser.mesh.position.z,0xffcc44,40,10);slowMo=0.15;slowMoT=0;if(loser===cR){scores.b++;$('sb').textContent=scores.b;ann('AZURE WINS','#40a0ff');}else{scores.r++;$('sr').textContent=scores.r;ann('CRIMSON WINS','#ff4060');}round++;$('rl').textContent='ROUND '+round;if(scores.r>=WS||scores.b>=WS){const w=scores.r>=WS?'CRIMSON':'AZURE',c=scores.r>=WS?'#ff4060':'#40a0ff';setTimeout(()=>{ann(w+' CHAMPION!',c,3000);scores={r:0,b:0};round=1;$('sr').textContent='0';$('sb').textContent='0';$('rl').textContent='ROUND 1';setTimeout(reset,2500);},RD);}else setTimeout(reset,RD);}

addEventListener('keydown',e=>{K[e.key.toLowerCase()]=true;K[e.key]=true;if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space'].includes(e.code))e.preventDefault();});
addEventListener('keyup',e=>{K[e.key.toLowerCase()]=false;K[e.key]=false;});

function humanAction(wasd){
  let fx=0,fz=0;
  if(wasd){if(K.w)fz=-1;if(K.s)fz=1;if(K.a)fx=-1;if(K.d)fx=1;}
  else{if(K.ArrowUp||K.arrowup)fz=-1;if(K.ArrowDown||K.arrowdown)fz=1;if(K.ArrowLeft||K.arrowleft)fx=-1;if(K.ArrowRight||K.arrowright)fx=1;}
  if(!fx&&!fz)return 0;
  if(fx===1&&fz===0)return 1;if(fx===-1&&fz===0)return 2;
  if(fx===0&&fz===1)return 3;if(fx===0&&fz===-1)return 4;
  if(fx===1&&fz===1)return 5;if(fx===1&&fz===-1)return 6;
  if(fx===-1&&fz===1)return 7;if(fx===-1&&fz===-1)return 8;
  return 0;
}

function updCam(dt){camAng+=dt*0.08;const mx=(cR.mesh.position.x+cB.mesh.position.x)/2,mz=(cR.mesh.position.z+cB.mesh.position.z)/2;const d=Math.sqrt((cR.mesh.position.x-cB.mesh.position.x)**2+(cR.mesh.position.z-cB.mesh.position.z)**2);const z=Math.max(10,d+8);camTY=6+d*0.3;cam.position.x=mx+Math.sin(camAng)*z;cam.position.z=mz+Math.cos(camAng)*z;cam.position.y+=(camTY-cam.position.y)*0.05;if(camShk>0.001){cam.position.x+=(Math.random()-0.5)*camShk;cam.position.y+=(Math.random()-0.5)*camShk;cam.position.z+=(Math.random()-0.5)*camShk;camShk*=0.9;}cam.lookAt(mx,RH+0.5,mz);}

const clock=new THREE.Clock();
function loop(){
  requestAnimationFrame(loop);
  let rdt=Math.min(clock.getDelta(),0.05);
  if(slowMo<1){slowMoT+=rdt;if(slowMoT>0.5)slowMo=Math.min(1,slowMo+rdt*0.8);}
  const dt=rdt*slowMo, t=clock.elapsedTime;

  for(let ss=0;ss<(frozen?1:spd);ss++){
    if(!frozen){
      sync(cR);sync(cB);
      let rA=0,bA=0;
      if(mode==='aivai'){rA=aiPredict(buildObs(cR,cB));bA=aiPredict(buildObs(cB,cR));}
      else if(mode==='pvai'){rA=humanAction(true);bA=aiPredict(buildObs(cB,cR));}
      else{rA=humanAction(true);bA=humanAction(false);}

      for(let s=0;s<SS;s++){
        applyA(cR,rA);applyA(cB,bA);
        const f=Math.pow(FR,DT*60);cR.vx*=f;cR.vz*=f;cB.vx*=f;cB.vz*=f;
        if(collide(cR,cB))resolve(cR,cB);
        cR.mesh.position.x+=cR.vx*DT;cR.mesh.position.z+=cR.vz*DT;
        cB.mesh.position.x+=cB.vx*DT;cB.mesh.position.z+=cB.vz*DT;
        if(!frozen&&outR(cR)&&cR.mesh.position.y<=RH+CS/2+0.1)ringOut(cR);
        if(!frozen&&outR(cB)&&cB.mesh.position.y<=RH+CS/2+0.1)ringOut(cB);
      }
    }
  }

  [cR,cB].forEach(c=>{
    if(!(!outR(c)&&c.mesh.position.y<=RH+CS/2+0.01)&&c.mesh.position.y>-5)c.vy+=GR*dt;
    c.mesh.position.y+=c.vy*dt;
    if(!outR(c)&&c.mesh.position.y<RH+CS/2){c.mesh.position.y=RH+CS/2;c.vy=0;}
    if(c.mesh.position.y<-8){c.mesh.position.y=-8;c.vy=0;}
    const sp=Math.sqrt(c.vx*c.vx+c.vz*c.vz),sq=1+sp*0.015;
    c.mesh.scale.set(1/Math.sqrt(sq),sq,1/Math.sqrt(sq));
    if(c.flash>0){c.flash-=dt*3;c.body.material.emissiveIntensity=0.15+c.flash*2;c.edges.material.opacity=0.6+c.flash;}
    else c.body.material.emissiveIntensity=0.15+Math.sin(t*3)*0.05;
    if(!outR(c))c.mesh.position.y+=Math.sin(t*4+(c===cR?0:Math.PI))*0.02;
    if(sp>0.5){const ta=Math.atan2(c.vx,c.vz);let df=ta-c.mesh.rotation.y;while(df>Math.PI)df-=Math.PI*2;while(df<-Math.PI)df+=Math.PI*2;c.mesh.rotation.y+=df*0.12;}
  });

  spawnT(cR,0xff2050);spawnT(cB,0x2060ff);
  for(let i=P.length-1;i>=0;i--){const p=P[i];p.mesh.position.x+=p.vx*dt;p.mesh.position.y+=p.vy*dt;p.mesh.position.z+=p.vz*dt;p.vy+=GR*0.4*dt;p.life-=p.decay;p.mesh.material.opacity=p.life;p.mesh.scale.setScalar(p.life);if(p.life<=0){scene.remove(p.mesh);p.mesh.material.dispose();P.splice(i,1);}}
  for(let i=TP.length-1;i>=0;i--){const p=TP[i];p.life-=p.decay;p.mesh.material.opacity=p.life*0.5;p.mesh.scale.setScalar(p.life*0.8);if(p.life<=0){scene.remove(p.mesh);p.mesh.material.dispose();TP.splice(i,1);}}
  if(sw){sw._s+=dt*20;sw.scale.setScalar(sw._s);sw.material.opacity-=dt*2;if(sw.material.opacity<=0){scene.remove(sw);sw.material.dispose();sw=null;}}
  bndMat.emissiveIntensity=0.3+Math.sin(t*2)*0.15;
  rsp.position.x+=(cR.mesh.position.x-rsp.position.x)*0.05;rsp.position.z+=(cR.mesh.position.z-rsp.position.z)*0.05;
  bsp.position.x+=(cB.mesh.position.x-bsp.position.x)*0.05;bsp.position.z+=(cB.mesh.position.z-bsp.position.z)*0.05;
  updCam(rdt);ren.render(scene,cam);
}
loop();
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight);});
setTimeout(()=>$('badge').style.opacity='1',300);
setTimeout(()=>ann('FIGHT!','#ffcc44',1200),500);
})();
</script>
</body>
</html>'''

with open("cube-sumo-arena.html", 'w') as f:
    f.write(html)
print(f"Written cube-sumo-arena.html ({os.path.getsize('cube-sumo-arena.html')/1024:.0f} KB)")
