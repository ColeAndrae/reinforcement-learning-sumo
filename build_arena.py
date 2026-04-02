#!/usr/bin/env python3
"""Build arena v4. Includes JS heuristic bots + RL policy."""
import json, os

policy_path = os.path.join(os.getcwd(), 'models', 'policy.json')
if os.path.exists(policy_path):
    with open(policy_path) as f:
        policy_json = f.read()
    print(f"Loaded RL policy ({os.path.getsize(policy_path)/1024:.0f} KB)")
else:
    policy_json = 'null'
    print("No RL policy found — arena will use heuristic bots only")

html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cube Sumo</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;overflow:hidden;font-family:'Rajdhani',sans-serif;color:#fff;user-select:none}
canvas{display:block}
#hud{position:fixed;top:0;left:0;right:0;z-index:10;display:flex;justify-content:center;padding:16px;pointer-events:none}
.sp{display:flex;align-items:center;gap:32px;background:rgba(0,0,0,0.65);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:12px 32px}
.ps{display:flex;flex-direction:column;align-items:center;gap:1px}
.pl{font-family:'Orbitron',monospace;font-size:10px;letter-spacing:3px;text-transform:uppercase;opacity:0.5}
.pl.r{color:#ff4060}.pl.b{color:#40a0ff}
.pp{font-family:'Orbitron',monospace;font-size:38px;font-weight:900;line-height:1}
.pp.r{color:#ff4060}.pp.b{color:#40a0ff}
.mid{display:flex;flex-direction:column;align-items:center;gap:2px}
.vs{font-family:'Orbitron',monospace;font-size:13px;font-weight:700;opacity:0.2;letter-spacing:2px}
#rl{font-family:'Orbitron',monospace;font-size:9px;letter-spacing:2px;opacity:0.25}
#ann{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:20;font-family:'Orbitron',monospace;font-size:44px;font-weight:900;letter-spacing:5px;text-transform:uppercase;opacity:0;transition:opacity 0.3s;text-shadow:0 0 30px currentColor,0 0 60px currentColor;pointer-events:none}
#mp{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);z-index:10;display:flex;gap:6px;background:rgba(0,0,0,0.65);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:8px 12px;pointer-events:all;flex-wrap:wrap;justify-content:center;max-width:95vw}
.mb{font-family:'Orbitron',monospace;font-size:9px;letter-spacing:1px;padding:7px 12px;border:1px solid rgba(255,255,255,0.12);border-radius:6px;background:rgba(255,255,255,0.02);color:rgba(255,255,255,0.7);cursor:pointer;transition:all 0.2s;text-transform:uppercase;white-space:nowrap}
.mb:hover{background:rgba(255,255,255,0.06);color:#fff}
.mb.ac{background:rgba(255,200,68,0.12);border-color:rgba(255,200,68,0.4);color:#ffcc44}
#badge{position:fixed;top:72px;left:50%;transform:translateX(-50%);z-index:10;font-family:'Orbitron',monospace;font-size:8px;letter-spacing:3px;opacity:0;transition:opacity 0.5s;color:#ffcc44;text-shadow:0 0 15px rgba(255,200,68,0.4);pointer-events:none;text-transform:uppercase}
</style>
</head>
<body>
<div id="hud"><div class="sp">
  <div class="ps"><div class="pl r">Crimson</div><div class="pp r" id="sr">0</div></div>
  <div class="mid"><div id="rl">ROUND 1</div><div class="vs">VS</div></div>
  <div class="ps"><div class="pl b">Azure</div><div class="pp b" id="sb">0</div></div>
</div></div>
<div id="ann"></div>
<div id="badge"></div>
<div id="mp">
  <button class="mb ac" onclick="SM('hvh')">Bot vs Bot</button>
  <button class="mb" onclick="SM('pvh')">Player vs Bot</button>
  <button class="mb" onclick="SM('rvr')">RL vs RL</button>
  <button class="mb" onclick="SM('rvh')">RL vs Bot</button>
  <button class="mb" onclick="SM('pvr')">Player vs RL</button>
  <button class="mb" onclick="CS()">Speed: 1×</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ═══════════════════════════════════════════════════════════
// RL POLICY (null if not trained yet)
// ═══════════════════════════════════════════════════════════
const POLICY = ''' + policy_json + r''';
const HAS_RL = POLICY !== null;

function rlPredict(obs) {
  if (!HAS_RL) return heuristicAggressive(obs);
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
// EGOCENTRIC OBSERVATION — must exactly match Python _obs()
// ═══════════════════════════════════════════════════════════
function buildObs(me, opp) {
  const dx = opp.x - me.x, dz = opp.z - me.z;
  const dist = Math.sqrt(dx*dx + dz*dz) || 1e-8;
  const angleToOpp = Math.atan2(dz, dx);
  const mySpd = Math.sqrt(me.vx*me.vx + me.vz*me.vz);
  const myHdg = mySpd > 0.01 ? Math.atan2(me.vz, me.vx) : 0;
  const oppSpd = Math.sqrt(opp.vx*opp.vx + opp.vz*opp.vz);
  const oppHdg = oppSpd > 0.01 ? Math.atan2(opp.vz, opp.vx) : 0;
  const myEdge = 5.3 - (Math.sqrt(me.x*me.x + me.z*me.z) || 1e-8);
  const oppEdge = 5.3 - (Math.sqrt(opp.x*opp.x + opp.z*opp.z) || 1e-8);
  let closing = 0;
  if (dist > 0.01) closing = ((me.vx-opp.vx)*dx + (me.vz-opp.vz)*dz) / dist;
  const oppAng = Math.atan2(opp.z, opp.x);
  return [dist/11, angleToOpp/Math.PI, mySpd/15, myHdg/Math.PI, oppSpd/15, oppHdg/Math.PI, myEdge/5.5, oppEdge/5.5, closing/15, oppAng/Math.PI];
}

// ═══════════════════════════════════════════════════════════
// JS HEURISTIC BOTS — work without any RL training
// ═══════════════════════════════════════════════════════════
const PF=28, DIAG=PF/Math.SQRT2, _DT=1/60;
const FORCES=[[0,0],[PF,0],[-PF,0],[0,PF],[0,-PF],[DIAG,DIAG],[DIAG,-DIAG],[-DIAG,DIAG],[-DIAG,-DIAG]];

function angleToAction(a) {
  a = ((a % (2*Math.PI)) + 2*Math.PI) % (2*Math.PI);
  const s = Math.round(a / (Math.PI/4)) % 8;
  return [1,5,3,7,2,8,4,6][s];
}

function heuristicAggressive(obs) {
  // obs: [dist, angleToOpp, mySpd, myHdg, oppSpd, oppHdg, myEdge, oppEdge, closing, oppAng]
  const myEdge = obs[6] * 5.5;
  const angle = obs[1] * Math.PI;
  if (myEdge < 1.0) {
    // Near edge: move toward center. Center is roughly opposite of our position.
    // Since we don't have absolute position, retreat from opponent briefly
    return angleToAction(angle + Math.PI);
  }
  return angleToAction(angle);
}

function heuristicFlanker(obs) {
  const dist = obs[0] * 11;
  const angle = obs[1] * Math.PI;
  const myEdge = obs[6] * 5.5;
  if (myEdge < 1.0) return angleToAction(angle + Math.PI);
  if (dist > 2.5) return angleToAction(angle + 0.5); // circle
  return angleToAction(angle); // charge
}

function heuristicTactical(obs) {
  const dist = obs[0] * 11;
  const angle = obs[1] * Math.PI;
  const myEdge = obs[6] * 5.5;
  const oppEdge = obs[7] * 5.5;
  const closing = obs[8] * 15;

  if (myEdge < 1.2) return angleToAction(angle + Math.PI);

  // If opponent near edge, charge hard
  if (oppEdge < 1.5) return angleToAction(angle);

  // If far, approach at slight angle
  if (dist > 3.0) return angleToAction(angle + 0.3);

  // Close range: charge
  return angleToAction(angle);
}

// Randomly pick a heuristic for variety
const BOTS = [heuristicAggressive, heuristicFlanker, heuristicTactical];
let botR = heuristicAggressive, botB = heuristicTactical;
function pickBots() { botR = BOTS[Math.random()*3|0]; botB = BOTS[Math.random()*3|0]; }

// ═══════════════════════════════════════════════════════════
// GAME ENGINE
// ═══════════════════════════════════════════════════════════
(function(){
const R=5.5, ROUT=5.3, RH=0.4, CS=0.9, FR=0.92, BN=0.3, GR=-25, RD=1600, WS=5, SS=2;
let scores={r:0,b:0}, round=1, frozen=false, slowMo=1, slowMoT=0, mode='hvh', spd=1;
const K={}, P=[], TP=[];
let camShk=0, camAng=0, camTY=9;
const $=id=>document.getElementById(id);

window.SM = function(m) {
  mode = m;
  document.querySelectorAll('.mb').forEach(b=>b.classList.remove('ac'));
  event.target.classList.add('ac');
  const bg=$('badge');
  const labels = {
    hvh:'Heuristic Bots Active', pvh:'Bot Opponent (WASD)',
    rvr: HAS_RL ? 'RL Neural Network — Both' : '⚠ No RL model — using bots',
    rvh: HAS_RL ? 'RL vs Heuristic Bot' : '⚠ No RL model — using bots',
    pvr: HAS_RL ? 'RL Opponent (WASD)' : '⚠ No RL model — using bot'
  };
  bg.textContent = labels[m]; bg.style.opacity='1';
  scores={r:0,b:0};round=1;$('sr').textContent='0';$('sb').textContent='0';$('rl').textContent='ROUND 1';
  pickBots(); reset();
};
window.CS = function() {
  const s=[1,2,4]; spd=s[(s.indexOf(spd)+1)%s.length];
  event.target.textContent='Speed: '+spd+'×';
};

const scene=new THREE.Scene();scene.fog=new THREE.FogExp2(0x0a0a1a,0.032);
const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.1,100);cam.position.set(0,9,12);
const ren=new THREE.WebGLRenderer({antialias:true});ren.setSize(innerWidth,innerHeight);
ren.setPixelRatio(Math.min(devicePixelRatio,2));ren.shadowMap.enabled=true;ren.shadowMap.type=THREE.PCFSoftShadowMap;
ren.toneMapping=THREE.ACESFilmicToneMapping;ren.toneMappingExposure=1.1;ren.outputEncoding=THREE.sRGBEncoding;
document.body.appendChild(ren.domElement);

scene.add(new THREE.AmbientLight(0x1a1a3a,0.5));
const ml=new THREE.DirectionalLight(0xffeedd,1.2);ml.position.set(8,15,5);ml.castShadow=true;
ml.shadow.mapSize.set(2048,2048);ml.shadow.camera.near=1;ml.shadow.camera.far=40;
ml.shadow.camera.left=-10;ml.shadow.camera.right=10;ml.shadow.camera.top=10;ml.shadow.camera.bottom=-10;ml.shadow.bias=-0.001;scene.add(ml);
const rimL=new THREE.DirectionalLight(0x4060ff,0.6);rimL.position.set(-6,8,-8);scene.add(rimL);
const rsp=new THREE.PointLight(0xff2040,2,15);rsp.position.set(-4,5,0);scene.add(rsp);
const bsp=new THREE.PointLight(0x2060ff,2,15);bsp.position.set(4,5,0);scene.add(bsp);
const ug=new THREE.PointLight(0xff8800,0.8,8);ug.position.set(0,-0.5,0);scene.add(ug);

scene.add(Object.assign(new THREE.Mesh(new THREE.PlaneGeometry(60,60),new THREE.MeshStandardMaterial({color:0x0c0c18,roughness:0.85})),{rotation:{x:-Math.PI/2},position:{y:-0.02},receiveShadow:true}));
const rng=new THREE.Mesh(new THREE.CylinderGeometry(R,R+0.3,RH,64),new THREE.MeshStandardMaterial({color:0x1a1520,roughness:0.4,metalness:0.3}));
rng.position.y=RH/2;rng.receiveShadow=true;rng.castShadow=true;scene.add(rng);
const srf=new THREE.Mesh(new THREE.CylinderGeometry(R-0.05,R-0.05,0.02,64),new THREE.MeshStandardMaterial({color:0x2a2030,roughness:0.3,metalness:0.5}));
srf.position.y=RH+0.01;scene.add(srf);
const bMat=new THREE.MeshStandardMaterial({color:0xffcc44,emissive:0xffaa00,emissiveIntensity:0.4,roughness:0.3,metalness:0.7});
const bnd=new THREE.Mesh(new THREE.TorusGeometry(R-0.1,0.08,16,64),bMat);bnd.rotation.x=-Math.PI/2;bnd.position.y=RH+0.08;scene.add(bnd);
const ib=new THREE.Mesh(new THREE.TorusGeometry(2.5,0.04,12,48),new THREE.MeshStandardMaterial({color:0x665544,emissive:0x332211,emissiveIntensity:0.3}));
ib.rotation.x=-Math.PI/2;ib.position.y=RH+0.04;scene.add(ib);
for(let s of[-1,1]){const l=new THREE.Mesh(new THREE.BoxGeometry(1.2,0.02,0.06),new THREE.MeshStandardMaterial({color:0xeee,emissive:0xaaa,emissiveIntensity:0.2}));l.position.set(s*1.5,RH+0.02,0);scene.add(l);}

function mkC(col,em,px){
  const g=new THREE.Group(),geo=new THREE.BoxGeometry(CS,CS,CS);
  const mat=new THREE.MeshStandardMaterial({color:col,roughness:0.2,metalness:0.6,emissive:em,emissiveIntensity:0.15});
  const m=new THREE.Mesh(geo,mat);m.castShadow=m.receiveShadow=true;g.add(m);
  const ed=new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({color:em,transparent:true,opacity:0.6}));g.add(ed);
  for(let e of[-0.15,0.15]){const eye=new THREE.Mesh(new THREE.SphereGeometry(0.07,12,12),new THREE.MeshStandardMaterial({color:0xfff,emissive:0xfff,emissiveIntensity:1}));eye.position.set(e,0.1,CS/2+0.01);g.add(eye);}
  g.position.set(px,RH+CS/2,0);scene.add(g);
  return{mesh:g,body:m,edges:ed,vx:0,vy:0,vz:0,sx:px,fl:0,x:px,z:0};
}
const cR=mkC(0xcc1030,0xff2050,-2.5),cB=mkC(0x1040cc,0x2060ff,2.5);
cR.mesh.rotation.y=Math.PI/2;cB.mesh.rotation.y=-Math.PI/2;
function sync(c){c.x=c.mesh.position.x;c.z=c.mesh.position.z;}

const pG=new THREE.SphereGeometry(0.06,6,6),tG=new THREE.SphereGeometry(0.04,4,4);
function spP(x,y,z,col,n=20,f=6){for(let i=0;i<n;i++){const mt=new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:1});const p=new THREE.Mesh(pG,mt);p.position.set(x,y,z);scene.add(p);const a=Math.random()*Math.PI*2,e=(Math.random()-0.3)*Math.PI,s=(Math.random()*0.7+0.3)*f;P.push({mesh:p,vx:Math.cos(a)*Math.cos(e)*s,vy:Math.sin(e)*s+3,vz:Math.sin(a)*Math.cos(e)*s,life:1,dec:0.015+Math.random()*0.025});}}
function spT(c,col){const sp=Math.sqrt(c.vx*c.vx+c.vz*c.vz);if(sp<1.5)return;const mt=new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.5});const p=new THREE.Mesh(tG,mt);p.position.set(c.mesh.position.x+(Math.random()-0.5)*0.3,c.mesh.position.y-0.3,c.mesh.position.z+(Math.random()-0.5)*0.3);scene.add(p);TP.push({mesh:p,life:1,dec:0.04});}
let sw=null;
function spSW(x,z){if(sw)scene.remove(sw);sw=new THREE.Mesh(new THREE.RingGeometry(0.1,0.3,32),new THREE.MeshBasicMaterial({color:0xffcc44,transparent:true,opacity:1,side:THREE.DoubleSide}));sw.rotation.x=-Math.PI/2;sw.position.set(x,RH+0.1,z);scene.add(sw);sw._s=0.5;}

function applyA(c,a){const[fx,fz]=FORCES[a]||[0,0];c.vx+=fx*_DT;c.vz+=fz*_DT;}
function col(a,b){return Math.abs(a.mesh.position.x-b.mesh.position.x)<CS&&Math.abs(a.mesh.position.z-b.mesh.position.z)<CS;}
function res(a,b){const dx=b.mesh.position.x-a.mesh.position.x,dz=b.mesh.position.z-a.mesh.position.z,d=Math.sqrt(dx*dx+dz*dz)||.01,nx=dx/d,nz=dz/d,ol=CS-d;if(ol>0){a.mesh.position.x-=nx*ol*.5;a.mesh.position.z-=nz*ol*.5;b.mesh.position.x+=nx*ol*.5;b.mesh.position.z+=nz*ol*.5;}const rv=(a.vx-b.vx)*nx+(a.vz-b.vz)*nz;if(rv<=0)return;const im=rv*(1+BN);a.vx-=im*nx*.5;a.vz-=im*nz*.5;b.vx+=im*nx*.5;b.vz+=im*nz*.5;const f=Math.abs(rv);if(f>2){spP((a.mesh.position.x+b.mesh.position.x)/2,RH+CS/2,(a.mesh.position.z+b.mesh.position.z)/2,0xffcc44,Math.min(30,f*4|0),f*.8);a.fl=b.fl=.4;camShk=Math.min(.3,f*.03);}}
function outR(c){return Math.sqrt(c.mesh.position.x**2+c.mesh.position.z**2)>ROUT;}

function reset(){cR.mesh.position.set(cR.sx,RH+CS/2,0);cB.mesh.position.set(cB.sx,RH+CS/2,0);cR.vx=cR.vz=cR.vy=0;cB.vx=cB.vz=cB.vy=0;cR.mesh.rotation.y=Math.PI/2;cB.mesh.rotation.y=-Math.PI/2;sync(cR);sync(cB);frozen=false;slowMo=1;}
function ann(t,c,d=1500){const e=$('ann');e.textContent=t;e.style.color=c;e.style.opacity='1';setTimeout(()=>e.style.opacity='0',d);}
function ringOut(loser){if(frozen)return;frozen=true;spSW(loser.mesh.position.x,loser.mesh.position.z);spP(loser.mesh.position.x,RH+CS/2,loser.mesh.position.z,0xffcc44,40,10);slowMo=.15;slowMoT=0;
if(loser===cR){scores.b++;$('sb').textContent=scores.b;ann('AZURE WINS','#40a0ff');}
else{scores.r++;$('sr').textContent=scores.r;ann('CRIMSON WINS','#ff4060');}
round++;$('rl').textContent='ROUND '+round;
if(scores.r>=WS||scores.b>=WS){const w=scores.r>=WS?'CRIMSON':'AZURE',c=scores.r>=WS?'#ff4060':'#40a0ff';setTimeout(()=>{ann(w+' CHAMPION!',c,3000);scores={r:0,b:0};round=1;$('sr').textContent='0';$('sb').textContent='0';$('rl').textContent='ROUND 1';pickBots();setTimeout(reset,2500);},RD);}
else{pickBots();setTimeout(reset,RD);}}

addEventListener('keydown',e=>{K[e.key.toLowerCase()]=true;K[e.key]=true;if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.code))e.preventDefault();});
addEventListener('keyup',e=>{K[e.key.toLowerCase()]=false;K[e.key]=false;});

function human(){
  let fx=0,fz=0;
  if(K.w)fz=-1;if(K.s)fz=1;if(K.a)fx=-1;if(K.d)fx=1;
  if(!fx&&!fz)return 0;
  if(fx===1&&!fz)return 1;if(fx===-1&&!fz)return 2;
  if(!fx&&fz===1)return 3;if(!fx&&fz===-1)return 4;
  if(fx===1&&fz===1)return 5;if(fx===1&&fz===-1)return 6;
  if(fx===-1&&fz===1)return 7;if(fx===-1&&fz===-1)return 8;
  return 0;
}

function getAction(who, cube, other) {
  const obs = buildObs(cube, other);
  if (who === 'human') return human();
  if (who === 'rl') return rlPredict(obs);
  if (who === 'botR') return botR(obs);
  return botB(obs);
}

function updCam(dt){camAng+=dt*.08;const mx=(cR.mesh.position.x+cB.mesh.position.x)/2,mz=(cR.mesh.position.z+cB.mesh.position.z)/2,d=Math.sqrt((cR.mesh.position.x-cB.mesh.position.x)**2+(cR.mesh.position.z-cB.mesh.position.z)**2),z=Math.max(10,d+8);camTY=6+d*.3;cam.position.x=mx+Math.sin(camAng)*z;cam.position.z=mz+Math.cos(camAng)*z;cam.position.y+=(camTY-cam.position.y)*.05;if(camShk>.001){cam.position.x+=(Math.random()-.5)*camShk;cam.position.y+=(Math.random()-.5)*camShk;cam.position.z+=(Math.random()-.5)*camShk;camShk*=.9;}cam.lookAt(mx,RH+.5,mz);}

const clock=new THREE.Clock();
function loop(){
  requestAnimationFrame(loop);
  let rdt=Math.min(clock.getDelta(),.05);
  if(slowMo<1){slowMoT+=rdt;if(slowMoT>.5)slowMo=Math.min(1,slowMo+rdt*.8);}
  const dt=rdt*slowMo,t=clock.elapsedTime;

  for(let ss=0;ss<(frozen?1:spd);ss++){
    if(!frozen){
      sync(cR);sync(cB);
      let rW,bW;
      switch(mode){
        case 'hvh': rW='botR';bW='botB';break;
        case 'pvh': rW='human';bW='botB';break;
        case 'rvr': rW='rl';bW='rl';break;
        case 'rvh': rW='rl';bW='botB';break;
        case 'pvr': rW='human';bW='rl';break;
        default: rW='botR';bW='botB';
      }
      const rA=getAction(rW,cR,cB), bA=getAction(bW,cB,cR);
      for(let s=0;s<SS;s++){
        applyA(cR,rA);applyA(cB,bA);
        const f=Math.pow(FR,_DT*60);cR.vx*=f;cR.vz*=f;cB.vx*=f;cB.vz*=f;
        if(col(cR,cB))res(cR,cB);
        cR.mesh.position.x+=cR.vx*_DT;cR.mesh.position.z+=cR.vz*_DT;
        cB.mesh.position.x+=cB.vx*_DT;cB.mesh.position.z+=cB.vz*_DT;
        if(!frozen&&outR(cR))ringOut(cR);
        if(!frozen&&outR(cB))ringOut(cB);
      }
    }
  }

  [cR,cB].forEach(c=>{
    const on=!outR(c)&&c.mesh.position.y<=RH+CS/2+.01;
    if(!on&&c.mesh.position.y>-5)c.vy+=GR*dt;
    c.mesh.position.y+=c.vy*dt;
    if(!outR(c)&&c.mesh.position.y<RH+CS/2){c.mesh.position.y=RH+CS/2;c.vy=0;}
    if(c.mesh.position.y<-8){c.mesh.position.y=-8;c.vy=0;}
    const sp=Math.sqrt(c.vx*c.vx+c.vz*c.vz),sq=1+sp*.015;
    c.mesh.scale.set(1/Math.sqrt(sq),sq,1/Math.sqrt(sq));
    if(c.fl>0){c.fl-=dt*3;c.body.material.emissiveIntensity=.15+c.fl*2;c.edges.material.opacity=.6+c.fl;}
    else c.body.material.emissiveIntensity=.15+Math.sin(t*3)*.05;
    if(!outR(c))c.mesh.position.y+=Math.sin(t*4+(c===cR?0:Math.PI))*.02;
    if(sp>.5){const ta=Math.atan2(c.vx,c.vz);let df=ta-c.mesh.rotation.y;while(df>Math.PI)df-=2*Math.PI;while(df<-Math.PI)df+=2*Math.PI;c.mesh.rotation.y+=df*.12;}
  });

  spT(cR,0xff2050);spT(cB,0x2060ff);
  for(let i=P.length-1;i>=0;i--){const p=P[i];p.mesh.position.x+=p.vx*dt;p.mesh.position.y+=p.vy*dt;p.mesh.position.z+=p.vz*dt;p.vy+=GR*.4*dt;p.life-=p.dec;p.mesh.material.opacity=p.life;p.mesh.scale.setScalar(p.life);if(p.life<=0){scene.remove(p.mesh);p.mesh.material.dispose();P.splice(i,1);}}
  for(let i=TP.length-1;i>=0;i--){const p=TP[i];p.life-=p.dec;p.mesh.material.opacity=p.life*.5;p.mesh.scale.setScalar(p.life*.8);if(p.life<=0){scene.remove(p.mesh);p.mesh.material.dispose();TP.splice(i,1);}}
  if(sw){sw._s+=dt*20;sw.scale.setScalar(sw._s);sw.material.opacity-=dt*2;if(sw.material.opacity<=0){scene.remove(sw);sw.material.dispose();sw=null;}}
  bMat.emissiveIntensity=.3+Math.sin(t*2)*.15;
  rsp.position.x+=(cR.mesh.position.x-rsp.position.x)*.05;rsp.position.z+=(cR.mesh.position.z-rsp.position.z)*.05;
  bsp.position.x+=(cB.mesh.position.x-bsp.position.x)*.05;bsp.position.z+=(cB.mesh.position.z-bsp.position.z)*.05;
  updCam(rdt);ren.render(scene,cam);
}
loop();
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight);});
setTimeout(()=>{$('badge').textContent='Heuristic Bots Active';$('badge').style.opacity='1';},300);
setTimeout(()=>ann('FIGHT!','#ffcc44',1200),500);
})();
</script>
</body>
</html>'''

with open('cube-sumo-arena.html', 'w') as f:
    f.write(html)
print(f"Built: cube-sumo-arena.html ({os.path.getsize('cube-sumo-arena.html')/1024:.0f} KB)")
