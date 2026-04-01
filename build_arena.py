#!/usr/bin/env python3
"""Generate the final Cube Sumo HTML arena with embedded trained AI policy."""

import json
import os

# Load compact policy
with open("models/policy.json") as f:
    policy_json = f.read()

html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cube Sumo — RL Arena</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0a0f;overflow:hidden;font-family:'Rajdhani',sans-serif;color:#fff;user-select:none}
  #canvas-container{position:fixed;inset:0;z-index:0}
  canvas{display:block}
  #hud{position:fixed;top:0;left:0;right:0;z-index:10;display:flex;justify-content:center;padding:20px 30px;pointer-events:none}
  .score-panel{display:flex;align-items:center;gap:40px;background:rgba(0,0,0,0.6);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:14px 36px}
  .player-score{display:flex;flex-direction:column;align-items:center;gap:2px}
  .player-label{font-family:'Orbitron',monospace;font-size:11px;letter-spacing:3px;text-transform:uppercase;opacity:0.6}
  .player-label.red{color:#ff4060}.player-label.blue{color:#40a0ff}
  .player-points{font-family:'Orbitron',monospace;font-size:42px;font-weight:900;line-height:1}
  .player-points.red{color:#ff4060}.player-points.blue{color:#40a0ff}
  .vs-divider{font-family:'Orbitron',monospace;font-size:14px;font-weight:700;opacity:0.25;letter-spacing:2px}
  #announcement{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:20;font-family:'Orbitron',monospace;font-size:48px;font-weight:900;letter-spacing:6px;text-transform:uppercase;opacity:0;transition:opacity 0.3s;text-shadow:0 0 40px currentColor,0 0 80px currentColor;pointer-events:none}
  #round-label{font-family:'Orbitron',monospace;font-size:10px;letter-spacing:3px;opacity:0.3;text-align:center;margin-bottom:-4px}
  #mode-panel{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:10;display:flex;gap:8px;background:rgba(0,0,0,0.6);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:10px 16px;pointer-events:all}
  .mode-btn{font-family:'Orbitron',monospace;font-size:10px;letter-spacing:1px;padding:8px 16px;border:1px solid rgba(255,255,255,0.15);border-radius:8px;background:rgba(255,255,255,0.03);color:#fff;cursor:pointer;transition:all 0.2s;text-transform:uppercase}
  .mode-btn:hover{background:rgba(255,255,255,0.08);border-color:rgba(255,255,255,0.3)}
  .mode-btn.active{background:rgba(255,200,68,0.15);border-color:rgba(255,200,68,0.5);color:#ffcc44}
  #speed-label{font-family:'Orbitron',monospace;font-size:9px;letter-spacing:2px;opacity:0.4;display:flex;align-items:center;padding:0 8px}
  #ai-badge{position:fixed;top:80px;left:50%;transform:translateX(-50%);z-index:10;font-family:'Orbitron',monospace;font-size:9px;letter-spacing:3px;opacity:0;transition:opacity 0.5s;color:#ffcc44;text-shadow:0 0 20px rgba(255,200,68,0.5);pointer-events:none;text-transform:uppercase}
</style>
</head>
<body>
<div id="canvas-container"></div>
<div id="hud">
  <div class="score-panel">
    <div class="player-score">
      <div class="player-label red">Crimson</div>
      <div class="player-points red" id="score-red">0</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:4px">
      <div id="round-label">ROUND 1</div>
      <div class="vs-divider">VS</div>
    </div>
    <div class="player-score">
      <div class="player-label blue">Azure</div>
      <div class="player-points blue" id="score-blue">0</div>
    </div>
  </div>
</div>
<div id="announcement"></div>
<div id="ai-badge">Neural Network Active</div>
<div id="mode-panel">
  <button class="mode-btn active" id="btn-aivai" onclick="setMode('aivai')">AI vs AI</button>
  <button class="mode-btn" id="btn-pvai" onclick="setMode('pvai')">Player vs AI</button>
  <button class="mode-btn" id="btn-pvp" onclick="setMode('pvp')">P1 vs P2</button>
  <span id="speed-label">1×</span>
  <button class="mode-btn" id="btn-speed" onclick="cycleSpeed()">Speed</button>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ─── TRAINED POLICY WEIGHTS ─────────────────────────────
const POLICY = ''' + policy_json + r''';

// ─── AI FORWARD PASS ────────────────────────────────────
function aiPredict(obs) {
  let x = new Float64Array(obs);
  for (let i = 0; i < POLICY.layers.length; i++) {
    const W = POLICY.layers[i].weight;
    const B = POLICY.layers[i].bias;
    const outSize = W.length;
    const y = new Float64Array(outSize);
    for (let j = 0; j < outSize; j++) {
      let sum = B[j];
      for (let k = 0; k < x.length; k++) { sum += W[j][k] * x[k]; }
      y[j] = (i < POLICY.layers.length - 1) ? Math.tanh(sum) : sum;
    }
    x = y;
  }
  // argmax
  let best = 0;
  for (let i = 1; i < x.length; i++) { if (x[i] > x[best]) best = i; }
  return best;
}

// ─── BUILD OBSERVATION ──────────────────────────────────
const RING_RADIUS = 5.5;
const RING_OUT_R = RING_RADIUS - 0.2;
const MAX_SPEED = 15.0;

function buildObs(me, opp) {
  const ownDist = Math.sqrt(me.x*me.x + me.z*me.z);
  const oppDist = Math.sqrt(opp.x*opp.x + opp.z*opp.z);
  const relX = opp.x - me.x;
  const relZ = opp.z - me.z;
  const angleToOpp = Math.atan2(relZ, relX);
  const angleToCenter = Math.atan2(-me.z, -me.x);
  return [
    me.x / RING_RADIUS, me.z / RING_RADIUS,
    me.vx / MAX_SPEED, me.vz / MAX_SPEED,
    relX / (2*RING_RADIUS), relZ / (2*RING_RADIUS),
    opp.vx / MAX_SPEED, opp.vz / MAX_SPEED,
    (RING_OUT_R - ownDist) / RING_RADIUS,
    (RING_OUT_R - oppDist) / RING_RADIUS,
    angleToOpp / Math.PI, angleToCenter / Math.PI,
  ];
}

(function() {
  // ─── CONFIG ───────────────────────────────────────────
  const RING_HEIGHT = 0.4;
  const CUBE_SIZE = 0.9;
  const CUBE_MASS = 1.0;
  const PUSH_FORCE = 28;
  const FRICTION = 0.92;
  const BOUNCE = 0.3;
  const GRAVITY = -25;
  const RESET_DELAY = 1800;
  const WIN_SCORE = 5;
  const FIXED_DT = 1/60;
  const SUBSTEPS = 2;

  // ─── STATE ────────────────────────────────────────────
  let scores = {red:0, blue:0};
  let round = 1;
  let frozen = false;
  let slowMo = 1.0;
  let slowMoTimer = 0;
  let gameMode = 'aivai'; // aivai, pvai, pvp
  let speedMult = 1;
  const keys = {};
  const particles = [];
  const trailParticles = [];

  // ─── MODE CONTROL ─────────────────────────────────────
  window.setMode = function(mode) {
    gameMode = mode;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-'+mode).classList.add('active');
    const badge = document.getElementById('ai-badge');
    badge.style.opacity = (mode === 'pvp') ? '0' : '1';
    badge.textContent = mode === 'aivai' ? 'Neural Network Active — Both Agents' :
                         mode === 'pvai' ? 'Neural Network Active — Azure (WASD to play)' : '';
    scores = {red:0, blue:0};
    round = 1;
    document.getElementById('score-red').textContent = '0';
    document.getElementById('score-blue').textContent = '0';
    document.getElementById('round-label').textContent = 'ROUND 1';
    resetCubes();
  };

  window.cycleSpeed = function() {
    const speeds = [1, 2, 4];
    const idx = (speeds.indexOf(speedMult) + 1) % speeds.length;
    speedMult = speeds[idx];
    document.getElementById('speed-label').textContent = speedMult + '×';
  };

  // ─── THREE.JS SETUP ──────────────────────────────────
  const container = document.getElementById('canvas-container');
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0a0a1a, 0.035);
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 100);
  camera.position.set(0, 9, 12);
  camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.outputEncoding = THREE.sRGBEncoding;
  container.appendChild(renderer.domElement);

  // ─── LIGHTING ─────────────────────────────────────────
  scene.add(new THREE.AmbientLight(0x1a1a3a, 0.5));
  const mainLight = new THREE.DirectionalLight(0xffeedd, 1.2);
  mainLight.position.set(8,15,5);
  mainLight.castShadow = true;
  mainLight.shadow.mapSize.set(2048,2048);
  mainLight.shadow.camera.near = 1; mainLight.shadow.camera.far = 40;
  mainLight.shadow.camera.left = -10; mainLight.shadow.camera.right = 10;
  mainLight.shadow.camera.top = 10; mainLight.shadow.camera.bottom = -10;
  mainLight.shadow.bias = -0.001;
  scene.add(mainLight);
  const rimLight = new THREE.DirectionalLight(0x4060ff, 0.6);
  rimLight.position.set(-6,8,-8);
  scene.add(rimLight);
  const redSpot = new THREE.PointLight(0xff2040,2,15);
  redSpot.position.set(-4,5,0); scene.add(redSpot);
  const blueSpot = new THREE.PointLight(0x2060ff,2,15);
  blueSpot.position.set(4,5,0); scene.add(blueSpot);
  scene.add(new THREE.PointLight(0xff8800,0.8,8));

  // ─── GROUND ───────────────────────────────────────────
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(60,60),
    new THREE.MeshStandardMaterial({color:0x0c0c18,roughness:0.85,metalness:0.1})
  );
  ground.rotation.x = -Math.PI/2; ground.position.y = -0.02;
  ground.receiveShadow = true; scene.add(ground);

  // ─── RING ─────────────────────────────────────────────
  const ring = new THREE.Mesh(
    new THREE.CylinderGeometry(RING_RADIUS, RING_RADIUS+0.3, RING_HEIGHT, 64),
    new THREE.MeshStandardMaterial({color:0x1a1520,roughness:0.4,metalness:0.3})
  );
  ring.position.y = RING_HEIGHT/2; ring.receiveShadow = true; ring.castShadow = true;
  scene.add(ring);

  const surface = new THREE.Mesh(
    new THREE.CylinderGeometry(RING_RADIUS-0.05, RING_RADIUS-0.05, 0.02, 64),
    new THREE.MeshStandardMaterial({color:0x2a2030,roughness:0.3,metalness:0.5})
  );
  surface.position.y = RING_HEIGHT+0.01; surface.receiveShadow = true; scene.add(surface);

  const boundaryMat = new THREE.MeshStandardMaterial({color:0xffcc44,emissive:0xffaa00,emissiveIntensity:0.4,roughness:0.3,metalness:0.7});
  const boundary = new THREE.Mesh(new THREE.TorusGeometry(RING_RADIUS-0.1,0.08,16,64), boundaryMat);
  boundary.rotation.x = -Math.PI/2; boundary.position.y = RING_HEIGHT+0.08; scene.add(boundary);

  const innerBound = new THREE.Mesh(
    new THREE.TorusGeometry(2.5,0.04,12,48),
    new THREE.MeshStandardMaterial({color:0x665544,emissive:0x332211,emissiveIntensity:0.3,roughness:0.5,metalness:0.4})
  );
  innerBound.rotation.x = -Math.PI/2; innerBound.position.y = RING_HEIGHT+0.04; scene.add(innerBound);

  for (let s of [-1,1]) {
    const line = new THREE.Mesh(
      new THREE.BoxGeometry(1.2,0.02,0.06),
      new THREE.MeshStandardMaterial({color:0xeeeeee,emissive:0xaaaaaa,emissiveIntensity:0.2})
    );
    line.position.set(s*1.5, RING_HEIGHT+0.02, 0); scene.add(line);
  }

  // ─── CUBES ────────────────────────────────────────────
  function createCube(color, emissiveColor, x) {
    const group = new THREE.Group();
    const geo = new THREE.BoxGeometry(CUBE_SIZE,CUBE_SIZE,CUBE_SIZE);
    const mat = new THREE.MeshStandardMaterial({color,roughness:0.2,metalness:0.6,emissive:emissiveColor,emissiveIntensity:0.15});
    const mesh = new THREE.Mesh(geo,mat);
    mesh.castShadow = true; mesh.receiveShadow = true; group.add(mesh);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({color:emissiveColor,transparent:true,opacity:0.6}));
    group.add(edges);
    for (let ey of [-0.15,0.15]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.07,12,12), new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xffffff,emissiveIntensity:1}));
      eye.position.set(ey,0.1,CUBE_SIZE/2+0.01); group.add(eye);
    }
    group.position.set(x, RING_HEIGHT+CUBE_SIZE/2, 0);
    scene.add(group);
    return {mesh:group, body:mesh, edges, vx:0, vy:0, vz:0, startX:x, impactFlash:0, x, z:0};
  }

  const cubeRed = createCube(0xcc1030, 0xff2050, -2.5);
  const cubeBlue = createCube(0x1040cc, 0x2060ff, 2.5);
  cubeRed.mesh.rotation.y = Math.PI/2;
  cubeBlue.mesh.rotation.y = -Math.PI/2;

  // Sync state helper
  function syncState(cube) {
    cube.x = cube.mesh.position.x;
    cube.z = cube.mesh.position.z;
  }

  // ─── PARTICLES ────────────────────────────────────────
  const particleGeo = new THREE.SphereGeometry(0.06,6,6);
  const trailGeo = new THREE.SphereGeometry(0.04,4,4);

  function spawnParticles(x,y,z,color,count=20,force=6) {
    for (let i=0;i<count;i++) {
      const mat = new THREE.MeshBasicMaterial({color,transparent:true,opacity:1});
      const p = new THREE.Mesh(particleGeo,mat);
      p.position.set(x,y,z); scene.add(p);
      const a=Math.random()*Math.PI*2, e=(Math.random()-0.3)*Math.PI;
      const s=(Math.random()*0.7+0.3)*force;
      particles.push({mesh:p,vx:Math.cos(a)*Math.cos(e)*s,vy:Math.sin(e)*s+3,vz:Math.sin(a)*Math.cos(e)*s,life:1,decay:0.015+Math.random()*0.025});
    }
  }

  function spawnTrail(cube,color) {
    const speed=Math.sqrt(cube.vx*cube.vx+cube.vz*cube.vz);
    if(speed<1.5)return;
    const mat=new THREE.MeshBasicMaterial({color,transparent:true,opacity:0.5});
    const p=new THREE.Mesh(trailGeo,mat);
    const pos=cube.mesh.position;
    p.position.set(pos.x+(Math.random()-0.5)*0.3,pos.y-0.3,pos.z+(Math.random()-0.5)*0.3);
    scene.add(p);
    trailParticles.push({mesh:p,life:1,decay:0.04});
  }

  let shockwave = null;
  function spawnShockwave(x,z) {
    if(shockwave){scene.remove(shockwave)}
    const geo=new THREE.RingGeometry(0.1,0.3,32);
    const mat=new THREE.MeshBasicMaterial({color:0xffcc44,transparent:true,opacity:1,side:THREE.DoubleSide});
    shockwave=new THREE.Mesh(geo,mat);
    shockwave.rotation.x=-Math.PI/2;
    shockwave.position.set(x,RING_HEIGHT+0.1,z);
    scene.add(shockwave);shockwave._scale=0.5;
  }

  // ─── PHYSICS ──────────────────────────────────────────
  const ACTION_FORCES = [[0,0],[PUSH_FORCE,0],[-PUSH_FORCE,0],[0,PUSH_FORCE],[0,-PUSH_FORCE]];
  let cameraShake = 0;

  function applyAction(cube, action) {
    const [fx,fz] = ACTION_FORCES[action] || [0,0];
    cube.vx += fx / CUBE_MASS * FIXED_DT;
    cube.vz += fz / CUBE_MASS * FIXED_DT;
  }

  function cubesCollide(a,b) {
    return Math.abs(a.mesh.position.x-b.mesh.position.x)<CUBE_SIZE && Math.abs(a.mesh.position.z-b.mesh.position.z)<CUBE_SIZE;
  }

  function resolveCollision(a,b) {
    const dx=b.mesh.position.x-a.mesh.position.x;
    const dz=b.mesh.position.z-a.mesh.position.z;
    const dist=Math.sqrt(dx*dx+dz*dz)||0.01;
    const nx=dx/dist, nz=dz/dist;
    const overlap=CUBE_SIZE-dist;
    if(overlap>0){
      a.mesh.position.x-=nx*overlap*0.5; a.mesh.position.z-=nz*overlap*0.5;
      b.mesh.position.x+=nx*overlap*0.5; b.mesh.position.z+=nz*overlap*0.5;
    }
    const relV=(a.vx-b.vx)*nx+(a.vz-b.vz)*nz;
    if(relV<=0)return;
    const impulse=relV*(1+BOUNCE);
    a.vx-=impulse*nx*0.5; a.vz-=impulse*nz*0.5;
    b.vx+=impulse*nx*0.5; b.vz+=impulse*nz*0.5;
    const midX=(a.mesh.position.x+b.mesh.position.x)/2;
    const midZ=(a.mesh.position.z+b.mesh.position.z)/2;
    const impactForce=Math.abs(relV);
    if(impactForce>2){
      spawnParticles(midX,RING_HEIGHT+CUBE_SIZE/2,midZ,0xffcc44,Math.min(30,Math.floor(impactForce*4)),impactForce*0.8);
      a.impactFlash=0.4; b.impactFlash=0.4;
      cameraShake=Math.min(0.3,impactForce*0.03);
    }
  }

  function isOutOfRing(cube) {
    const x=cube.mesh.position.x, z=cube.mesh.position.z;
    return Math.sqrt(x*x+z*z) > RING_OUT_R;
  }

  // ─── GAME LOGIC ───────────────────────────────────────
  function resetCubes() {
    cubeRed.mesh.position.set(cubeRed.startX,RING_HEIGHT+CUBE_SIZE/2,0);
    cubeBlue.mesh.position.set(cubeBlue.startX,RING_HEIGHT+CUBE_SIZE/2,0);
    cubeRed.vx=cubeRed.vz=cubeRed.vy=0; cubeBlue.vx=cubeBlue.vz=cubeBlue.vy=0;
    cubeRed.mesh.rotation.y=Math.PI/2; cubeBlue.mesh.rotation.y=-Math.PI/2;
    syncState(cubeRed); syncState(cubeBlue);
    frozen=false; slowMo=1.0;
  }

  function showAnnouncement(text,color,duration=1500) {
    const el=document.getElementById('announcement');
    el.textContent=text; el.style.color=color; el.style.opacity='1';
    setTimeout(()=>{el.style.opacity='0'},duration);
  }

  function ringOut(loser) {
    if(frozen)return; frozen=true;
    spawnShockwave(loser.mesh.position.x,loser.mesh.position.z);
    spawnParticles(loser.mesh.position.x,RING_HEIGHT+CUBE_SIZE/2,loser.mesh.position.z,0xffcc44,40,10);
    slowMo=0.15; slowMoTimer=0;
    if(loser===cubeRed){scores.blue++;document.getElementById('score-blue').textContent=scores.blue;showAnnouncement('AZURE WINS','#40a0ff');}
    else{scores.red++;document.getElementById('score-red').textContent=scores.red;showAnnouncement('CRIMSON WINS','#ff4060');}
    round++;document.getElementById('round-label').textContent='ROUND '+round;
    if(scores.red>=WIN_SCORE||scores.blue>=WIN_SCORE){
      const w=scores.red>=WIN_SCORE?'CRIMSON':'AZURE';
      const c=scores.red>=WIN_SCORE?'#ff4060':'#40a0ff';
      setTimeout(()=>{showAnnouncement(w+' CHAMPION!',c,3000);scores={red:0,blue:0};round=1;
        document.getElementById('score-red').textContent='0';document.getElementById('score-blue').textContent='0';
        document.getElementById('round-label').textContent='ROUND 1';setTimeout(resetCubes,2500);},RESET_DELAY);
    } else { setTimeout(resetCubes,RESET_DELAY); }
  }

  // ─── INPUT ────────────────────────────────────────────
  window.addEventListener('keydown',e=>{keys[e.key.toLowerCase()]=true;keys[e.key]=true;
    if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space'].includes(e.code))e.preventDefault();});
  window.addEventListener('keyup',e=>{keys[e.key.toLowerCase()]=false;keys[e.key]=false;});

  function getHumanAction(wasd) {
    if(wasd) {
      if(keys['w']) return 4; if(keys['s']) return 3;
      if(keys['a']) return 2; if(keys['d']) return 1;
    } else {
      if(keys['ArrowUp']) return 4; if(keys['ArrowDown']) return 3;
      if(keys['ArrowLeft']) return 2; if(keys['ArrowRight']) return 1;
    }
    return 0;
  }

  // ─── AI TICK RATE ─────────────────────────────────────
  let aiTickCounter = 0;
  const AI_TICK_EVERY = 3; // AI decides every 3 physics frames
  let lastRedAction = 0, lastBlueAction = 0;

  // ─── CAMERA ───────────────────────────────────────────
  let cameraAngle = 0;
  let targetCamY = 9;

  function updateCamera(dt) {
    cameraAngle += dt * 0.08;
    const midX=(cubeRed.mesh.position.x+cubeBlue.mesh.position.x)/2;
    const midZ=(cubeRed.mesh.position.z+cubeBlue.mesh.position.z)/2;
    const dist=Math.sqrt(Math.pow(cubeRed.mesh.position.x-cubeBlue.mesh.position.x,2)+Math.pow(cubeRed.mesh.position.z-cubeBlue.mesh.position.z,2));
    const zoom=Math.max(10,dist+8);
    targetCamY=6+dist*0.3;
    camera.position.x=midX+Math.sin(cameraAngle)*zoom;
    camera.position.z=midZ+Math.cos(cameraAngle)*zoom;
    camera.position.y+=(targetCamY-camera.position.y)*0.05;
    if(cameraShake>0.001){
      camera.position.x+=(Math.random()-0.5)*cameraShake;
      camera.position.y+=(Math.random()-0.5)*cameraShake;
      camera.position.z+=(Math.random()-0.5)*cameraShake;
      cameraShake*=0.9;
    }
    camera.lookAt(midX,RING_HEIGHT+0.5,midZ);
  }

  // ─── MAIN LOOP ───────────────────────────────────────
  const clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    let rawDt=clock.getDelta();
    rawDt=Math.min(rawDt,0.05);
    if(slowMo<1){slowMoTimer+=rawDt;if(slowMoTimer>0.5)slowMo=Math.min(1,slowMo+rawDt*0.8);}
    const dt=rawDt*slowMo;
    const time=clock.elapsedTime;

    // Run multiple sim steps for speed multiplier
    const simSteps = frozen ? 1 : speedMult;
    for (let ss = 0; ss < simSteps; ss++) {

      if (!frozen) {
        // Sync positions to state
        syncState(cubeRed); syncState(cubeBlue);

        // Decide actions
        aiTickCounter++;
        if (aiTickCounter >= AI_TICK_EVERY) {
          aiTickCounter = 0;

          // Red agent
          if (gameMode === 'aivai') {
            const obsRed = buildObs(cubeRed, cubeBlue);
            lastRedAction = aiPredict(obsRed);
          } else if (gameMode === 'pvai' || gameMode === 'pvp') {
            lastRedAction = getHumanAction(true);
          }

          // Blue agent
          if (gameMode === 'pvp') {
            lastBlueAction = getHumanAction(false);
          } else {
            const obsBlue = buildObs(cubeBlue, cubeRed);
            lastBlueAction = aiPredict(obsBlue);
          }
        }

        // Physics substeps
        for (let sub = 0; sub < SUBSTEPS; sub++) {
          applyAction(cubeRed, lastRedAction);
          applyAction(cubeBlue, lastBlueAction);

          // Friction
          const f = Math.pow(FRICTION, FIXED_DT*60);
          cubeRed.vx*=f; cubeRed.vz*=f; cubeBlue.vx*=f; cubeBlue.vz*=f;

          // Collision
          if (cubesCollide(cubeRed,cubeBlue)) resolveCollision(cubeRed,cubeBlue);

          // Integrate
          cubeRed.mesh.position.x+=cubeRed.vx*FIXED_DT;
          cubeRed.mesh.position.z+=cubeRed.vz*FIXED_DT;
          cubeBlue.mesh.position.x+=cubeBlue.vx*FIXED_DT;
          cubeBlue.mesh.position.z+=cubeBlue.vz*FIXED_DT;

          // Ring out
          if(!frozen && isOutOfRing(cubeRed) && cubeRed.mesh.position.y<=RING_HEIGHT+CUBE_SIZE/2+0.1) ringOut(cubeRed);
          if(!frozen && isOutOfRing(cubeBlue) && cubeBlue.mesh.position.y<=RING_HEIGHT+CUBE_SIZE/2+0.1) ringOut(cubeBlue);
        }
      }

    } // end speed loop

    // Gravity for fallen cubes
    [cubeRed,cubeBlue].forEach(c => {
      const onRing = !isOutOfRing(c) && c.mesh.position.y <= RING_HEIGHT+CUBE_SIZE/2+0.01;
      if(!onRing && c.mesh.position.y > -5) c.vy += GRAVITY*dt;
      c.mesh.position.y += c.vy*dt;
      if(!isOutOfRing(c) && c.mesh.position.y < RING_HEIGHT+CUBE_SIZE/2){
        c.mesh.position.y=RING_HEIGHT+CUBE_SIZE/2; c.vy=0;
      }
      if(c.mesh.position.y<-8){c.mesh.position.y=-8;c.vy=0;}
    });

    // Visuals
    [cubeRed,cubeBlue].forEach(c => {
      const speed=Math.sqrt(c.vx*c.vx+c.vz*c.vz);
      const sq=1+speed*0.015;
      c.mesh.scale.set(1/Math.sqrt(sq),sq,1/Math.sqrt(sq));
      if(c.impactFlash>0){c.impactFlash-=dt*3;c.body.material.emissiveIntensity=0.15+c.impactFlash*2;c.edges.material.opacity=0.6+c.impactFlash;}
      else{c.body.material.emissiveIntensity=0.15+Math.sin(time*3)*0.05;}
      if(!isOutOfRing(c))c.mesh.position.y+=Math.sin(time*4+(c===cubeRed?0:Math.PI))*0.02;
      if(speed>0.5){
        const ta=Math.atan2(c.vx,c.vz);let diff=ta-c.mesh.rotation.y;
        while(diff>Math.PI)diff-=Math.PI*2;while(diff<-Math.PI)diff+=Math.PI*2;
        c.mesh.rotation.y+=diff*0.12;
      }
    });

    // Particles
    spawnTrail(cubeRed,0xff2050); spawnTrail(cubeBlue,0x2060ff);
    for(let i=particles.length-1;i>=0;i--){
      const p=particles[i];
      p.mesh.position.x+=p.vx*dt;p.mesh.position.y+=p.vy*dt;p.mesh.position.z+=p.vz*dt;
      p.vy+=GRAVITY*0.4*dt;p.life-=p.decay;p.mesh.material.opacity=p.life;p.mesh.scale.setScalar(p.life);
      if(p.life<=0){scene.remove(p.mesh);p.mesh.material.dispose();particles.splice(i,1);}
    }
    for(let i=trailParticles.length-1;i>=0;i--){
      const p=trailParticles[i];p.life-=p.decay;p.mesh.material.opacity=p.life*0.5;p.mesh.scale.setScalar(p.life*0.8);
      if(p.life<=0){scene.remove(p.mesh);p.mesh.material.dispose();trailParticles.splice(i,1);}
    }
    if(shockwave){shockwave._scale+=dt*20;shockwave.scale.setScalar(shockwave._scale);shockwave.material.opacity-=dt*2;
      if(shockwave.material.opacity<=0){scene.remove(shockwave);shockwave.material.dispose();shockwave=null;}}

    boundary.material.emissiveIntensity=0.3+Math.sin(time*2)*0.15;
    redSpot.position.x+=(cubeRed.mesh.position.x-redSpot.position.x)*0.05;
    redSpot.position.z+=(cubeRed.mesh.position.z-redSpot.position.z)*0.05;
    blueSpot.position.x+=(cubeBlue.mesh.position.x-blueSpot.position.x)*0.05;
    blueSpot.position.z+=(cubeBlue.mesh.position.z-blueSpot.position.z)*0.05;

    updateCamera(rawDt);
    renderer.render(scene,camera);
  }

  animate();

  window.addEventListener('resize',()=>{
    camera.aspect=window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth,window.innerHeight);
  });

  // Show AI badge on load
  setTimeout(()=>{document.getElementById('ai-badge').style.opacity='1';},300);
  setTimeout(()=>showAnnouncement('FIGHT!','#ffcc44',1200),500);
})();
</script>
</body>
</html>'''

# Write output
output_path = "cube-sumo-arena.html"
with open(output_path, 'w') as f:
    f.write(html)

print(f"Written to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.0f} KB")
