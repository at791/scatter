/* ---------- serial-counting network (Mante-style rate RNN, trained with a free response) ----------
   One pool of 300 tanh units, dt 20 ms, each unit with its own time constant (half of them 100 ms, half 1-5 s):
   x_k <- (1 - a_k) x_k + a_k (J tanh x + B u + c)_k, a_k = dt / tau_k.  Played here without noise.
   Inputs: a 60 ms onset pulse (amplitude 2) when it lands on a stone (blue / dark / hidden), the number, and "scan complete".
   Outputs: three units (rest dark / rest blue / can't tell) that stay near 0 until it has an answer.
   It works one circle at a time, looking at one stone at a time with short gaps, like a player counting. */
const CN_N=300,CN_IN=5,CN_OUT=3,CN_POST=50,CN_THR=0.5,CN_PULSE=3,CN_AMP=2;
const CN_SHAPES={J:[CN_N,CN_N],B:[CN_N,CN_IN],c:[CN_N],W:[CN_OUT,CN_N],cz:[CN_OUT],alpha:[CN_N]};
function cnUnpack(b64){const bin=atob(b64),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);
  const f=new Float32Array(u.buffer),W={};let o=0;for(const k in CN_SHAPES){const len=CN_SHAPES[k].reduce((a,b)=>a*b,1);W[k]=f.slice(o,o+len);o+=len;}return W;}
const CN_WEIGHTS="@@WEIGHTS@@";
const CN_W=cnUnpack(CN_WEIGHTS);
const CN_LAND=@@LAND@@;   // one plane (mean, two axes; count_landscape_export.py) through the count states of all numbers: where the page finds the nearest trough point
const CN_TROUGH=@@TROUGH@@;   // per number (count_trough_export.py): the trough unrolled x the direction the state takes to its end-of-scan verdict; speed, flow and the three verdict regions
const CN_CROSS=@@CROSS@@;   // per number (count_cross_export.py): the trough's cross-section through each count state, spanned by the trace a hidden stone leaves (h) and the push of the dark input (e); speed and flow there
function cnInv3(G){const[a,b,c]=G[0],[d,e,f]=G[1],[g,h,i]=G[2],A=e*i-f*h,B=f*g-d*i,C=d*h-e*g,t=a*A+b*B+c*C;
  return[[A/t,(c*h-b*i)/t,(b*f-c*e)/t],[B/t,(a*i-c*g)/t,(c*d-a*f)/t],[C/t,(b*g-a*h)/t,(a*e-b*d)/t]];}
const cnAx={};
function cnAxes(num){const key=String(Math.min(15,Math.max(1,num)));if(cnAx[key])return cnAx[key];   // per number: the off-trough axes of the two sheets, their images on the shared plane, their Gram inverse
  const R=CN_TROUGH.nums[key],Q=CN_CROSS.nums[key],A=[Q.h,R.d,Q.e],dot=(u,v)=>{let s=0;for(let k=0;k<CN_N;k++)s+=u[k]*v[k];return s;};
  return cnAx[key]={R,Q,PA:A.map(v=>[dot(CN_LAND.P[0],v),dot(CN_LAND.P[1],v)]),Gi:cnInv3(A.map(u=>A.map(v=>dot(u,v))))};}
function cnTrough(x,pc,num){const A=cnAxes(num),R=A.R,Q=A.Q,PA=A.PA;let xh=0,xd=0,xe=0;   // -> [arc position, offset toward the answers, nearest trough point, hidden trace, dark push]
  for(let k=0;k<CN_N;k++){xh+=x[k]*Q.h[k];xd+=x[k]*R.d[k];xe+=x[k]*Q.e[k];}
  const near=(px,py)=>{let jb=0,db=1e9;for(let j=0;j<R.cx.length;j++){const dx=px-R.cx[j],dy=py-R.cy[j],e=dx*dx+dy*dy;if(e<db){db=e;jb=j;}}return jb;};
  let jb=near(pc[0],pc[1]);
  for(let it=0;it<2;it++){const b=[xh-Q.ch[jb],xd-R.cd[jb],xe-Q.ce[jb]],c=A.Gi.map(r=>r[0]*b[0]+r[1]*b[1]+r[2]*b[2]);   // nearest trough point after taking out the state's displacement along
    jb=near(pc[0]-c[0]*PA[0][0]-c[1]*PA[1][0]-c[2]*PA[2][0],pc[1]-c[0]*PA[0][1]-c[1]*PA[1][1]-c[2]*PA[2][1]);}            // the off-trough axes, so hidden stones and the answer push do not slide it along
  return[R.s[jb],xd-R.cd[jb],jb,xh-Q.ch[jb],xe-Q.ce[jb]];}
const CN_R3=@@RIB3D@@;   // 3-D view (count_rib3d_export.py): PC1-3 of all count states; per number its trough and a ribbon along it
function cn3(x){const M=CN_R3.mu,P=CN_R3.P,o=[0,0,0];for(let k=0;k<CN_N;k++){const d=x[k]-M[k];o[0]+=d*P[0][k];o[1]+=d*P[1][k];o[2]+=d*P[2][k];}return o;}
// Unit groups found in the trained network: each unit's end-of-scan activity on real positions was fit as a plane
// in (blues still needed) x (hidden stones). Need-type units lie within 20 degrees of the need axis, slack-type
// units within 35-60 degrees towards need - hidden. The signs align each group so its mean rises with its variable.
const CN_NEED=[0,5,7,9,13,18,19,26,27,30,36,39,50,51,60,61,63,70,72,76,79,81,84,89,94,95,99,104,106,107,113,116,121,127,128,129,133,140,144,153,162,171,173,177,179,180,181,183,187,192,196,197,198,199,200,208,211,213,221,225,228,231,237,242,246,249,251,252,253,257,260,264,267,268,271,276,278,288,289,293,297];
const CN_NEED_S=[1,1,-1,1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,1,-1,-1,1,-1,1,1,1,-1,1,-1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,-1,1,1,-1];
const CN_SLACK=[24,47,55,105,112,120,126,131,143,148,159,175,188,189,193,203,204,220,240,247,262,286,287],CN_SLACK_S=[1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1];
const CN_REF={need:[[0,-0.56],[1,-0.008],[2,0.094],[3,0.143],[4,0.175],[5,0.205]],slack:[[-4,-0.503],[-3,-0.459],[-2,-0.394],[-1,-0.28],[0,0.036]]};  // where real positions with each tally sit, end of scan
// one 20 ms step; u = [blue, dark, hidden, number/10, scan complete]. Returns the outputs and the two group means.
function cnStep(S,u){const J=CN_W.J,B=CN_W.B,AL=CN_W.alpha,x=S.x,r=S.r,nx=S.nx;
  for(let k=0;k<CN_N;k++)r[k]=Math.tanh(x[k]);
  for(let k=0;k<CN_N;k++){let a=CN_W.c[k];for(let i=0;i<CN_IN;i++)a+=B[k*CN_IN+i]*u[i];const row=k*CN_N;for(let j=0;j<CN_N;j++)a+=J[row+j]*r[j];const al=AL[k];nx[k]=(1-al)*x[k]+al*a;}
  x.set(nx);for(let k=0;k<CN_N;k++)r[k]=Math.tanh(x[k]);
  const z=[0,0,0];for(let o=0;o<CN_OUT;o++){let s=CN_W.cz[o];for(let k=0;k<CN_N;k++)s+=CN_W.W[o*CN_N+k]*r[k];z[o]=s;}
  let pn=0,ps=0;for(let i=0;i<CN_NEED.length;i++)pn+=CN_NEED_S[i]*r[CN_NEED[i]];for(let i=0;i<CN_SLACK.length;i++)ps+=CN_SLACK_S[i]*r[CN_SLACK[i]];
  let pa=0,pb=0;const MU=CN_LAND.mu,P0=CN_LAND.P[0],P1=CN_LAND.P[1];for(let k=0;k<CN_N;k++){const d=x[k]-MU[k];pa+=d*P0[k];pb+=d*P1[k];}   // the state on the landscape plane
  return{z,pn:pn/CN_NEED.length,ps:ps/CN_SLACK.length,pc:[pa,pb]};}

/* ---------- the AI player ----------
   It picks a number whose whole circle is on screen, looks at the circle's stones one by one (200-300 ms each,
   short gaps), and marks the hidden stones as soon as an output crosses 0.5. "Can't tell" moves it on to the
   next number; it comes back once that circle has changed. A wrong mark means the stone is the other colour,
   so it fixes that at once and looks again. The weights are fixed: nothing is learned during play. */
const AI_MS=20,AI_STEP_MS=240,AI_GUESS_AFTER=6,AI_NAMES=["rest dark","rest blue","can't tell","no answer"];
let aiSpeed=1,aiPaused=false,aiAcc=0,aiFwd=false;   // playback: 1 = real time (one 20 ms network step per 20 ms); slower spreads the steps out; the Step buttons pause it and move a stone at a time (aiFwd: stepping on into the next circle)
let aiOn=false,aiLayer=null,aiTrial=null,aiNext=0,aiSkip=new Map(),aiLast=new Map(),aiIdle=0,aiBusy=false;
const aiRnd=(a,b)=>a+Math.floor(Math.random()*(b-a+1)),aiKind=id=>vis(id)?(P.blue[id]?0:1):2;
function aiSig(c){return P.reg[c].map(j=>vis(j)?(P.blue[j]?"b":"d"):G.unl[j]?"h":"x").join("");}
function aiClick(i,v0,v1){return !!G.unl[i]&&!G.revealed[i]&&P.x[i]>=v0+RS&&P.x[i]<=v1-RS;}
function aiClues(){const v=viewUnits();if(!v)return[];const[v0,v1]=v;
  return P.all.filter(c=>P.lvl[c]>0&&readable(c)&&P.x[c]-P.R[c]>=v0-RS&&P.x[c]+P.R[c]<=v1+RS&&P.reg[c].some(j=>j!==c&&aiClick(j,v0,v1))).sort((a,b)=>P.x[a]-P.x[b]);}
function aiStart(c){
  const ids=P.reg[c].filter(j=>j!==c),ang=j=>(Math.atan2(P.y[j]-P.y[c],P.x[j]-P.x[c])+Math.PI/2+2*Math.PI)%(2*Math.PI);ids.sort((a,b)=>ang(a)-ang(b));   // round the circle from the top
  const angs=ids.map(ang);   // each stone's angle from the top, clockwise: the board shading slices the circle by these
  const seq=[];for(let k=aiRnd(10,15);k>0;k--)seq.push(-1);
  ids.forEach((id,j)=>{for(let k=aiRnd(10,15);k>0;k--)seq.push(j);for(let k=aiRnd(0,8);k>0;k--)seq.push(-1);});
  aiTrial={c,ids,num:P.num[c],seq,done:seq.length,L:seq.length+CN_POST,t:0,sig:aiSig(c),S:{x:new Float32Array(CN_N),r:new Float32Array(CN_N),nx:new Float32Array(CN_N)},z:[],pn:[],ps:[],pc:[],tq:[],p3:[],sp:[],mk:[],u:[],seen:[],fix:-1,verdict:-1,angs,on:-9,now:null};
  aiPrecompute(aiTrial);aiNote("RNN: counting the "+P.num[c]+"'s circle");aiDrawRing();vzTrial(aiTrial);}
function aiInput(T,t){   // the network's input at step t (same rule as live play): number, scan complete, a 3-step onset pulse per fixation
  const j=t<T.done?T.seq[t]:-1,u=[0,0,0,T.num/10,t>=T.done?1:0];let on=-9;
  if(j>=0){on=t;while(on>0&&T.seq[on-1]===j)on--;if(t-on<CN_PULSE)u[aiKind(T.ids[j])]=CN_AMP;}return u;}
function cnSpeed(S,u){   // speed of the dynamics at the current state with the tonic inputs only (no stone pulse): the trail colour, as in the mockup
  const J=CN_W.J,B=CN_W.B,AL=CN_W.alpha,x=S.x,r=S.r;let s2=0;for(let k=0;k<CN_N;k++)r[k]=Math.tanh(x[k]);
  for(let k=0;k<CN_N;k++){let a=CN_W.c[k];for(let i=0;i<CN_IN;i++)a+=B[k*CN_IN+i]*u[i];const row=k*CN_N;for(let j=0;j<CN_N;j++)a+=J[row+j]*r[j];const d=AL[k]*(a-x[k]);s2+=d*d;}
  return Math.max(0,Math.min(1,(0.5*Math.log10(s2+1e-18)+1.8)/2.7));}
const vzSpCol=s=>"rgb(255,"+Math.round(255*(1-s))+",0)";   // autumn: fast = red, slow = yellow
function aiPrecompute(T){   // run the whole circle once, up front; playback then shows it step by step like a video, with no compute stalls
  const S={x:new Float32Array(CN_N),r:new Float32Array(CN_N),nx:new Float32Array(CN_N)};T.pre=[];
  for(let t=0;t<T.L;t++){const u=aiInput(T,t),o=cnStep(S,u);T.pre.push({o,u,tq:cnTrough(S.x,o.pc,T.num),p3:cn3(S.x),sp:cnSpeed(S,[0,0,0,u[3],u[4]])});const m=Math.max(o.z[0],o.z[1],o.z[2]);if(m>CN_THR)break;}
  const n=T.pre.length,s=[T.seq.indexOf(0)];   // where the Step buttons stop: the number on (just before the first stone), the end of each stone's look, the answer
  for(let t=0;t<T.done;t++)if(T.seq[t]>=0&&T.seq[t+1]!==T.seq[t])s.push(t+1);
  s.push(n);T.stops=s.filter((v,k)=>v>0&&v<=n&&s.indexOf(v)===k);}
const aiStale=T=>T.sig!==aiSig(T.c)||!readable(T.c);   // the circle changed under it (a cascade, or a mark made while paused): start over
function aiAdvance(T){   // one precomputed 20 ms step (aiPrecompute): the bookkeeping only, no drawing
  const j=T.t<T.done?T.seq[T.t]:-1;T.fix=j;
  if(j>=0){if(T.t===0||T.seq[T.t-1]!==j){T.on=T.t;T.mk.push([T.t,aiKind(T.ids[j])]);}if(!T.seen.includes(j))T.seen.push(j);}   // onset pulse: 3 steps at the start of each fixation, nothing while it dwells
  const F=T.pre[Math.min(T.t,T.pre.length-1)],o=F.o,u=F.u;
  T.z.push(o.z);T.pn.push(o.pn);T.ps.push(o.ps);T.pc.push(o.pc);T.tq.push(F.tq);T.p3.push(F.p3);T.sp.push(F.sp);T.u.push(u.slice());T.t++;
  const m=Math.max(o.z[0],o.z[1],o.z[2]);
  if(m>CN_THR)T.verdict=o.z.indexOf(m);else if(T.t>=T.L)T.verdict=3;}
function aiTick(){const T=aiTrial;
  if(aiStale(T)){aiTrial=null;aiDrawRing();return;}
  if(T.verdict>=0){aiAct();return;}   // an answer the Step buttons stopped on: act on it now
  const f=T.fix;aiAdvance(T);if(T.fix!==f)aiDrawRing();
  if(T.now)T.now.setAttribute("fill-opacity",T.t-T.on<=CN_PULSE?0.55:0.12);   // the lit slice fades once the onset pulse is over
  vzDraw(T);   // every step is drawn: the trial is precomputed, so a tick is only drawing
  if(T.verdict>=0&&!aiPaused)aiAct();}
function aiRewind(T){T.t=0;T.on=-9;T.fix=-1;T.verdict=-1;for(const k of["z","pn","ps","pc","tq","p3","sp","u","mk","seen"])T[k]=[];}
function aiSeek(T,t){   // show the circle as it stood after t steps: replay the precomputed steps without drawing, then draw once
  if(t<T.t){aiRewind(T);vzTrial(T);}   // backwards: the circle's panels start over
  while(T.t<t)aiAdvance(T);
  aiDrawRing();vzDraw(T);
  const v=T.verdict;aiNote(v>=0?"RNN: "+AI_NAMES[v]+" for the "+T.num+(v<2?" · Step ▶ marks its stones":" · Step ▶ moves on"):"RNN: counting the "+T.num+"'s circle");}
const aiNextStop=T=>T.stops.find(s=>s>T.t),aiPrevStop=T=>{let p=null;for(const s of T.stops)if(s<T.t)p=s;return p;};
function aiMark(id,blue,after){   // a wrong mark means the stone is the other colour: fix it at once
  aiBusy=true;const p=[P.x[id],P.y[id]],ok=revealAs(id,blue,p);
  setTimeout(()=>{if(!ok&&aiOn&&G&&!G.won&&!G.revealed[id])revealAs(id,!blue,p);aiBusy=false;after(ok);},AI_STEP_MS);}
function aiAct(){const T=aiTrial,v=T.verdict,vw=viewUnits(),last=T.z[T.z.length-1];aiTrial=null;
  aiLast.set(T.c,{z0:last[0],z1:last[1]});
  aiNote("RNN: "+AI_NAMES[v]+" for the "+T.num+(T.t<T.done?" (decided before the end of the circle)":""));
  if(v>=2){aiSkip.set(T.c,T.sig);aiIdle++;aiNext=performance.now()+500;aiDrawRing();return;}
  const targets=vw?T.ids.filter(id=>aiClick(id,vw[0],vw[1])):[];aiIdle=0;
  targets.forEach(id=>el("circle",{class:"pick","data-id":id,cx:f1(P.x[id]),cy:f1(P.y[id]),r:f1(RS+6),stroke:v===1?"var(--blue-out)":"var(--ink-out)"},aiLayer));
  let k=0;const go=ok=>{if(!aiOn||!G||G.won)return;
    if(!ok||k>=targets.length){aiNext=performance.now()+400;if(!ok)aiNote("RNN: wrong - that one is the other colour; looking again");aiDrawRing();return;}   // wrong: stop, the circle has changed
    const id=targets[k++];if(G.revealed[id]||!G.unl[id]){go(true);return;}
    const r=aiLayer&&aiLayer.querySelector('[data-id="'+id+'"]');if(r)r.remove();aiMark(id,v===1,go);};
  aiBusy=true;setTimeout(()=>{aiBusy=false;go(true);},AI_STEP_MS);}   // busy from the answer on, so no circle starts before the marks are made
function aiGuess(cl){   // every circle in view says "can't tell" on its own, yet a move exists: go with the least undecided circle
  const vw=viewUnits();let best=null,bm=-1,blue=false;
  for(const c of cl){const l=aiLast.get(c);if(!l)continue;const m=Math.max(l.z0,l.z1);if(m>bm){bm=m;best=c;blue=l.z1>l.z0;}}
  if(best==null)best=cl[0];const id=P.reg[best].find(j=>j!==best&&aiClick(j,vw[0],vw[1]));if(id==null)return;
  aiNote("RNN: guessing");aiIdle=0;aiMark(id,blue,()=>{aiNext=performance.now()+400;});}
function aiLoop(){
  if(!aiOn||!G||!P||G.won||generating||G.anim||demoOn||replay||sliding||G.advPending||!P.clueGrow||aiBusy)return;
  if(aiTrial){aiTick();return;}
  const now=performance.now();if(now<aiNext)return;
  const cl=aiClues(),fresh=cl.filter(c=>aiSkip.get(c)!==aiSig(c));
  if(fresh.length){aiIdle=0;aiStart(fresh[0]);return;}
  aiDrawRing();
  if(!cl.length||!movesInView()){aiNote(cl.length?"RNN: no circle here tells on its own":"RNN: waiting for the board");aiNext=now+500;return;}
  if(++aiIdle<AI_GUESS_AFTER){aiNote("RNN: thinking…");aiNext=now+500;return;}
  aiGuess(cl);}
// what the network sees, on the board: the circle is cut into one slice per stone (boundaries halfway between neighbours);
// the slices scanned so far are shaded, and the slice it is on is lit while the stone's 60 ms onset pulse is on, then dimmed
function aiSector(cx,cy,R,a0,a1){const p=a=>(cx+R*Math.sin(a)).toFixed(1)+" "+(cy-R*Math.cos(a)).toFixed(1);
  if(a1-a0>=2*Math.PI-1e-6)return null;return "M"+cx.toFixed(1)+" "+cy.toFixed(1)+" L"+p(a0)+" A"+R.toFixed(1)+" "+R.toFixed(1)+" 0 "+(a1-a0>Math.PI?1:0)+" 1 "+p(a1)+" Z";}
function aiBounds(T,k){const A=T.angs,n=A.length;return[k===0?(A[n-1]-2*Math.PI+A[0])/2:(A[k-1]+A[k])/2,k===n-1?(A[n-1]+A[0]+2*Math.PI)/2:(A[k]+A[k+1])/2];}
function aiDrawRing(){
  if(!aiLayer||!svg.contains(aiLayer))aiLayer=el("g",{class:"ai"},svg);
  aiLayer.textContent="";const T=aiTrial;if(!T){return;}
  const cx=P.x[T.c],cy=P.y[T.c],R=P.R[T.c],last=T.seen.length?Math.max(...T.seen):-1;
  if(last>=0){const[a0]=aiBounds(T,0),[,a1]=aiBounds(T,last),d=aiSector(cx,cy,R,a0,a1);
    if(d)el("path",{class:"seen",d},aiLayer);else el("circle",{class:"seen",cx:f1(cx),cy:f1(cy),r:f1(R)},aiLayer);}
  el("circle",{class:"scan",cx:f1(cx),cy:f1(cy),r:f1(R)},aiLayer);
  T.now=null;
  if(T.fix>=0){const id=T.ids[T.fix],[b0,b1]=aiBounds(T,T.fix);T.now=el("path",{class:"now",d:aiSector(cx,cy,R,b0,b1),"fill-opacity":T.t-T.on<=CN_PULSE?0.55:0.12},aiLayer);
    el("circle",{class:"fov",cx:f1(P.x[id]),cy:f1(P.y[id]),r:f1(RS+7)},aiLayer);}}
function aiNote(s){$("#aiNote").textContent=s;}
let aiCtl=null;
setInterval(()=>{if(!aiOn)return;
  if(!aiPaused){aiAcc+=aiSpeed;while(aiAcc>=1){aiAcc-=1;aiLoop();}}   // at speed s, one network step every 20/s ms of real time
  else if(aiFwd&&!aiTrial){aiLoop();if(aiTrial){aiFwd=false;aiSeek(aiTrial,aiTrial.stops[0]);}}   // stepping on into the next circle: marks and board first, then stop at that circle's start
  aiSync();},AI_MS);
function aiStep(dir){if(!aiOn)return;aiSetPaused(true);let T=aiTrial;   // the Step buttons: on or back one stone (stepping pauses playback)
  if(T&&aiStale(T)){aiTrial=T=null;aiDrawRing();}
  if(dir<0){const s=T&&aiPrevStop(T);if(s!=null)aiSeek(T,s);}
  else if(T&&T.verdict<0){const s=aiNextStop(T);if(s!=null)aiSeek(T,s);}
  else{if(T)aiAct();aiFwd=true;if(!aiBusy)aiNext=0;}   // the answer is in: act on it, then stop at the start of the next circle
  aiSync();}
function aiSetPaused(p){aiPaused=p;aiAcc=0;if(!p)aiFwd=false;if(aiCtl){aiCtl._pause.textContent=p?"Resume":"Pause";aiCtl._pause.setAttribute("aria-pressed",String(p));}}
function aiSync(){const c=aiCtl;if(!c)return;const T=aiTrial,b=!(T&&aiPrevStop(T)!=null),s=!T?"–":T.verdict>=0?"answer":T.mk.length?"stone "+T.mk.length+" / "+T.ids.length:"start";   // the stone count between the Step buttons
  if(c._back.disabled!==b)c._back.disabled=b;if(c._pos.textContent!==s)c._pos.textContent=s;}
function aiControls(){if(aiCtl)return aiCtl;   // right under the board: ◀ Step · stone k / n · Step ▶ · Pause · speed, then the status note
  const c=aiCtl=document.createElement("span");c.id="aiCtl";c.style.cssText="display:inline-flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:center";
  const mk=(t,ti)=>{const b=document.createElement("button");b.type="button";b.className="btn";b.textContent=t;b.title=ti;return b;};
  const back=mk("◀︎ Step","Back one stone (pauses; within this circle)"),fwd=mk("Step ▶︎","On one stone (pauses); after the answer, acts on it and goes to the next circle");
  const pos=document.createElement("span");pos.className="aipos";
  const pause=mk("Pause","Pause or resume the network");pause.setAttribute("aria-pressed","false");pause.style.minWidth="6.4em";   // Pause / Resume: one width
  const seg=document.createElement("span");seg.className="seg";seg.setAttribute("role","group");seg.setAttribute("aria-label","Playback speed");
  [1,0.75,0.5,0.25].forEach(s=>{const b=document.createElement("button");b.type="button";b.textContent=s+"×";b.setAttribute("aria-pressed",String(s===aiSpeed));
    b.title="Play at "+s+"× real time: one 20 ms network step every "+Math.round(20/s)+" ms";
    b.addEventListener("click",()=>{aiSpeed=s;aiAcc=0;seg.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",String(x===b)));});seg.appendChild(b);});
  back.addEventListener("click",()=>aiStep(-1));fwd.addEventListener("click",()=>aiStep(1));
  pause.addEventListener("click",()=>{aiSetPaused(!aiPaused);if(aiPaused&&aiTrial)vzDraw(aiTrial);});
  c.append(back,pos,fwd,pause,seg);c._pause=pause;c._back=back;c._pos=pos;$("#aiPlay").prepend(c);return c;}
function setAI(on){
  aiOn=on;aiTrial=null;aiSkip=new Map();aiLast=new Map();aiIdle=0;aiBusy=false;aiFwd=false;$("#aiBtn").setAttribute("aria-pressed",String(on));
  $("#aiBtn").textContent=on?"Stop RNN":"RNN plays";aiNote(on?"RNN: starting…":"");$("#aiViz").hidden=!on;$("#aiPlay").hidden=!on;
  aiControls();aiSetPaused(false);aiSync();
  if(on)vzBuild();else{aiDrawRing();}
}

/* ---------- what the network is doing, drawn under the board ----------
   Cartoon plane: the tallies a player would keep (blues still needed across, slack = needed - hidden up).
   Outputs: the three output units through the current circle.  Real plane: the mean activity of the
   network's own need-type and slack-type unit groups, with ticks where real positions with each tally sit. */
const VZ={built:false};
function vzEl(p,t,a,tx){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(tx!=null)e.textContent=tx;p.appendChild(e);return e;}
const vzLX=v=>52+(v+2)/8*236,vzLY=v=>270-(v+3)/8*236,vzQX=v=>52+(v+0.8)/1.7*236,vzQY=v=>270-(v+0.8)/1.7*236;
const VZ_DK="var(--ink-out)",VZ_BL="var(--blue-out)",VZ_AM="#ba7517",VZ_PU="#7f77dd",VZ_LAB=["rest dark","rest blue","can't tell"],VZ_LC=[VZ_DK,VZ_BL,"#ba7517"];
function vzBuild(){if(VZ.built)return;VZ.built=true;
  const L=$("#vzL"),O=$("#vzO"),Q=$("#vzP"),I=$("#vzI");
  vzEl(I,"text",{x:0,y:14,class:"th"},"What the network sees");vzEl(I,"text",{x:0,y:28,class:"ts"},"its five input channels, one row each, 20 ms steps");
  const IX0=78,IX1=290,IY0=44,IY1=236,ilh=(IY1-IY0)/5,lanes=["blue stone","dark stone","hidden stone","number ÷ 10","scan complete"];VZ.i={X0:IX0,X1:IX1,Y0:IY0,Y1:IY1,lh:ilh,mx:[2,2,2,1.5,1]};
  lanes.forEach((l,k)=>{const y=IY0+k*ilh;vzEl(I,"line",{x1:IX0,x2:IX1,y1:y+ilh-2,y2:y+ilh-2,class:"grid"});vzEl(I,"text",{x:IX0-5,y:y+ilh/2+3,"text-anchor":"end",class:"ts"},l);});
  VZ.idyn=vzEl(I,"g",{});VZ.itr=[VZ_BL,VZ_DK,"#888780",VZ_AM,VZ_PU].map(col=>vzEl(I,"polyline",{fill:"none",stroke:col,"stroke-width":1.6}));VZ.icur=vzEl(I,"line",{y1:IY0,y2:IY1,stroke:"var(--fg)","stroke-width":1,opacity:0});
  vzEl(I,"text",{x:IX0,y:IY1+28,class:"ts"},"each stone: a 60 ms pulse at landing, then nothing");
  vzEl(L,"text",{x:0,y:14,class:"th"},"Speed landscape across the trough");vzEl(L,"text",{x:0,y:28,class:"ts"},"cross-section through this number's trough (dark = slow)");
  const LX0=34,LX1=292,LY0=38,LY1=232,E=CN_CROSS.ext,cl=(v,a,b)=>Math.max(a,Math.min(b,v));   // across: hidden-stone trace; up: dark-stone push (count_cross_export.py)
  VZ.lv={X0:LX0,X1:LX1,Y0:LY0,Y1:LY1,ax:u=>LX0+(cl(u,E[0],E[1])-E[0])/(E[1]-E[0])*(LX1-LX0),ay:v=>LY1-(cl(v,E[2],E[3])-E[2])/(E[3]-E[2])*(LY1-LY0)};
  VZ.limg=vzEl(L,"image",{x:LX0,y:LY0,width:LX1-LX0,height:LY1-LY0,preserveAspectRatio:"none"});VZ.lflow=vzEl(L,"g",{});VZ.lticks=vzEl(L,"g",{});
  const o0=[VZ.lv.ax(0),VZ.lv.ay(0)];vzEl(L,"circle",{cx:o0[0].toFixed(1),cy:o0[1].toFixed(1),r:4,fill:"none",stroke:"#fff","stroke-width":1.2});vzEl(L,"text",{x:o0[0].toFixed(1),y:(o0[1]+15).toFixed(1),"text-anchor":"middle",class:"ts",style:"fill:#fff"},"trough");
  VZ.ll=vzEl(L,"g",{});VZ.lm=vzEl(L,"g",{});VZ.ldot=vzEl(L,"circle",{r:5,fill:"#fff",stroke:"#fff","stroke-width":1.5,opacity:0});
  VZ.lfield=vzEl(L,"text",{x:LX1-4,y:LY0+12,"text-anchor":"end",class:"ts",style:"fill:#fff;font-weight:600"},"");VZ.lcount=vzEl(L,"text",{x:LX0+4,y:LY0+12,class:"ts",style:"fill:#fff;font-weight:600"},"");
  vzEl(L,"text",{x:LX0,y:LY1+26,class:"ts"},"ticks: hidden stones seen · arrows: flow");
  vzEl(L,"text",{x:LX0,y:LY1+38,class:"ts"},"→ hidden-stone trace: a flat floor, so it stays");
  vzEl(L,"text",{x:LX0,y:LY1+50,class:"ts"},"↑ dark-stone push: steep walls pull it straight back");
  vzEl(L,"text",{x:LX0,y:LY1+62,class:"ts"},"blue stones move along the trough, out of this plane");
  const V3=$("#vz3");vzEl(V3,"text",{x:0,y:14,class:"th"},"The troughs in 3-D");vzEl(V3,"text",{x:0,y:28,class:"ts"},"PC1-3 of the count states of all 15 numbers · drag to rotate");
  VZ.r3={X0:10,X1:290,Y0:36,Y1:262,az:Math.PI,el:0.38};VZ.ghost=[];   // one fixed view for every circle (the fan of all 15 troughs), so the cue's switch of trough shows
  const cp=vzEl(vzEl(V3,"defs",{}),"clipPath",{id:"vz3clip"});vzEl(cp,"rect",{x:4,y:34,width:292,height:232});
  const V3c=vzEl(V3,"g",{"clip-path":"url(#vz3clip)"});VZ.g3=vzEl(V3c,"g",{});VZ.gh3=vzEl(V3c,"g",{});VZ.tr3=vzEl(V3c,"polyline",{fill:"none",stroke:"#fff","stroke-width":0.6,opacity:0.6});
  VZ.l3c=vzEl(V3c,"g",{});VZ.l3=vzEl(V3c,"g",{});VZ.m3=vzEl(V3c,"g",{});
  VZ.drop3=vzEl(V3c,"line",{stroke:"var(--fg)","stroke-width":1,"stroke-dasharray":"2 2",opacity:0});VZ.dot3=vzEl(V3,"circle",{r:5,fill:"#fff",stroke:"#000","stroke-width":1.2,opacity:0});
  vzEl(V3,"text",{x:10,y:275,class:"ts"},"fixed view: the number cue picks the trough");vzEl(V3,"text",{x:10,y:286,class:"ts"},"bold: this circle's number, a band per blue");
  vzEl(V3,"text",{x:10,y:297,class:"ts"},"faded: the paths of earlier circles");
  let drag=null;V3.style.cursor="grab";V3.style.touchAction="none";
  V3.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY];V3.setPointerCapture(e.pointerId);V3.style.cursor="grabbing";});
  V3.addEventListener("pointermove",e=>{if(!drag||!VZ.cur)return;const r=VZ.r3;r.az-=(e.clientX-drag[0])*0.01;r.el=Math.max(-1.5,Math.min(1.5,r.el+(e.clientY-drag[1])*0.01));drag=[e.clientX,e.clientY];vz3Scene();vz3Live(VZ.cur);});
  V3.addEventListener("pointerup",()=>{drag=null;V3.style.cursor="grab";});
  const TV=$("#vzT");vzEl(TV,"text",{x:0,y:14,class:"th"},"Trough and verdicts");vzEl(TV,"text",{x:0,y:28,class:"ts"},"this number's trough unrolled; up = toward the end-of-scan answers");
  const TX0=34,TX1=292,TY0=38,TY1=232;VZ.tv={X0:TX0,X1:TX1,Y0:TY0,Y1:TY1};
  VZ.timg=vzEl(TV,"image",{x:TX0,y:TY0,width:TX1-TX0,height:TY1-TY0,preserveAspectRatio:"none"});VZ.tflow=vzEl(TV,"g",{});VZ.tticks=vzEl(TV,"g",{});
  VZ.ttr=vzEl(TV,"polyline",{fill:"none",stroke:"#fff","stroke-width":0.6,opacity:0.6});VZ.tl=vzEl(TV,"g",{});VZ.tm=vzEl(TV,"g",{});VZ.tdot=vzEl(TV,"circle",{r:5,fill:"#fff",stroke:"#fff","stroke-width":1.5,opacity:0});
  vzEl(TV,"text",{x:TX0,y:TY1+52,class:"ts"},"dot fills with the network's actual answer: its state");vzEl(TV,"text",{x:TX0,y:TY1+63,class:"ts"},"also lies partly off this sheet, so regions are a guide");
  VZ.tfield=vzEl(TV,"text",{x:TX1-4,y:TY0+12,"text-anchor":"end",class:"ts",style:"fill:#fff;font-weight:600"},"");
  vzEl(TV,"text",{x:TX0,y:TY1+24,class:"ts"},"ticks: blues seen · arrows: flow · lines: output = 0.5");
  [["rest dark","#ffffff"],["rest blue","#378add"],["can't tell","#ba7517"]].forEach(([l,c],k)=>{vzEl(TV,"rect",{x:TX0+k*86,y:TY1+31,width:14,height:4,fill:c,stroke:k?"none":"#2c2c2a","stroke-width":0.5});vzEl(TV,"text",{x:TX0+k*86+18,y:TY1+36,class:"ts"},l);});
  vzEl(O,"text",{x:0,y:14,class:"th"},"The three output units");vzEl(O,"text",{x:0,y:28,class:"ts"},"free response: an output above 0.5 is the answer");
  const X0=40,X1=290,Y0=44,Y1=230,sy=v=>Y1-(v+0.2)/1.4*(Y1-Y0);VZ.o={X0,X1,Y0,Y1,sy};
  [0,0.5,1].forEach(v=>{vzEl(O,"line",{x1:X0,x2:X1,y1:sy(v),y2:sy(v),class:"grid"});vzEl(O,"text",{x:X0-5,y:sy(v)+4,"text-anchor":"end",class:"ts"},v);});
  VZ.odyn=vzEl(O,"g",{});VZ.otr=[0,1,2].map(k=>vzEl(O,"polyline",{fill:"none",stroke:VZ_LC[k],"stroke-width":2}));VZ.ocur=vzEl(O,"line",{y1:Y0,y2:Y1,stroke:"var(--fg)","stroke-width":1,opacity:0});
  VZ_LAB.forEach((l,k)=>{vzEl(O,"rect",{x:X0+k*85,y:250,width:12,height:10,fill:VZ_LC[k]});vzEl(O,"text",{x:X0+k*85+16,y:259,class:"ts"},l);});
  vzEl(Q,"text",{x:0,y:14,class:"th"},"Real population plane");vzEl(Q,"text",{x:0,y:28,class:"ts"},"mean activity of the need-type (81) and slack-type (23) units");
  CN_REF.need.forEach(([v,x])=>{vzEl(Q,"line",{x1:vzQX(x),x2:vzQX(x),y1:vzQY(-0.8),y2:vzQY(-0.74),stroke:"var(--muted)"});vzEl(Q,"text",{x:vzQX(x),y:vzQY(-0.8)+14,"text-anchor":"middle",class:"ts"},v);});
  CN_REF.slack.forEach(([v,y])=>{vzEl(Q,"line",{x1:vzQX(-0.8),x2:vzQX(-0.74),y1:vzQY(y),y2:vzQY(y),stroke:"var(--muted)"});vzEl(Q,"text",{x:vzQX(-0.8)-5,y:vzQY(y)+4,"text-anchor":"end",class:"ts"},v);});
  vzEl(Q,"text",{x:vzQX(0.9),y:vzQY(-0.8)+27,"text-anchor":"end",class:"ts"},"← blues still needed (ticks: where real positions sit)");vzEl(Q,"text",{x:vzQX(-0.72),y:vzQY(0.88),class:"ts"},"↑ slack");
  const n0=CN_REF.need[0][1],s0=CN_REF.slack[4][1];
  vzEl(Q,"rect",{x:vzQX(-0.8),y:vzQY(0.9),width:vzQX(n0+0.25)-vzQX(-0.8),height:vzQY(-0.8)-vzQY(0.9),fill:VZ_DK,opacity:0.1});
  vzEl(Q,"rect",{x:vzQX(n0+0.25),y:vzQY(s0+0.1),width:vzQX(0.9)-vzQX(n0+0.25),height:vzQY(s0-0.12)-vzQY(s0+0.1),fill:VZ_BL,opacity:0.12});
  VZ.qtr=vzEl(Q,"polyline",{fill:"none",stroke:VZ_PU,"stroke-width":1.6,opacity:0.75});VZ.qdot=vzEl(Q,"circle",{r:7,fill:VZ_PU,opacity:0});}
function vzTrial(T){   // per circle: the number's start point on the cartoon plane, the time axis and scan-complete line of the outputs
  if(!VZ.built)return;const o=VZ.o,sx=i=>o.X0+i/(T.L-1)*(o.X1-o.X0);VZ.sx=sx;VZ.odyn.textContent="";
  vzEl(VZ.odyn,"line",{x1:sx(T.done),x2:sx(T.done),y1:o.Y0,y2:o.Y1,stroke:VZ_AM,"stroke-dasharray":"3 3"});vzEl(VZ.odyn,"text",{x:sx(T.done)-3,y:o.Y1-6,"text-anchor":"end",class:"ts"},"scan complete");
  for(let s=0;s*50<T.L;s+=2)vzEl(VZ.odyn,"text",{x:sx(s*50),y:o.Y1+16,"text-anchor":"middle",class:"ts"},s+" s");
  VZ.otr.forEach(p=>p.setAttribute("points",""));VZ.qtr.setAttribute("points","");
  const li=VZ.i,ix=i=>li.X0+i/(T.L-1)*(li.X1-li.X0);VZ.ix=ix;VZ.idyn.textContent="";
  vzEl(VZ.idyn,"line",{x1:ix(T.done),x2:ix(T.done),y1:li.Y0,y2:li.Y1,stroke:VZ_AM,"stroke-dasharray":"3 3"});
  for(let s=0;s*50<T.L;s+=2)vzEl(VZ.idyn,"text",{x:ix(s*50),y:li.Y1+14,"text-anchor":"middle",class:"ts"},s+" s");
  VZ.itr.forEach(p=>p.setAttribute("points",""));VZ.icur.setAttribute("opacity",1);
  const Q=CN_CROSS.nums[String(Math.min(15,Math.max(1,T.num)))],lv=VZ.lv;VZ.ck=-1;VZ.csc=-1;VZ.ll.textContent="";VZ.lm.textContent="";VZ.lln=9;VZ.lmn=0;VZ.lv2=false;VZ.ldot.setAttribute("opacity",0);   // cross-section panel
  VZ.lticks.textContent="";let lx=-99;Q.ticks.forEach((u,m)=>{if(u>CN_CROSS.ext[1])return;const x=lv.ax(u);vzEl(VZ.lticks,"line",{x1:x.toFixed(1),x2:x.toFixed(1),y1:lv.Y1,y2:lv.Y1+4,stroke:"var(--muted)"});   // where m hidden stones leave the state
    if(x-lx>=10){vzEl(VZ.lticks,"text",{x:x.toFixed(1),y:lv.Y1+14,"text-anchor":"middle",class:"ts"},m);lx=x;}});
  VZ.qdot.setAttribute("opacity",1);VZ.ocur.setAttribute("opacity",1);
  VZ.tR=CN_TROUGH.nums[String(Math.min(15,Math.max(1,T.num)))];VZ.tsc=-1;VZ.ttr.setAttribute("points","");VZ.tdot.setAttribute("opacity",0);
  VZ.tl.textContent="";VZ.tm.textContent="";VZ.tln=9;VZ.tmn=0;VZ.tv2=false;VZ.drop3&&VZ.drop3.setAttribute("opacity",0);
  const tv=VZ.tv,E=VZ.tR.ext;VZ.tx=v=>tv.X0+(v-E[0])/(E[1]-E[0])*(tv.X1-tv.X0);VZ.ty=v=>tv.Y1-(Math.max(E[2],Math.min(E[3],v))-E[2])/(E[3]-E[2])*(tv.Y1-tv.Y0);
  VZ.tticks.textContent="";VZ.tR.ticks.forEach((s,k)=>{if(VZ.tR.ticks.length>9&&k%2&&k!==VZ.tR.ticks.length-1)return;vzEl(VZ.tticks,"line",{x1:VZ.tx(s),x2:VZ.tx(s),y1:tv.Y1,y2:tv.Y1+4,stroke:"var(--muted)"});vzEl(VZ.tticks,"text",{x:VZ.tx(s),y:tv.Y1+14,"text-anchor":"middle",class:"ts"},k);});
  vzTField(T,0);
  const P3=VZ.cur&&VZ.cur!==T?VZ.cur.p3:null;   // the last three circles' paths stay on the map, faded
  if(P3&&P3.length>12){VZ.ghost.unshift({num:VZ.cur.num,p:P3.filter((_,i)=>i>=9&&(i%2===0||i===P3.length-1))});VZ.ghost.length=Math.min(VZ.ghost.length,3);}
  VZ.cur=T;VZ.tr3.setAttribute("points","");VZ.dot3.setAttribute("opacity",0);vz3Scene();}
const VZ_VIR=[[68,1,84],[71,45,123],[59,82,139],[44,114,142],[33,145,140],[40,174,128],[94,201,98],[173,220,48],[253,231,37]];   // viridis, as in the trough panel's images
function vzVirT(t){t=Math.max(0,Math.min(1,t));const i=Math.min(7,Math.floor(t*8)),f=t*8-i,a=VZ_VIR[i],b=VZ_VIR[i+1];return[0,1,2].map(c=>Math.round(a[c]+(b[c]-a[c])*f));}
const VZ_CRC={};   // rendered cross-section fields, by number:count:field
function vzCImg(num,k,sc){const key=num+":"+k+":"+sc;if(VZ_CRC[key])return VZ_CRC[key];
  const nu=CN_CROSS.NU,nv=CN_CROSS.NV,b=atob(CN_CROSS.nums[String(num)]["q"+sc][k]),c0=document.createElement("canvas");c0.width=nu;c0.height=nv;
  const x0=c0.getContext("2d"),im=x0.createImageData(nu,nv);for(let i=0;i<nu*nv;i++){const c=vzVirT(b.charCodeAt(i)/255);im.data[4*i]=c[0];im.data[4*i+1]=c[1];im.data[4*i+2]=c[2];im.data[4*i+3]=255;}
  x0.putImageData(im,0,0);const c1=document.createElement("canvas");c1.width=nu*8;c1.height=nv*8;const x1=c1.getContext("2d");x1.imageSmoothingEnabled=true;x1.imageSmoothingQuality="high";x1.drawImage(c0,0,0,c1.width,c1.height);
  return VZ_CRC[key]=c1.toDataURL();}
const vzNumCol=k=>"hsl("+(180+k*8)+",65%,50%)";   // the troughs of other numbers: cyan (1) to violet (15)
function vz3Scene(){const T=VZ.cur;if(!T)return;const r=VZ.r3,nq=Math.min(15,Math.max(1,T.num)),R=CN_R3.nums[String(nq)];
  const v=[Math.cos(r.el)*Math.cos(r.az),Math.cos(r.el)*Math.sin(r.az),Math.sin(r.el)];let e1=[-v[1],v[0],0];const n1=Math.hypot(e1[0],e1[1])||1;e1=[e1[0]/n1,e1[1]/n1,0];
  const e2=[v[1]*e1[2]-v[2]*e1[1],v[2]*e1[0]-v[0]*e1[2],v[0]*e1[1]-v[1]*e1[0]];r.v=v;r.e1=e1;r.e2=e2;
  const pj=p=>[p[0]*e1[0]+p[1]*e1[1]+p[2]*e1[2],p[0]*e2[0]+p[1]*e2[1]+p[2]*e2[2]];
  let lo=[1e9,1e9],hi=[-1e9,-1e9];const fit=p=>{const s=pj(p);lo=[Math.min(lo[0],s[0]),Math.min(lo[1],s[1])];hi=[Math.max(hi[0],s[0]),Math.max(hi[1],s[1])];};
  for(const k in CN_R3.nums)CN_R3.nums[k].tr.forEach(fit);R.g.forEach(row=>row.forEach(fit));   // frame all 15 troughs: the same map for every circle
  const sc=0.9*Math.min((r.X1-r.X0)/(hi[0]-lo[0]),(r.Y1-r.Y0)/(hi[1]-lo[1])),cx=(lo[0]+hi[0])/2,cy=(lo[1]+hi[1])/2,mx=(r.X0+r.X1)/2,my=(r.Y0+r.Y1)/2;
  r.P=p=>{const s=pj(p);return[mx+(s[0]-cx)*sc,my-(s[1]-cy)*sc];};const g=VZ.g3;g.textContent="";const P=r.P,f=a=>a.map(p=>{const s=P(p);return s[0].toFixed(1)+","+s[1].toFixed(1);}).join(" ");
  for(const k in CN_R3.nums){if(+k===nq)continue;const tr=CN_R3.nums[k].tr,s=P(tr[0]);   // the other numbers' troughs, each labelled at its start
    vzEl(g,"polyline",{points:f(tr),fill:"none",stroke:vzNumCol(+k),"stroke-width":1.1,opacity:0.8});vzEl(g,"text",{x:(s[0]+3).toFixed(1),y:(s[1]+3).toFixed(1),class:"ts"},k);}
  const quads=[],G=R.g,dep=p=>p[0]*v[0]+p[1]*v[1]+p[2]*v[2];
  for(let i=0;i<G.length-1;i++)for(let j=0;j<G[0].length-1;j++){const c=[G[i][j],G[i+1][j],G[i+1][j+1],G[i][j+1]];quads.push([(dep(c[0])+dep(c[2]))/2,c,Math.floor(i/6)%2]);}   // this number's ribbon, a band per count step (6 grid rows each)
  quads.sort((a,b)=>a[0]-b[0]).forEach(([_,c,band])=>vzEl(g,"polygon",{points:f(c),fill:"var(--fg)",stroke:"none","fill-opacity":band?0.26:0.1}));
  vzEl(g,"polyline",{points:f(R.tr),fill:"none",stroke:"var(--fg)","stroke-width":2});R.tr.forEach(p=>{const s=P(p);vzEl(g,"circle",{cx:s[0].toFixed(1),cy:s[1].toFixed(1),r:1.8,fill:"var(--fg)"});});
  const s0=P(R.tr[0]);vzEl(g,"text",{x:(s0[0]+4).toFixed(1),y:(s0[1]+4).toFixed(1),class:"th"},nq);
  VZ.gh3.textContent="";VZ.ghost.forEach((gh,i)=>vzEl(VZ.gh3,"polyline",{points:f(gh.p),fill:"none",stroke:vzNumCol(Math.min(15,Math.max(1,gh.num))),"stroke-width":1.4,opacity:(0.7-0.18*i).toFixed(2)}));
  VZ.l3.textContent="";VZ.l3c.textContent="";VZ.m3.textContent="";VZ.l3n=8;VZ.m3n=0;VZ.v3=false;}   // trail and markers are redrawn in full on the next frame
function vzMark(g,s,kind){const col=["#378add","#2c2c2a","#ba7517"][kind],x=s[0],y=s[1];   // stone onset: blue circle, dark square, hidden triangle
  if(kind===0)vzEl(g,"circle",{cx:x.toFixed(1),cy:y.toFixed(1),r:4,fill:col,stroke:"#fff","stroke-width":1});
  else if(kind===1)vzEl(g,"rect",{x:(x-3.5).toFixed(1),y:(y-3.5).toFixed(1),width:7,height:7,fill:col,stroke:"#fff","stroke-width":1});
  else vzEl(g,"polygon",{points:[[x,y-4.5],[x-4,y+3],[x+4,y+3]].map(p=>p[0].toFixed(1)+","+p[1].toFixed(1)).join(" "),fill:col,stroke:"#fff","stroke-width":1});}
function vzStar(g,s){const pts=[];for(let i=0;i<10;i++){const r=i%2?3.5:8,a=-Math.PI/2+i*Math.PI/5;pts.push((s[0]+r*Math.cos(a)).toFixed(1)+","+(s[1]+r*Math.sin(a)).toFixed(1));}vzEl(g,"polygon",{points:pts.join(" "),fill:"#ffd400",stroke:"#000","stroke-width":0.8});}
function vzTrail(T,g,proj,from,key,gc){   // the whole trajectory since the number came on, one segment per 20 ms step coloured by speed; drawn incrementally (gc: a dark casing underneath)
  const n=T.z.length;for(let i=Math.max(from,VZ[key]);i<n;i++){if(i<9)continue;const a=proj(i-1),b=proj(i),at={x1:a[0].toFixed(1),y1:a[1].toFixed(1),x2:b[0].toFixed(1),y2:b[1].toFixed(1),"stroke-linecap":"round"};
    if(gc)vzEl(gc,"line",Object.assign({stroke:"#000","stroke-opacity":0.5,"stroke-width":4.4},at));vzEl(g,"line",Object.assign({stroke:vzSpCol(T.sp[i]),"stroke-width":2.4},at));}VZ[key]=n;}
function vz3Live(T){const r=VZ.r3;if(!r.P||!T.p3.length)return;const n=T.p3.length,pr=i=>r.P(T.p3[i]);
  vzTrail(T,VZ.l3,pr,9,"l3n",VZ.l3c);   // on the map the trail crosses the page background, so it gets a casing
  for(;VZ.m3n<T.mk.length;VZ.m3n++){const[t,k]=T.mk[VZ.m3n];if(t<n)vzMark(VZ.m3,pr(t),k);else break;}
  if(T.verdict>=0&&T.verdict<3&&!VZ.v3){vzStar(VZ.m3,pr(n-1));VZ.v3=true;}
  if(n>8){const s=pr(n-1),z=T.z[n-1],k=z.indexOf(Math.max(z[0],z[1],z[2]));VZ.dot3.setAttribute("cx",s[0].toFixed(1));VZ.dot3.setAttribute("cy",s[1].toFixed(1));VZ.dot3.setAttribute("opacity",1);
    VZ.dot3.setAttribute("fill",z[k]>CN_THR?["#2c2c2a","#378add","#ba7517"][k]:"#fff");VZ.dot3.setAttribute("r",z[k]>CN_THR?7:5);
    const G=CN_R3.nums[String(Math.min(15,Math.max(1,T.num)))].g,mid=Math.floor(G[0].length/2),p=T.p3[n-1];let jb=0,db=1e9;   // drop line to the nearest point of this number's trough
    G.forEach((row,i)=>{const q=row[mid],d=(q[0]-p[0])**2+(q[1]-p[1])**2+(q[2]-p[2])**2;if(d<db){db=d;jb=i;}});const c=r.P(G[jb][mid]);
    VZ.drop3.setAttribute("x1",s[0].toFixed(1));VZ.drop3.setAttribute("y1",s[1].toFixed(1));VZ.drop3.setAttribute("x2",c[0].toFixed(1));VZ.drop3.setAttribute("y2",c[1].toFixed(1));VZ.drop3.setAttribute("opacity",0.8);}}
function vzTField(T,sc){if(VZ.tsc===sc)return;VZ.tsc=sc;const R=VZ.tR;VZ.timg.setAttribute("href",R["png"+sc]);VZ.tflow.textContent="";   // the field the state is in: scan-complete off, then on
  VZ.tfield.textContent=sc?"field with scan complete ON":"field while scanning";
  const kx=(VZ.tv.X1-VZ.tv.X0)/(R.ext[1]-R.ext[0]),ky=(VZ.tv.Y1-VZ.tv.Y0)/(R.ext[3]-R.ext[2]);
  R["flow"+sc].forEach(([x,y,dx,dy])=>{const px=dx*kx,py=-dy*ky,sp=Math.hypot(px,py)||1e-9,x0=VZ.tx(x),y0=VZ.ty(y),x1=x0+px/sp*7,y1=y0+py/sp*7;
    vzEl(VZ.tflow,"line",{x1:x0.toFixed(1),y1:y0.toFixed(1),x2:x1.toFixed(1),y2:y1.toFixed(1),stroke:"#fff","stroke-opacity":0.5,"stroke-width":1});vzEl(VZ.tflow,"circle",{cx:x1.toFixed(1),cy:y1.toFixed(1),r:1.4,fill:"#fff","fill-opacity":0.75});});}
function vzCField(T,k,sc){if(VZ.ck===k&&VZ.csc===sc)return;VZ.ck=k;VZ.csc=sc;   // the cross-section the state is in: through count state k, scan complete off / on
  const nq=Math.min(15,Math.max(1,T.num)),Q=CN_CROSS.nums[String(nq)],lv=VZ.lv,E=CN_CROSS.ext,fu=CN_CROSS.fu,fv=CN_CROSS.fv,b=atob(Q["f"+sc][k]);
  VZ.limg.setAttribute("href",vzCImg(nq,k,sc));VZ.lfield.textContent=sc?"scan complete ON":"while scanning";VZ.lcount.textContent="at count "+k;VZ.lflow.textContent="";
  const kx=(lv.X1-lv.X0)/(E[1]-E[0]),ky=(lv.Y1-lv.Y0)/(E[3]-E[2]),s8=i=>{const c=b.charCodeAt(i);return(c>127?c-256:c)/CN_CROSS.fq;};
  for(let i=0;i<fu.length;i++)for(let j=0;j<fv.length;j++){   // flow on the plane; arrow length grows with its strength, so the flat floor shows as dots
    const n=2*(i*fv.length+j),du=s8(n),dv=s8(n+1),px=du*kx,py=-dv*ky,sp=Math.hypot(px,py)||1e-9,L=2+7*Math.min(1,Math.hypot(du,dv)/0.8),x0=lv.ax(fu[i]),y0=lv.ay(fv[j]),x1=x0+px/sp*L,y1=y0+py/sp*L;
    vzEl(VZ.lflow,"line",{x1:x0.toFixed(1),y1:y0.toFixed(1),x2:x1.toFixed(1),y2:y1.toFixed(1),stroke:"#fff","stroke-opacity":0.5,"stroke-width":1});vzEl(VZ.lflow,"circle",{cx:x1.toFixed(1),cy:y1.toFixed(1),r:1.4,fill:"#fff","fill-opacity":0.75});}}
function vzDraw(T){if(!VZ.built)return;
  let b=0,h=0;T.seen.forEach(j=>{const k=aiKind(T.ids[j]);if(k===0)b++;else if(k===2)h++;});
  const nn=T.tq.length,zt=T.z[nn-1],kt=zt.indexOf(Math.max(zt[0],zt[1],zt[2])),dcol=zt[kt]>CN_THR?["#2c2c2a","#378add","#ba7517"][kt]:"#fff",drad=zt[kt]>CN_THR?7:5;   // dots = the network's actual answer
  vz3Live(T);vzTField(T,T.t>T.done?1:0);   // trough panel: the whole trajectory since the number came on, as in the mockup
  const pt=i=>[VZ.tx(T.tq[i][0]),VZ.ty(T.tq[i][1])];vzTrail(T,VZ.tl,pt,9,"tln");
  for(;VZ.tmn<T.mk.length;VZ.tmn++){const[t,k]=T.mk[VZ.tmn];if(t<nn)vzMark(VZ.tm,pt(t),k);else break;}
  if(T.verdict>=0&&T.verdict<3&&!VZ.tv2){vzStar(VZ.tm,pt(nn-1));VZ.tv2=true;}
  if(nn>8){VZ.tdot.setAttribute("opacity",1);VZ.tdot.setAttribute("cx",VZ.tx(T.tq[nn-1][0]).toFixed(1));VZ.tdot.setAttribute("cy",VZ.ty(T.tq[nn-1][1]).toFixed(1));}
  VZ.tdot.setAttribute("fill",dcol);VZ.tdot.setAttribute("r",drad);
  const kq=Math.max(0,Math.min(Math.min(15,Math.max(1,T.num)),T.tq[nn-1][2]/CN_CROSS.seg));   // cross-section panel: through the count state nearest the state, with a little hysteresis
  vzCField(T,VZ.ck>=0&&Math.abs(kq-VZ.ck)<=0.6?VZ.ck:Math.round(kq),T.t>T.done?1:0);
  const pl=i=>[VZ.lv.ax(T.tq[i][3]),VZ.lv.ay(T.tq[i][4])];vzTrail(T,VZ.ll,pl,9,"lln");
  for(;VZ.lmn<T.mk.length;VZ.lmn++){const[t,k]=T.mk[VZ.lmn];if(t<nn)vzMark(VZ.lm,pl(t),k);else break;}
  if(T.verdict>=0&&T.verdict<3&&!VZ.lv2){vzStar(VZ.lm,pl(nn-1));VZ.lv2=true;}
  if(nn>8){const s=pl(nn-1);VZ.ldot.setAttribute("opacity",1);VZ.ldot.setAttribute("cx",s[0].toFixed(1));VZ.ldot.setAttribute("cy",s[1].toFixed(1));}
  VZ.ldot.setAttribute("fill",dcol);VZ.ldot.setAttribute("r",drad);
  const n=T.z.length,sx=VZ.sx,sy=VZ.o.sy;VZ.otr.forEach((p,k)=>p.setAttribute("points",T.z.map((v,i)=>sx(i)+","+sy(v[k])).join(" ")));VZ.ocur.setAttribute("x1",sx(n-1));VZ.ocur.setAttribute("x2",sx(n-1));
  const a0=Math.max(0,n-40);VZ.qtr.setAttribute("points",T.pn.slice(a0).map((v,i)=>vzQX(v)+","+vzQY(T.ps[a0+i])).join(" "));VZ.qdot.setAttribute("cx",vzQX(T.pn[n-1]));VZ.qdot.setAttribute("cy",vzQY(T.ps[n-1]));
  const li=VZ.i,ix=VZ.ix;VZ.itr.forEach((p,c)=>{const y0=li.Y0+c*li.lh+li.lh-2,hh=li.lh-5;p.setAttribute("points",T.u.map((v,i)=>ix(i)+","+(y0-Math.min(1,v[c]/li.mx[c])*hh).toFixed(1)).join(" "));});
  VZ.icur.setAttribute("x1",ix(n-1));VZ.icur.setAttribute("x2",ix(n-1));
  const z=T.z[n-1],k=z.indexOf(Math.max(z[0],z[1],z[2])),kind=["blue","dark","hidden"],j=T.fix,un=T.u[n-1];
  const inp=un[0]>0?"blue pulse":un[1]>0?"dark pulse":un[2]>0?"hidden pulse":un[4]>0?"scan-complete signal":"number only";
  $("#vzNote").innerHTML=(T.t*20/1000).toFixed(1)+" s · "+(j>=0?"looking at stone "+(j+1)+" of "+T.ids.length+" ("+kind[aiKind(T.ids[j])]+")":T.t<T.done?(T.seen.length?"gap: nothing in view, the count must be held":"number on, no stone looked at yet"):"scan complete")
    +" · input now: "+inp+" · seen "+b+" blue, "+h+" hidden · tallies: need "+(T.num-b)+", slack "+(T.num-b-h)+" · output: <b>"+(z[k]>CN_THR?VZ_LAB[k]:"no answer yet")+"</b>";}
$("#aiBtn").addEventListener("click",()=>{
  setAI(!aiOn);
  if(aiOn&&(!G||!G.run||G.won)){SESSION=SESSION||{i:0,res:[],bonus:{},demoDone:true};SESSION.demoDone=true;startRun();}
});
$("#btnAI").addEventListener("click",()=>{
  try{audioCtx=audioCtx||new (window.AudioContext||window.webkitAudioContext)();audioCtx.resume();}catch(e){}
  $("#intro").hidden=true;SESSION={i:0,res:[],bonus:{},demoDone:true};hintsLeft=2;setAI(true);startRun();
});
