"""Serve a lightweight browser viewer for three synchronized trajectories."""
from __future__ import annotations
import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
import signal
import subprocess
import time

HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>三轨迹查看器</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{font:14px system-ui;margin:18px;background:#f5f6f8;color:#20242a}h2{margin:0 0 8px}.bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;background:white;padding:12px;border-radius:8px}.card{background:white;padding:10px;border-radius:8px;margin:10px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:10px}.card h3{margin:0 0 6px}.card input{max-width:220px}.map{display:grid;grid-template-columns:auto 1fr;gap:3px 8px}.map select{width:150px}#plot{width:100%;height:calc(100vh - 230px);min-height:500px;background:#fff;border-radius:8px;display:block;cursor:grab}.hint{color:#667;font-size:12px}</style></head>
<body><h2>三轨迹 6DoF 可视化</h2><div class="bar"><label>时间范围 <input id="lo" type="range" min="0" max="100" value="0"><input id="hi" type="range" min="0" max="100" value="100"></label><label>时间偏移(ms) <input id="offset" type="number" value="0" step="1"></label><label><input id="align" type="checkbox"> VIO/实验室刚体对齐到视觉轨迹</label><button id="refresh">刷新</button><button onclick="location.href='matching_viewer.html'">逐帧查看匹配</button></div>
<div class="grid"><div class="card"><h3 style="color:#d62728">彩色贴片视觉</h3><input id="f0" type="file" accept=".csv"><div id="m0" class="map"></div><div class="hint">默认读取本目录 trajectory.csv；单位按 mm。</div></div><div class="card"><h3 style="color:#1f77b4">IMU+单目 VIO</h3><input id="f1" type="file" accept=".csv"><div id="m1" class="map"></div></div><div class="card"><h3 style="color:#2ca02c">三方标定实验室</h3><input id="f2" type="file" accept=".csv"><div id="m2" class="map"></div></div></div><canvas id="plot"></canvas>
<script>
const colors=['#d62728','#1f77b4','#2ca02c'], names=['视觉','VIO','实验室']; let data=[null,null,null], maps=[];
function parse(s){let a=s.trim().split(/\r?\n/).map(x=>x.split(',').map(y=>y.trim()));let h=a.shift();return {h,rows:a.filter(r=>r.length>=h.length)};}
function guess(h,keys){let l=h.map(x=>x.toLowerCase());for(let k of keys){let i=l.findIndex(x=>x===k);if(i>=0)return i}for(let k of keys.filter(x=>x.length>1)){let i=l.findIndex(x=>x.includes(k));if(i>=0)return i}return 0}
function selectors(k){let box=document.getElementById('m'+k),h=data[k]?.h||['x','y','z','time'];let aliases=[['x',['x','tx','pos_x','position_x','px']],['y',['y','ty','pos_y','position_y','py']],['z',['z','tz','pos_z','position_z','pz']],['t',['timestamp','time','stamp']]];box.innerHTML='';maps[k]={};aliases.forEach(([n,ks])=>{let lab=document.createElement('label');lab.textContent=n+':';let s=document.createElement('select');h.forEach((v,i)=>{let o=document.createElement('option');o.value=i;o.textContent=v;s.appendChild(o)});s.value=guess(h,ks);s.onchange=()=>update();lab.appendChild(s);box.appendChild(lab);maps[k][n]=s});}
function load(k,text){data[k]=parse(text);selectors(k);update()}
document.querySelectorAll('input[type=file]').forEach((e,k)=>e.onchange=()=>{let f=e.files[0];if(f){let r=new FileReader();r.onload=()=>load(k,r.result);r.readAsText(f)}});
async function boot(){try{let r=await fetch('trajectory.csv');if(r.ok)load(0,await r.text())}catch(e){}}
function points(k){if(!data[k])return null;let m=maps[k],ix=+m.x.value,iy=+m.y.value,iz=+m.z.value,it=+m.t.value,off=k?+document.getElementById('offset').value*1000:0;let p=data[k].rows.map((r,i)=>({x:+r[ix],y:+r[iy],z:+r[iz],t:+r[it]+off,i})).filter(p=>[p.x,p.y,p.z].every(Number.isFinite));return p}
function nearest(a,t){let lo=0,hi=a.length-1;while(lo<hi){let m=(lo+hi)>>1;if(a[m].t<t)lo=m+1;else hi=m}return a[lo]}
function rigid(src,dst){let n=Math.min(src.length,dst.length);if(n<3)return src;let A=src.slice(0,n),B=dst.slice(0,n),ca=A.reduce((s,p)=>s.map((v,j)=>v+p[['x','y','z'][j]]/n),[0,0,0]),cb=B.reduce((s,p)=>s.map((v,j)=>v+p[['x','y','z'][j]]/n),[0,0,0]);let H=[[0,0,0],[0,0,0],[0,0,0]];for(let i=0;i<n;i++){let a=['x','y','z'].map(q=>A[i][q]-ca[['x','y','z'].indexOf(q)]),b=['x','y','z'].map(q=>B[i][q]-cb[['x','y','z'].indexOf(q)]);for(let r=0;r<3;r++)for(let c=0;c<3;c++)H[r][c]+=a[r]*b[c]}return src}
function update(){let p=data.map((_,k)=>points(k));let traces=[];p.forEach((q,k)=>{if(!q)return;let lo=Math.floor(q.length*(+document.getElementById('lo').value/100)),hi=Math.ceil(q.length*(+document.getElementById('hi').value/100));q=q.slice(lo,hi);let text=q.map(v=>'frame '+v.i+'<br>t='+v.t);traces.push({type:'scatter3d',x:q.map(v=>v.x),y:q.map(v=>v.y),z:q.map(v=>v.z),mode:'lines+markers',name:names[k],line:{color:colors[k],width:k?5:7},marker:{size:k?2.5:3.5,color:colors[k]},text,hovertemplate:'%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra>'+names[k]+'</extra>'});if(q.length){traces.push({type:'scatter3d',x:[q[0].x,q[q.length-1].x],y:[q[0].y,q[q.length-1].y],z:[q[0].z,q[q.length-1].z],mode:'markers',name:names[k]+' 起/终点',showlegend:false,marker:{size:6,color:[colors[k],colors[k]],symbol:['diamond','circle']},text:['起点','终点'],hovertemplate:'%{text}<extra>'+names[k]+'</extra>'})}});let axis=t=>({title:t+' (mm)',range:[0,600],autorange:false,dtick:100,showspikes:false});Plotly.react('plot',traces,{margin:{l:0,r:0,t:25,b:0},paper_bgcolor:'#fff',scene:{aspectmode:'cube',xaxis:axis('X'),yaxis:axis('Y'),zaxis:axis('Z')},legend:{orientation:'h'},uirevision:'fixed-space'},{scrollZoom:false,displayModeBar:false,responsive:true})}
['lo','hi','offset','align'].forEach(id=>document.getElementById(id).oninput=update);document.getElementById('refresh').onclick=update;boot();
</script></body></html>'''

MATCH_HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐帧匹配检查</title>
<style>body{font:14px system-ui;margin:18px;background:#f5f6f8}video{width:min(100%,1200px);background:#111}.bar{background:#fff;padding:12px;border-radius:8px}#info{white-space:pre-wrap;background:#fff;padding:12px;margin-top:10px;border-radius:8px}</style>
<h2>逐帧双目匹配检查</h2><div class="bar"><button onclick="step(-1)">上一帧</button><button onclick="step(1)">下一帧</button><input id="frame" type="range" min="0" max="561" value="0" style="width:60%"><span id="num"></span></div><video id="v" controls preload="auto" playsinline src="matching_preview.mp4?v=h264"></video><div id="info">读取匹配记录中…</div>
<script>let v=document.getElementById('v'),f=document.getElementById('frame'),n=document.getElementById('num'),rows=[],seeking=false;fetch('frame_matches.jsonl').then(r=>r.text()).then(s=>{rows=s.trim().split(/\n/).map(JSON.parse);f.max=rows.length-1;render(0)});function render(i){if(!rows.length)return;i=Math.max(0,Math.min(rows.length-1,Math.round(i)));f.value=i;n.textContent='帧 '+i;document.getElementById('info').textContent=JSON.stringify(rows[i],null,2)}function seek(i){i=Math.max(0,Math.min(rows.length-1,Math.round(i)));seeking=true;v.currentTime=i/30;render(i)}function step(d){seek(+f.value+d)}f.oninput=()=>seek(f.value);v.ontimeupdate=()=>{if(!seeking)render(v.currentTime*30)};v.onseeked=()=>{seeking=false;render(v.currentTime*30)};</script></html>'''

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--port',type=int,default=8765);p.add_argument('--no-browser',action='store_true');p.add_argument('--keep-existing',action='store_true',help='keep an existing viewer on this port');a=p.parse_args(argv)
    root=a.trajectory.parent; (root/'trajectory_viewer.html').write_text(HTML,encoding='utf-8'); (root/'matching_viewer.html').write_text(MATCH_HTML,encoding='utf-8')
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(root),**kwargs)
    def close_existing_viewer():
        if a.keep_existing:
            return
        try:
            result=subprocess.run(['pgrep','-af','nexumi-view-trajectories'],capture_output=True,text=True,check=False)
        except OSError:
            return
        needle=f'--port {a.port}'
        for line in result.stdout.splitlines():
            fields=line.split(None,1)
            if len(fields)!=2 or needle not in fields[1]:
                continue
            try:
                pid=int(fields[0])
                if pid==os.getpid():
                    continue
                os.kill(pid,signal.SIGTERM)
            except (ProcessLookupError,ValueError,PermissionError):
                continue
        time.sleep(0.15)
    close_existing_viewer()
    try:
        server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    except OSError as error:
        if error.errno==98:
            raise SystemExit(f'端口 {a.port} 已被其他程序占用；请换 --port，或先关闭占用程序') from error
        raise
    url=f'http://127.0.0.1:{a.port}/trajectory_viewer.html'
    print(url,flush=True)
    if not a.no_browser:
        import webbrowser;webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
