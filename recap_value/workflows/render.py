#!/usr/bin/env python3
"""Standalone scientific plots and local, video-linked HTML report (matplotlib)."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','savefig.facecolor':'white','axes.grid':True,'grid.alpha':.17})
BLUE='#2463b9';ORANGE='#d26121';PURPLE='#8652a0';GREEN='#098779'
STYLE='''body{margin:0;background:#f3f6fa;color:#192b41;font:16px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:28px;line-height:1.3}h2{font-size:21px}a{color:#165ca6}nav{display:flex;gap:20px;flex-wrap:wrap}section,.card{background:white;padding:20px;border-radius:12px;margin:20px 0;box-shadow:0 2px 10px #142a4210}.muted{color:#5c6b7d}.notice{background:#fff4df;border-left:4px solid #d99628;padding:15px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #e5eaf1}th{background:#eef3f9}img{width:100%;height:auto}video{width:100%;max-height:520px;background:#111;border-radius:8px}input[type=range]{width:100%}select{padding:7px;margin-right:12px}code{font-size:13px;overflow-wrap:anywhere}.metric{display:inline-block;margin:6px 22px 6px 0}.metric b{display:block;font-size:25px}.plot{position:relative;padding:0}.cursor{position:absolute;top:10%;bottom:8.5%;width:1px;background:#25394c;pointer-events:none}.good{color:#16765d}.bad{color:#b94735}.scroll{overflow:auto}footer{margin-top:30px;font-size:13px;color:#5c6b7d}button{padding:7px 12px;cursor:pointer}'''


def page(title,body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style><main>'+body+'</main></html>'


def shade(ax,t,flags):
    starts=np.flatnonzero((flags==1)&np.r_[True,flags[:-1]==0])
    ends=np.flatnonzero((flags==1)&np.r_[flags[1:]==0,True])
    for a,b in zip(starts,ends):ax.axvspan(t[a],t[b],color='#a094b8',alpha=.13,lw=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output');args=parser.parse_args(argv)
    root=(Path(__file__).resolve().parents[2]/args.output).resolve();eps=json.loads((root/'episodes.json').read_text());m=json.loads((root/'metrics.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text()); data=np.load(root/'predictions.npz')
    y=data['target'];p=data['prediction'];t=data['timestamp'];flags=data['intervention'];base=data['dataset_frame_linear']
    (root/'plots').mkdir(exist_ok=True);(root/'trajectories').mkdir(exist_ok=True)
    for i,e in enumerate(eps):
        start=e['offset'];end=start+e['frames'];sl=slice(start,end);x=t[sl];yt=y[sl];pt=p[sl];ft=flags[sl]
        fig,axes=plt.subplots(3,1,figsize=(12,8.5),sharex=True)
        fig.subplots_adjust(left=.11,right=.98,bottom=.085,top=.90,hspace=.34)
        status='成功' if e['success'] else '失败'
        fig.suptitle(f"{e['dataset']} / episode {e['episode_index']:03d} · {status}\nMSE {e['mse']:.5f}   MAE {e['mae']:.4f}   轨迹内 Spearman {e['spearman']:.3f}",y=.99,fontsize=13)
        axes[0].plot(x,yt,color=ORANGE,lw=2,label='实际 MC 回报（归一化）')
        axes[0].plot(x,pt,color=BLUE,lw=1,label='模型预测（原始逐帧）')
        axes[0].plot(x,base[sl],color=PURPLE,lw=1,ls='--',label='批次 + 帧序号对照（只拟合 train）')
        axes[0].set(ylim=(-1.02,.03),ylabel='Value');axes[0].legend(loc='lower right',fontsize=8,ncol=1)
        axes[1].plot(x,pt-yt,color=BLUE,lw=1);axes[1].axhline(0,color='#333',lw=.8)
        fr=e['max_error_frame'];axes[1].scatter([x[fr]],[pt[fr]-yt[fr]],color=ORANGE,s=22,zorder=3)
        axes[1].set(ylabel='预测 − 实际回报');lim=max(.05,float(np.abs(pt-yt).max())*1.15);axes[1].set_ylim(-lim,lim)
        a1=np.diff(pt)-np.diff(yt)
        axes[2].plot(x[:-1],a1,color=BLUE,lw=.7,alpha=.85,label='1 帧 TD 残差')
        if len(pt)>30:
            a30=(pt[30:]-pt[:-30]-(yt[30:]-yt[:-30]))/30
            axes[2].plot(x[:-30],a30,color=GREEN,lw=1,label='30 帧 TD 残差 / 30（仅诊断）')
        axes[2].axhline(0,color='#333',lw=.8);axes[2].set(ylabel='每帧尺度残差',xlabel='轨迹时间（秒）')
        axes[2].legend(loc='upper right',fontsize=8)
        for ax in axes:
            shade(ax,x,ft);ax.set_xlim(0,x[-1])
        fig.text(.11,.014,'紫灰阴影：人工介入标记。MC 标签线性上升由奖励定义决定；TD 残差并非动作优劣真值。',fontsize=9,color='#566274')
        fig.savefig(root/'plots'/f"{e['slug']}.png",dpi=150);plt.close(fig)
        prev=f"<a href='{eps[i-1]['slug']}.html'>← 上一条</a>" if i else ''
        nex=f"<a href='{eps[i+1]['slug']}.html'>下一条 →</a>" if i+1<len(eps) else ''
        values=json.dumps(dict(t=np.round(x,6).tolist(),y=np.round(yt,7).tolist(),p=np.round(pt,7).tolist(),intervention=ft.tolist()),separators=(',',':'))
        body=f'''<nav><a href="../index.html">全部 37 条轨迹</a>{prev}{nex}</nav>
<h1>{html.escape(e['dataset'])} / episode {e['episode_index']:03d} · {status}</h1>
<p class="muted">{e['frames']:,} 帧 · {e['duration_seconds']:.1f} 秒 · cam2 · 模型第 {m['checkpoint_epoch']} 轮</p>
<div class="notice">该轨迹来自固定测试集。原始逐帧预测未平滑。紫灰阴影表示人工介入，不能据此认定发生了错误或成功恢复。</div>
<section><video id="video" controls preload="metadata" src="../previews/{e['slug']}.webm"></video>
<p><label for="frame">选择帧／点击曲线定位视频</label><input id="frame" type="range" min="0" max="{e['frames']-1}" step="1" value="0"></p>
<div id="readout" aria-live="polite"></div></section>
<p><a href="../plots/{e['slug']}.png" target="_blank">打开高清曲线图</a> · <a href="../videos/{e['slug']}.mp4">原始 MP4</a></p>
<section class="plot" id="plot"><img src="../plots/{e['slug']}.png" alt="实际回报、预测、误差和 TD 残差曲线"><div id="cursor" class="cursor" style="left:11%"></div></section>
<section><h2>固定时间点与最大误差帧</h2><img src="../storyboards/{e['slug']}.jpg" alt="轨迹关键帧"><p class="muted">帧号：{', '.join(map(str,e['storyboard_frames']))}；最大误差帧：{fr}。这是预先规定位置加数值误差最大的帧，不代表人工确认的失败时刻。</p></section>
<footer>数值来源：predictions.npz / episodes.json。浏览器播放 VP9/WebM 预览，评估仍基于原始视频。</footer>
<script>const d={values};const v=document.getElementById('video'),s=document.getElementById('frame'),r=document.getElementById('readout'),c=document.getElementById('cursor');
function nearest(t){{let l=0,h=d.t.length-1;while(l<h){{const m=Math.floor((l+h)/2);if(d.t[m]<t)l=m+1;else h=m}}return l>0&&Math.abs(d.t[l-1]-t)<Math.abs(d.t[l]-t)?l-1:l}}
function show(i){{s.value=i;r.textContent=`帧 ${{i}} | ${{d.t[i].toFixed(2)}} 秒 | 预测 ${{d.p[i].toFixed(4)}} | 实际回报 ${{d.y[i].toFixed(4)}} | 误差 ${{(d.p[i]-d.y[i]).toFixed(4)}} | 人工介入 ${{d.intervention[i]?'是':'否'}}`;c.style.left=(11+87*d.t[i]/d.t[d.t.length-1])+'%'}}
function seek(i){{v.pause();v.currentTime=d.t[i];show(i)}}s.addEventListener('input',()=>seek(Number(s.value)));v.addEventListener('timeupdate',()=>show(nearest(v.currentTime)));
document.getElementById('plot').addEventListener('click',e=>{{const b=e.currentTarget.getBoundingClientRect();const f=Math.max(0,Math.min(1,((e.clientX-b.left)/b.width-.11)/.87));seek(nearest(f*d.t[d.t.length-1]))}});show(0);</script>'''
        (root/'trajectories'/f"{e['slug']}.html").write_text(page(e['slug'],body))
    # Global overview; no sampling of frames for error distributions.
    fig,axs=plt.subplots(2,2,figsize=(13,9));fig.subplots_adjust(hspace=.38,wspace=.30)
    axs[0,0].grid(False)
    im=axs[0,0].hexbin(y,p,gridsize=55,mincnt=1,bins='log',cmap='Blues',extent=(-1,0,-1,0))
    axs[0,0].plot([-1,0],[-1,0],color=ORANGE,lw=1);axs[0,0].set(xlabel='实际 MC 回报',ylabel='预测价值',title='测试集全部帧：预测与标签');
    with plt.rc_context({'axes.grid':False}):fig.colorbar(im,ax=axs[0,0],label='帧数（对数色标）')
    cal=m['calibration'];axs[0,1].plot([-1,0],[-1,0],color='#999',ls='--')
    axs[0,1].plot([r['prediction_mean'] for r in cal],[r['target_mean'] for r in cal],'o-',color=BLUE)
    axs[0,1].set(xlabel='每个固定区间的平均预测',ylabel='该区间的平均实际回报',title='回归校准：固定 10 个 value 区间',xlim=(-1,0),ylim=(-1,0))
    names=['模型','全局均值','批次均值','批次 + 帧序号'];overall=m['groups'][0]
    vals=[overall['mse']]+[overall['baselines'][k]['mse'] for k in ['global_mean','dataset_mean','dataset_frame_linear']]
    axs[1,0].bar(names,vals,color=[BLUE,'#9aabc0',PURPLE,GREEN]);axs[1,0].set(ylabel='帧加权 MSE',title='对照参数全部只由 train 拟合')
    for k,v in enumerate(vals):axs[1,0].text(k,v,f'{v:.5f}',ha='center',va='bottom',fontsize=9)
    for name,color in [('成功',BLUE),('失败',ORANGE)]:
        es=[e for e in eps if e['success']==(name=='成功')]
        axs[1,1].scatter([e['spearman'] for e in es],[e['mse'] for e in es],label=name,color=color,alpha=.8)
    axs[1,1].set(xlabel='轨迹内 Spearman（趋势一致性）',ylabel='单条轨迹 MSE',title='每个点 = 一条独立测试轨迹');axs[1,1].legend()
    fig.suptitle('固定 checkpoint 独立测试 · 37 条轨迹 / 34,759 帧',fontsize=16)
    fig.savefig(root/'overview.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # All trajectories on one page, sorted by test error for inspection only.
    order=sorted(eps,key=lambda e:e['mse'],reverse=True)
    fig,axes=plt.subplots(10,4,figsize=(16,24));fig.subplots_adjust(hspace=.75,wspace=.30,top=.96,bottom=.04)
    for ax,e in zip(axes.ravel(),order):
        sl=slice(e['offset'],e['offset']+e['frames']);ax.plot(t[sl],y[sl],color=ORANGE,lw=1);ax.plot(t[sl],p[sl],color=BLUE,lw=.6)
        ax.set_ylim(-1.02,.03);ax.set_title(f"{e['dataset'][5:]} / {e['episode_index']:03d} {'成功' if e['success'] else '失败'}\nMSE={e['mse']:.4f}",fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(order):]:ax.set_visible(False)
    fig.suptitle('全部测试轨迹，按 MSE 从高到低排列（仅用于检查）\n蓝：预测；橙：实际 MC 回报；横轴：秒',fontsize=16)
    fig.savefig(root/'all_trajectories.png',dpi=120);plt.close(fig)
    # Intervention event diagnostic; trajectories, not events, are the independent unit.
    fig,ax=plt.subplots(figsize=(10,4.5));event_episodes={}
    for ev in m['intervention_events']:
        ep=next(e for e in eps if e['slug']==ev['slug']);fr=ev['frame'];h=ev['window_frames'];start=ep['offset']
        values=p[start+fr-h:start+fr+h+1]-p[start+fr]
        xx=np.arange(-h,h+1)/ep['fps'];event_episodes.setdefault(ev['slug'],[]).append(np.interp(np.linspace(-1,1,61),xx,values))
    if event_episodes:
        traces=np.stack([np.mean(v,axis=0) for v in event_episodes.values()]);xx=np.linspace(-1,1,61)
        for trace in traces:ax.plot(xx,trace,lw=.8,alpha=.35,color=BLUE)
        ax.plot(xx,traces.mean(0),lw=2,color=BLUE,label='先轨迹内平均，再跨轨迹平均')
        ax.plot(xx,xx*.0075,ls='--',color=ORANGE,label='30fps / scale4000 的时间漂移参考')
        ax.legend(fontsize=9)
    ax.axvline(0,color='#444',ls=':');ax.axhline(0,color='#888',lw=.7)
    ax.set(xlabel='相对人工接管开始时间（秒）',ylabel='V(t) − V(接管开始)',title=f"接管前后：{len(m['intervention_events'])} 个有效事件 / {len(event_episodes)} 条轨迹（描述性，非因果证据）")
    fig.tight_layout();fig.savefig(root/'intervention_events.png',dpi=150);plt.close(fig)
    ci=m['bootstrap']['global_mean']['frame_mse_ci95'];best_baseline=min(overall['baselines'],key=lambda k:overall['baselines'][k]['mse'])
    contrast='模型优于批次/时间对照，但仍需检查跨场景泛化与优势噪声。' if overall['mse']<overall['baselines']['dataset_frame_linear']['mse'] else '模型未优于批次/时间对照，当前结果不足以证明可靠的视觉进展或优势信号。'
    table=''
    for e in order:
        table+=f'''<tr data-source="{e['dataset']}" data-outcome="{'success' if e['success'] else 'failure'}"><td><a href="trajectories/{e['slug']}.html">{e['dataset']} / {e['episode_index']:03d}</a></td><td>{'成功' if e['success'] else '失败'}</td><td>{e['frames']}</td><td>{e['mse']:.5f}</td><td>{e['mae']:.4f}</td><td>{e['spearman']:.3f}</td><td>{e['baselines']['dataset_frame_linear']['mse']:.5f}</td><td>{100*e['temporal']['1']['negative_delta_fraction']:.1f}%</td></tr>'''
    groups=''.join(f"<tr><td>{html.escape(g['group'])}</td><td>{g['episodes']}</td><td>{g['frames']}</td><td>{g['mse']:.5f}</td><td>{g['mae']:.4f}</td><td>{g['baselines']['dataset_frame_linear']['mse']:.5f}</td></tr>" for g in m['groups'])
    options=''.join(f'<option value="{d}">{d}</option>' for d in sorted({e['dataset'] for e in eps}))
    body=f'''<h1>当前 checkpoint · 独立测试报告</h1><p class="muted">第 {m['checkpoint_epoch']} 轮模型 · 37 条测试轨迹 · 34,759 帧 · 成功 24 / 失败 13 · cam2</p>
<div class="notice"><b>{contrast}</b><br>本报告固定 checkpoint，不用测试结果选择模型或阈值。人工接管和价值变化的关系仅作描述。</div>
<section><span class="metric"><b>{overall['mse']:.5f}</b>帧加权 MSE</span><span class="metric"><b>{overall['rmse']:.4f}</b>RMSE</span><span class="metric"><b>{overall['mae']:.4f}</b>MAE</span><span class="metric"><b>{overall['episode_macro_mse']:.5f}</b>每条轨迹等权 MSE</span><p>按 episode 重采样的 MSE 95% 区间：[{ci[0]:.5f}, {ci[1]:.5f}]。置信区间只覆盖当前同批次划分，不保证跨日期或跨场景泛化。</p></section>
<section><h2>总体表现与对照</h2><img src="overview.png" alt="测试集总体指标与对照图"><p>批次/帧序号对照不读取图像，所有参数仅由训练集拟合；不使用该测试轨迹的最终长度或结局。它用于诊断采集批次与时间线索，不是部署方案。</p></section>
<section><h2>分组误差</h2><div class="scroll"><table><thead><tr><th>分组</th><th>轨迹</th><th>帧</th><th>模型 MSE</th><th>MAE</th><th>批次+帧 MSE</th></tr></thead><tbody>{groups}</tbody></table></div></section>
<section><h2>逐轨迹浏览</h2><p>按误差从高到低排列，点击轨迹进入同步视频与曲线页面。</p><label>批次 <select id="source"><option value="all">全部</option>{options}</select></label><label>结局 <select id="outcome"><option value="all">全部</option><option value="success">成功</option><option value="failure">失败</option></select></label><p id="count"></p><div class="scroll"><table><thead><tr><th>轨迹</th><th>结局</th><th>帧数</th><th>MSE</th><th>MAE</th><th>轨迹内 rho</th><th>批次+帧 MSE</th><th>价值下降帧占比</th></tr></thead><tbody id="episodes">{table}</tbody></table></div></section>
<section><h2>人工接管前后</h2><img src="intervention_events.png" alt="人工接管前后价值变化"><p>人工接管不一定表示错误发生，结束也不等于恢复成功。现有标签缺少独立的失误/恢复时刻标注，不能把这张图解释为因果关系或恢复检测准确率。</p></section>
<section><h2>全部轨迹缩略图</h2><a href="all_trajectories.png">打开高清完整图</a><img loading="lazy" src="all_trajectories.png" alt="37条轨迹全部曲线"></section>
<footer>checkpoint SHA256：<code>{m['checkpoint_sha256']}</code><br><a href="report.md">文字报告</a> · <a href="metrics.json">全部指标 JSON</a> · <a href="episodes.json">逐轨迹指标</a> · <a href="protocol.json">评估协议</a> · <a href="audit.json">完整性审计</a><br>浏览器播放逐帧同长度的 VP9/WebM 预览；原始 cam2 文件仅作符号链接，未改写。</footer>
<script>const source=document.getElementById('source'),outcome=document.getElementById('outcome');function filter(){{let n=0;for(const r of document.querySelectorAll('#episodes tr')){{const ok=(source.value==='all'||r.dataset.source===source.value)&&(outcome.value==='all'||r.dataset.outcome===outcome.value);r.hidden=!ok;if(ok)n++}}document.getElementById('count').textContent='显示 '+n+' / 37 条轨迹'}}source.onchange=outcome.onchange=filter;filter();</script>'''
    (root/'index.html').write_text(page('独立测试报告',body));print(root/'index.html')


if __name__=='__main__':main()
