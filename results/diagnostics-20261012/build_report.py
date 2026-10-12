"""Read-only analysis of the uploaded diagnostics bundle; produce a static report."""
from collections import Counter
from pathlib import Path
import base64
import hashlib
import html
import json
import statistics
import argparse
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle-dir', type=Path, required=True,
                    help='Directory containing the extracted vllm-diag-complete.tar.gz')
parser.add_argument('--output-dir', type=Path, required=True,
                    help='New analysis output directory; raw input is never modified')
args = parser.parse_args()
ROOT = args.bundle_dir.resolve()
HERE = args.output_dir.resolve()
HERE.mkdir(parents=True, exist_ok=False)
REPO = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, str(REPO / 'analyze_trace_dynamics.py'),
                str(ROOT / 'results/trace-diag-experiment-96-r1'),
                '--output-dir', str(HERE / 'reproduced')], check=True)
RUN = ROOT / 'results/trace-diag-experiment-96-r1'
SUMMARY = json.loads((HERE / 'reproduced/summary.json').read_text())['runs'][0]
MANIFEST = json.loads((RUN / 'manifest.json').read_text())
START = MANIFEST['measurement_start']
EVENTS = [json.loads(x) for x in (RUN / 'gateway.jsonl').read_text().splitlines()]
CLIENTS = json.loads((RUN / 'client.json').read_text())['requests']
LEDGER = [json.loads(x) for x in (HERE / 'reproduced/trace-diag-experiment-96-r1-requests.jsonl').read_text().splitlines()]
ENGINE = [json.loads(x) for x in (RUN / 'engine_observations.jsonl').read_text().splitlines()]
STATE = json.loads((RUN / 'state_observations.json').read_text())
GC = [e for e in EVENTS if e['event'] == 'gc_span']
GC2 = [e for e in GC if e['generation'] == 2]
MONITORS = [m for m in SUMMARY['diagnostics']['monitor_overdue_intervals'] if m['overdue_s'] > .3 and 0 <= m['due_s'] < 300]

def overlap(a, b, c, d):
    return max(0., min(b, d) - max(a, c))

stall_table = []
for monitor in MONITORS:
    g = max(GC2, key=lambda x: overlap(monitor['due_s'], monitor['emission_s'], x['span_start']-START, x['span_end']-START))
    a, b = g['span_start']-START, g['span_end']-START
    covered = overlap(monitor['due_s'], monitor['emission_s'], a, b)
    phases = []
    for e in EVENTS:
        if e['event'] == 'request_diagnostics':
            for name in ('json_parse', 'cost_estimation', 'client_construction'):
                p = e['phases'][name]
                if p['start'] is not None and p['start'] <= g['span_start'] <= p['end']:
                    phases.append(dict(request_id=e['request_id'], phase=name, elapsed_s=p['end']-p['start']))
    stall_table.append(dict(gc_start_s=a, gc_end_s=b, generation=g['generation'],
        monitor_due_s=monitor['due_s'], monitor_resumed_s=monitor['emission_s'],
        monitor_overdue_ms=1000*monitor['overdue_s'], gc_wall_ms=1000*(b-a),
        gc_thread_cpu_ms=1000*g['callback_thread_cpu_seconds'], overlap_ms=1000*covered,
        overlap_fraction=covered/monitor['overdue_s'], collected=g['collected'],
        on_loop_thread=g['on_loop_thread'], containing_sync_phases=phases))

success = [r for r in LEDGER if r['success']]
tail = sorted(success, key=lambda r:r['ttft_s'], reverse=True)[:21]
tail_gc2 = sum(any(overlap(r['dispatch_s'], r['gateway_content_s'], g['span_start']-START, g['span_end']-START)>0 for g in GC2) for r in tail)
late = [dict(scheduled_s=c['scheduled_at']-START, started_s=c['started_at']-START,
             lag_ms=1000*(c['started_at']-c['scheduled_at']), success=c['success'], rejected=c['rejected'])
        for c in CLIENTS if c['started_at']-c['scheduled_at']>.05]
value = lambda e,k: sum(x['value'] for x in e['metrics'][k]['samples']
                       if e['metrics'][k]['type']=='gauge' or x['name'].endswith('_total'))
construction = [e['phases']['client_construction']['end']-e['phases']['client_construction']['start']
                for e in EVENTS if e['event']=='request_diagnostics' and e['phases']['client_construction']['status']=='completed']
facts = dict(archive_sha256='32676c162da7182e1b22693f1e0eff899ad7179e10bf28a3074b7b9c0324c454',
    run_id=MANIFEST['run_id'], measurement_start=START, commit=MANIFEST['git_commit'],
    report=SUMMARY['report'], original_report_reproduced=SUMMARY['original_report_reproduced'],
    runtime_code_matches=SUMMARY['runtime_code_matches'], stalls=stall_table,
    gc_by_generation={str(gen):dict(count=sum(g['generation']==gen for g in GC),
                        wall_seconds=sum(g['span_end']-g['span_start'] for g in GC if g['generation']==gen)) for gen in range(3)},
    weighted_gc2_overlap_fraction=sum(x['overlap_ms'] for x in stall_table)/sum(x['monitor_overdue_ms'] for x in stall_table),
    tail=dict(count=21, mean_ttft_s=statistics.mean(r['ttft_s'] for r in tail),
              post_dispatch_fraction=SUMMARY['slowest_one_percent']['post_dispatch_fraction'],
              post_dispatch_gc2_overlap_count=tail_gc2),
    late_requests=late, waiting_sample_max=max(value(e,'waiting') for e in ENGINE),
    preemptions_delta=value(ENGINE[-1],'preemptions')-value(ENGINE[0],'preemptions'),
    construction=dict(count=len(construction),median_ms=1000*statistics.median(construction),
                      mean_ms=1000*statistics.mean(construction),max_ms=1000*max(construction)),
    raw_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file()})
assert len(GC2)==len(MONITORS)==4 and tail_gc2==0 and len(late)==9
(HERE/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(3,1,figsize=(13,9),sharex=True)
axes[0].scatter([r['started_s'] for r in success],[r['ttft_s'] for r in success],s=8,alpha=.55,color='#26749b',label='Successful requests')
axes[0].axhline(.5,color='#9a6630',ls='--',label='TTFT target: 0.5 s')
axes[0].set_ylabel('Client TTFT (s)');axes[0].legend(loc='upper right')
t=[e['sent_at']-START for e in ENGINE]
axes[1].plot(t,[value(e,'waiting') for e in ENGINE],color='#206eaa',label='vLLM waiting (1 s samples)')
axes[1].plot([o['metrics']['state_sampled_at']-START for o in STATE],
             [o['metrics']['awaiting_first_content'] for o in STATE],color='#aa793b',alpha=.8,label='Gateway awaiting first content')
axes[1].set_ylabel('Requests');axes[1].legend(loc='upper right')
axes[2].scatter([c['scheduled_at']-START for c in CLIENTS],
                [1000*(c['started_at']-c['scheduled_at']) for c in CLIENTS],s=7,alpha=.55,color='#26749b',label='Client scheduling lag')
axes[2].plot(t,[1000*(e['resources']['host_cpu']['sample_start']-e['received_at']) for e in ENGINE],
             color='#aa793b',alpha=.8,label='Collector post-body processing elapsed time')
axes[2].axhline(50,color='#9a6630',ls='--',label='Declared max scheduling lag: 50 ms')
axes[2].set_ylabel('Elapsed / lag (ms)');axes[2].set_xlabel('Seconds since measurement start');axes[2].legend(loc='upper right')
for ax in axes:
    for i,g in enumerate(GC2):
        ax.axvspan(g['span_start']-START,g['span_end']-START,color='#b64e58',alpha=.3)
    ax.set_xlim(0,300);ax.grid(alpha=.15)
fig.suptitle('Fixed budget 96: gateway GC, engine backlog and client launch lag\nRed bands = measured generation-2 GC spans on gateway loop thread',fontsize=14)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(HERE/'diagnostic_overview.png',dpi=160);plt.close(fig)

fig, ax = plt.subplots(figsize=(10,5.4))
y=np.arange(4);ax.barh(y+.16,[x['monitor_overdue_ms'] for x in stall_table],height=.28,color='#487aa1',label='Monitor overdue duration')
ax.barh(y-.16,[x['overlap_ms'] for x in stall_table],height=.28,color='#b76c64',label='GC2 actual-span overlap with overdue interval')
ax.set_yticks(y,[f"GC starts at {x['gc_start_s']:.3f} s" for x in stall_table]);ax.invert_yaxis()
for i,x in enumerate(stall_table):ax.text(x['monitor_overdue_ms']+12,i,f"{100*x['overlap_fraction']:.1f}% overlap",va='center')
ax.set_xlim(0,850);ax.set_xlabel('Milliseconds');ax.grid(axis='x',alpha=.15);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17))
ax.set_title('All four large gateway delays overlap long GC2 spans\nOverlap is measured time, not a predicted improvement from disabling GC')
fig.tight_layout();fig.savefig(HERE/'gc_overlap.png',dpi=160);plt.close(fig)

def image(name):
    return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((HERE/name).read_bytes()).decode()+'">'

table=''.join('<tr><td>{gc_start_s:.3f}</td><td>{monitor_overdue_ms:.1f}</td><td>{gc_wall_ms:.1f}</td><td>{gc_thread_cpu_ms:.1f}</td><td>{percent:.1f}%</td><td>{collected:,}</td></tr>'.format(**x,percent=100*x['overlap_fraction']) for x in stall_table)
report=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>vLLM diagnostics — 2026-10-12</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 24px;line-height:1.7;color:#24313b}}h1,h2{{line-height:1.3}}h2{{margin-top:34px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%;height:auto}}code{{background:#f2f4f6;padding:2px 4px}}.callout{{padding:18px;background:#edf5f7;border-left:4px solid #437e95}}.small{{font-size:13px;color:#58636c}}@media print{{body{{margin:0}}h2{{break-after:avoid}}img,table{{break-inside:avoid}}}}</style>
<h1>新 telemetry 已確認：長 GC 是四次 gateway 停頓的重要成分</h1>
<p class="callout">上傳資料完整，五份審核報告與分析 summary 都能精確重現。GPU 已無需保留。四次長停頓均重疊 event-loop thread 的 generation-2 GC；但多秒 TTFT 波峰、client scheduling lag 仍是另外需要處理的現象。</p>
<p>Capture：2026-10-12，RTX 4090 / driver 580.178.04 / Python 3.12.3 / vLLM 0.30.0 / Torch 2.13.0+cu130。Qwen3-4B-Instruct-2507；context 8192；prefix caching 由啟動紀錄確認關閉；budget 96；output cap 128；main <code>f1f7cef</code>。</p>
<h2>1. 資料資格與保存</h2>
<ul><li>Smoke 與五分鐘實驗：client integrity、diagnostic integrity 均通過；GC 邊界／shutdown 完整。</li><li>五分鐘 load fidelity 未通過：9 / 7,643 個 request 的 scheduling lag 超過 50 ms；最大 146.527 ms。原判定保留。</li><li>原生 CPU suite log：135 passed in 31.10s。External-only 的 success/rejected/failed 三份 capture 都重新通過完整審核；有效 capture 在測試刻意破壞臨時原件之前另存。</li><li>五份 report.json 都與獨立重算結果相同；新的 summary.json 也精確重現。Runtime code fingerprints 全相符。</li><li>測量：2,063 成功、5,580 拒絕、0 失敗；409 個完成串流且達 TTFT SLO，goodput 1.363/s，offered fraction 5.351%。Gateway completed 2,066 包含三個 warmup。</li></ul>
<p>原始壓縮檔 SHA-256：<code>{facts['archive_sha256']}</code>。所有分析寫在原件之外，原始 report／trace／log 未改寫。</p>
{image('diagnostic_overview.png')}
<h2>2. 四次 gateway 停頓與長 GC 的直接時間證據</h2>
<table><tr><th>GC 開始（s）</th><th>Monitor overdue（ms）</th><th>完整 GC span（ms）</th><th>Span 內 thread CPU（ms）</th><th>Overdue 與 GC 重疊</th><th>Collected objects</th></tr>{table}</table>
<p>650 個 GC spans（含 warmup／drain）中，592 個 generation 0、54 個 generation 1、只有 4 個 generation 2。四個 gen2 spans 共 1.698 s，與四次 overdue interval 的重疊共覆蓋 {100*facts['weighted_gc2_overlap_fraction']:.1f}% 的總 overdue 時間；GC span 內 thread CPU 幾乎等於 wall time。這比單純『約每分鐘一次』提供更直接的證據，支持長 GC 是本次停頓的主要成分之一。</p>
{image('gc_overlap.png')}
<p>Overdue interval 是 timer 應執行至實際恢復的範圍，不是連續 CPU blocking 的精確邊界。GC span 內 CPU 也不等於 heap scan 的獨占 CPU；它可能包括 finalizers／其他 GC callbacks。未重疊的時間不可自動歸因於 GC，也不能將 66.2% 解讀成停用 GC 的改善幅度。</p>
<p>120.406 s 的 GC 開始落在一個被拒絕 request 的 cost_estimation phase；249.716 s 的 GC 落在 client_construction phase（該 phase 502.8 ms，GC 484.6 ms）。另外兩次無法從已記錄的同步 phase 確定觸發位置。觸發所在 phase 不能證明被回收物件來自該 request。</p>
<h2>3. 為甚麼有週期感？</h2>
<p>三次相鄰 GC 起點間隔為 61.888、66.436、62.874 s，並非精確固定時鐘。各次之間 accepted 累計增量約為 427、462、422，也不支持固定 500 個 admitted request 的規則。記錄的 GC thresholds 為 (700, 10, 10)。Python 3.12 依物件配置／回收及世代條件決定自動 GC；穩定的物件建立速率可以產生近似週期。這是與資料相符的機制解釋，沒有回收物件類型或 allocation stack，仍無法指定唯一物件來源。</p>
<h2>4. 多秒 TTFT 波峰仍有 engine backlog</h2>
<p>最慢 21 個成功 request 的平均 client TTFT 為 3.260 s，其中 post-dispatch 平均為 3.055 s（93.72%）。它們的 dispatch → gateway first-content 區間全部沒有直接重疊四次長 gen2 GC；小 GC 的總重疊只有毫秒級。</p>
<p>例如 [186,189) s 放行 43 個 request、71,461 個估計 input tokens；187 秒內完成 33 個 request。Gateway reserved cost 在 187.529 s 為 94.876、188.532 s 為 95.740，但 awaiting-first 從 4 增到 30、streaming proxy 從 46 減到 14。vLLM waiting 在 187.815 s 的樣本為 0，188.816 s 升到 24。成本線近乎平坦，仍能掩蓋 prefill/backlog 的劇烈改變。</p>
<p>[0,2) s 初始放行 44 個、72,296 input tokens；1.754 s 的 waiting 樣本為 21，也有多秒 TTFT，而當時沒有長 gen2 GC。225 秒內又放行 21 個、45,038 input tokens，waiting 樣本為 12，並出現 3.266 s TTFT。這些資料支持『集中補入 work 與 engine 排隊』，不等於證明 GPU saturation，也沒有 request-linked prefill 時間。GC 後積壓恢復可能放大入流，沒有直接重疊不代表 GC 完全沒有間接影響。</p>
<h2>5. Load fidelity 失敗沒有被四次 gateway GC 解釋</h2>
<p>9 個超過 50 ms 的 client launch delay 分成兩群：97.667–97.752 s（4 個）、177.892–177.929 s（5 個）。它們不在四次長 gateway gen2 GC 的時間區間中；8 個最後被拒絕、1 個成功且達 TTFT SLO。</p>
<p>第二群附近，external collector 在 runner 的工作執行緒從 177.831 s 收完 HTTP body 到 177.998 s 開始 host-resource reads，耗時 166.7 ms，與 delayed launches 的區間重疊。第一群的主要延遲則發生在當次 collector HTTP request 發出之前。前者提示 runner 內的 Python parsing／GC／GIL 或 scheduling 是值得區分的候選；目前沒有 runner GC callback 或 sampler thread CPU／stack，不能指定是哪一個，也不能宣稱兩群都由 collector 造成。</p>
<h2>6. 下一步如何服務 MPC 目標</h2>
<ol><li><strong>先把明確的服務開銷列成可驗證改動。</strong>每個 admitted request 都建立一個 AsyncClient；2,066 次 construction 的中位 elapsed 17.81 ms。Reusable AsyncClient／connection pooling 是合理的下一個可選實作，但沒有證據證明它是所有 GC 物件的來源。必須保留 response-close、reservation-release、cancellation、headers／cookie isolation 與 pool-capacity 語意，再做有同樣監測的比較。</li>
<li><strong>修正量測工具的獨立性。</strong>External sampler 的 worker thread 仍與 runner 共享 interpreter／GIL。可以考慮用獨立 process 做 sampling，保留正確 runner PID、process identity、實際時間與 persistence audit；另外只加入最小 runner GC／thread-CPU probe，區分兩群 launch delay。不要藉由改大 50 ms 門檻使結果合格。</li>
<li><strong>同步準備 budget-response identification。</strong>先用固定 budget 掃描找工作範圍，再加入預先宣告的 step／PRBS budget schedule 與實際 applied timestamps。候選狀態包含 awaiting-first、streaming proxy、reserved cost，並以 engine waiting/running 校對；擾動包含已觀測 arrival rate 與 input demand。僅固定 budget 96 的這次資料不能辨識控制增益、決定更新頻率或預測 horizon。</li>
<li><strong>在獨立資料上驗證模型與 MPC。</strong>目標維持 scheduled-arrival TTFT SLO goodput，同時報告 rejection／failure／E2E。正式 matched policy comparison 保留原 fidelity gates；目前資料可以作描述診斷與候選模型探索。無需把找完每個 pause 的根因變成 MPC 模型工作的前置條件。</li></ol>
<p>沒有 matched diagnostics-off/on experiment，不能把本次 409 個 SLO success 相對於舊 r1/r2 的變化歸因於 observer overhead 或任何單一因素。沒有量測 GPU utilization／per-request GPU execution；observed preemptions counter delta 為 0。</p>
<h2>來源與重現方式</h2>
<p>主要證據：上傳 bundle 的 <code>results/trace-diag-experiment-96-r1/{{manifest.json,client.json,gateway.jsonl,engine_observations.jsonl,state_observations.json,report.json}}</code>；startup <code>launch.json/runtime.json</code>；native validation status/logs。相對時間以 measurement_start = {START:.9f} 的同主機 monotonic clock 正規化。GC overlap 取實際 span 的 interval intersection，沒有使用較晚 log emission 時間。</p>
<p>分析程式：<code>analyze_trace_pilot.py</code>（在臨時複本審核）及 <code>analyze_trace_dynamics.py</code>（獨立輸出）；五份報告精確重現，原 uploaded summary 精確重現。細部事實另存 <code>facts.json</code>。Primary documentation：<a href="https://docs.python.org/3.12/library/gc.html">CPython 3.12 GC</a>、<a href="https://www.python-httpx.org/async/">HTTPX Async Support</a>、<a href="https://docs.vllm.ai/en/v0.30.0/design/metrics/">vLLM 0.30 metrics</a>。</p>
<details><summary>機讀事實與原始檔案 SHA-256</summary><pre>{html.escape(json.dumps(facts,indent=2,ensure_ascii=False))}</pre></details>
<details><summary>重現本報告的分析程式</summary><pre>{html.escape(Path(__file__).read_text())}</pre></details>
</html>'''
(HERE/'vllm-diagnostics-report-20261012.html').write_text(report)
print(json.dumps({k:facts[k] for k in ['weighted_gc2_overlap_fraction','tail','construction']},indent=2))
print(HERE/'vllm-diagnostics-report-20261012.html')
