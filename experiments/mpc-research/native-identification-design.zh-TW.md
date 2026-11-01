# PR32：Native GPU 辨識、真實 HTTP 驗證與負面結果診斷

## 0. 執行邊界與完成方式

本次由 Mayor 獨立分析，沒有重新派出 Polecat。開始時唯讀確認：
pr32 四個 Polecat 都是 idle/recovery-needed，沒有 pr32 tmux Session，
Merge Queue 為空。**未呼叫 `gt done`，未 Push、Merge、修改 Remote，
未啟動 GPU 或付費服務。** 原先 Onyx 三次違規 Push 的分支與證據保留，
不以本次安全完成掩蓋先前事故。

替代原生完成流程採用：

1. 在既有獨立分支 `research/pr32-mpc-8e92371` 直接執行 CPU-only 分析。
2. 只 Stage 指定檔案，檢查後作本機 Commit；不執行任何完成／整合佇列命令。
3. 以 Mayor 的 Beads 註記和本機 Todo 記錄結果，GPU Bead 保持 blocked。
4. 不清除 Worker Worktree、研究檔案、失敗紀錄或歷史提交。

這不是聲稱 Gas Town 正常 `gt done` 能保證不 Push；先前實際證明不能。
本次選擇不進入那條流程，也沒有靠修改 Git Remote 或使用新的完成 Hook 避險。
本機 Commit 以單次 `git -c core.hooksPath=/dev/null commit` 禁用 Hook，
避免間接執行外部完成動作；不寫入 Git 設定。必要測試與 Diff 檢查另行執行。

## 1. 證據、範圍與可重現命令

產品來源維持 `de36581f16fc39784297aadcd6186883aaf1d35a`，
上一輪交付 Commit 為 `3234459`。**沒有修改 MPC 係數、Action 集合、
Horizon、Scenario 數量、100ms Deadline 或 Fallback。**

本次新增的是實驗目錄的純離線工具 `native_id_preparation.py`、
其測試、診斷結果與本報告。工具沒有網路、GPU、租機或 Git 完成介面。

```bash
cd /home/daniel_lin/gt/pr32/mayor/mpc-8e92371
PY=/home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/pr32-mpc-py312/bin/python
"$PY" experiments/mpc-research/native_id_preparation.py \
  --manifest-only --output /tmp/pr32-native-id-manifest.json
```

不要用這個命令覆寫已保存的 Results。它只產生設計 Template，不會執行實驗。
最終診斷檔：
`results/mpc-research-cpu/native-id-diagnostics-final-validated.json`。
前兩次工作版本的診斷結果亦保留，不挑選較好 Timing 當作性能證明。

本次最終針對準備工具、Controller、Runtime、Actuator 和 Gateway 的回歸：
**226 passed，1.40s，其中14項是新增準備測試**。
證據：`native-id-final-readiness-junit.xml` 及同名 pytest log。
涵蓋全 Offered 分母、Late Failure、Reject、跨 Clock 先拒絕、
SSE 完成／身分／錯誤條件、重複／遺漏、Right Censoring 不冒充完成、
共同 Cohort 的 Horizon 比較及成對干預設計。
之前的完整1976-pass Suite 不冒稱已包含這14項新測試。

## 2. 已確認：MPC 輸給 Fixed16，不是單一 Deadline 問題

原始75個凍結合成 Episodes 仍然有效，不重跑／調參追分：
MPC4.4222、Fixed16為5.5778 good req/s。
Fixed16 在 stationary、burst、shift 三類平均都較好。

**原始10次 Fallback 全在 burst**：
Seeds10001–10005 分別2、3、1、1、3次；
相關 Episode 的最大 Controller Wall 約102.75–103.91ms。
Stationary／shift 沒有這類 Fallback，MPC 仍然輸。因此
「只要消除 Deadline 就能解釋並消除全部差距」已被現有資料否定。

### 2.1 模型與 Truth 的結構差異是事實；個別貢獻尚未識別

來源：`mpc_lab/contracts.py:ModelParameters`、
`model.py:_build/rollout/generate_scenarios`、
`benchmark.py:run_episode`。

| 項目 | 預測模型 | 合成 Truth | 判讀 |
|---|---|---|---|
| 基礎 Decode | .004s/token，即250 token/s | 100 token/s | 不是同一 Service Law |
| Decode 負載項 | `1+.02D` | `1+.04D+W/100000` | 缺少 Prefill Backlog 對 Decode 的影響 |
| Prefill | `.03+.00005p`，乘`1+.02D` | `(p+200)/14000`，乘`1+.08D`，另有25ms Grid | 干擾與尺度不一致 |
| 未來 Output | 一律 Requested Maximum | Uniform[max/4,max] | 未辨識 EOS，不能把 Maximum 當實際長度 |
| 既有 Awaiting | 重置為完整 Prefill Work | 已有部分工作完成 | 隱藏進度重建有偏 |
| 既有 Decode | 依名目速率從 Age 扣 Tokens，至少剩1個 | 實際歷史服務率依當時負載變化 | Age 不足以唯一決定 Remaining Work |
| Scenario | Poisson、零傳輸延遲、預設成功 | Burst／Shift、2% Latent Failure | Distribution 與 Label 機制不同 |

例如只比較 D=16、W=0、Scale=1 的服務公式：
預測每 Stream 約189.4 token/s，Truth 約61.0。
這是公式推導，**不是 RTX4090 實測**。
4個 Scenario 的 Scale 實際為`.8,1,1.3,.8`；它們不是校準後的風險分布。
不同偏差可能互相抵消，不能僅看其中一項宣稱已找到主要原因。

本次另做6個 Development Trace 的 **Known-Future/EOS 離線反例檢查**：
固定 Budget、相同12s Offered Cohort、20s Drain，將實際未來與 EOS
只交給診斷用 Model Rollout，排除 Forecast 不知道未來的因素。
這不是可部署 Baseline，也沒有用 Held-out 資料。

| Seed／Workload | Truth 的 Good(C32)-Good(C16) | Model 的同一差值 |
|---|---:|---:|
| 1 stationary | -31 | -78 |
| 1 burst | -57 | -48 |
| 1 shift | -37 | -67 |
| 2 stationary | -30 | -75 |
| 2 burst | -16 | -45 |
| 2 shift | -27 | -46 |

**六例中，模型與 Truth 都認為固定16較好。**
所以不能籠統宣稱「模型必然把 Fixed32 排在 Fixed16 前」。
但預測的絕對 Good Count 與 Effect Size 明顯不準。
目前較合理的未決解釋是：Online State 重建、到達／EOS 預測、
有限 Cohort／Tail、短計畫和 Replanning 共同產生錯誤動作；
尚未有介入消融把每個因素的閉迴路損失分離。

另一項確定限制：合成 Benchmark 的 Plant 時間不隨求解 Wall Time 前進，
Budget 被視為在該邏輯 Tick 立即套用。CPU Cost 雖另行報告，
它沒有完整模擬真實的計算／HTTP／套用延遲。這不是 Native 可部署證據。

## 3. 已確認：多步的模型內優勢，不等於實際首步優勢

One-move 和 MPC 都使用同一5s 評分窗／10s Tail，差別是
1或3個可最佳化的1s Slot；One-move **不是只看未來1s**。
三步候選包含 `[a,16,16]`，因此在相同 State／Scenarios、完整搜尋時，
其最佳模型分數不低於 One-move 對應候選。

在事前指定的9個 Development Checkpoints，離線完整比較相同 Scenarios：
7個有較高 Open-loop Score，但只有2個改變首個 Action；
這2個又分別是 stationary／shift 在變化前的相同 Prefix，不是獨立樣本。
這些數字不能解讀成普遍機率。

未來計畫後兩步不會直接執行，下一秒重新最佳化。
因此更高的計畫分數可能只是尚未實現的未來承諾。
此外，8個計畫比2個計畫更容易挑到模型／4情境抽樣誤差的樂觀極值，
是待驗證的 Optimizer's Curse 假設，不是本次已證的損失比例。
固定兩個 Action、降 Cap 不驅逐既有 Requests，也限制了短期可操控性。

需要從 Native Trace 估計：
Budget→Admission→Reservation Residence→First Content／Terminal 的延遲，
以及1／3步是否真正產生不同首步、不同風險與不同效益。
不先把 Horizon 增加到5或換 Solver 來製造較高模型分數。

## 4. Deadline 成因：可確認的計算路徑與尚缺的歸因

2個 Action、3步、4情境意味最多32次 Rollout；
One-move 是8次，工作量增加約4倍，不是免費增加兩步。
生成 Scenario 延伸15s，高到達率會放大大量未計分 Tail Arrivals。
這與 burst 集中出現 Deadline 的事實一致，但原始凍結檔沒有每次
Snapshot／Scenario／OS Trace，無法精確重建那10次超時。

本次最終 Development-only 72個決策全選出結果，平均 Planner41.97ms；
Rollout 累計時間占 Planner 約95.2%。前一工作版本在相同 Development
設定中觀測到1次 burst 超時：Seed2、Tick6、2245個 Scenario Offers、
10個 Active Requests、開始32次 Rollout、完整計畫7個，
完成的 Rollout Events 合計28651，耗時100.081ms。
另一次工作版本沒有超時。Timing 有擾動，這不是凍結實驗的替代結果。

程式以 Cooperative Check 判 Deadline，不是 OS 硬即時；
超時會丟棄不完整搜尋並回 Budget16，不挑有利 Partial Best。
這個 Fallback 可能保護而非傷害 Goodput，不能直接把它計作全部損失。
`choose_plan` 的100ms 也不包含完整 GET、Scenario Generation、
序列化和 Actuation；Native 必須另量完整路徑。

必收 Timing：GET RTT／State Age、Generation、Validation／Identity、
每計畫／情境 Rollout、Events／Offers／Active、整體 CPU／Wall、
Completed Plans、Fallback、POST RTT、實際 Applied At。
同時記錄 CPU Affinity、Cgroup Quota／Throttling、Runqueue、
GC／Event-loop Lag。沒有這些就不能區分工作量與排程干擾。
本次不放寬100ms，也不以刪掉 Late Samples 修飾結果。

## 5. Native 實驗前已確認的介面阻礙

**不能直接沿用 Raw-token Bridge：** `gateway.py` 要求
`PREFILL_LAB_RAW_COMPLETIONS` 使用 `protected`，
而 `mpc_access` 要求 `cost_aware`。應使用標準 Chat HTTP、
相同 Tokenizer／Chat Template，收集 Gateway 的估計 Prompt Tokens
和 vLLM 實際 Prompt Tokens，不把兩者混為一談。
Prompt Target 若不能準確達到，事前保存實際 Token Manifest，不偷偷重標。

**不能偽造 Calibration 來啟動辨識：** 現有 `ActuationPermit`
要求五種 Native Evidence，包含 Calibration；它是部署 MPC 的 Gate，
不能為收集 Calibration 填假 Hash。Native ID 應使用獨立、明確授權的
實驗 Session／單一 Writer，僅允許16／32，保留相同 CAS／Expiry／Pending
語意並標記 `unfitted`；不得因此宣稱 MPC 已校準。
本次沒有新增可對外 POST 的 Runner，也沒有繞過現有 Gate。

現有 `live_http.probe` 主要保存 First／Finished，沒有完整 Last Content、
Usage／每 Token Timestamp。本次的離線 Auditor 不能取代 Collector。
在真正執行前必須完成 Collector Schema Mapping 和無 GPU 的 Mock 合約驗證；
此交付是設計與驗證準備，不是假稱已有已驗收的 Native Runner。

## 6. 高資訊量4090 Pilot：不是性能勝負賽

### 6.1 設計與變因

一張 RTX4090；Qwen/Qwen3-4B-Instruct-2507，固定 Model/Tokenizer Revision、
vLLM0.30.0、Torch2.13.0+cu130、Python3.12、CUDA／Driver Image Digest。
TP1、Context8192、B2048、Max Seqs32，固定 Graph／Async／Chunked Prefill
設定，Prefix Cache／Speculation 關閉。這些是待實機確認的計畫 Pin。

只干預 Weighted Admission Budget16／32；不混入 Engine Token Budget、
物理 Concurrency、Priority 或 Runtime Scheduler 改動。
四個因子格：Short/Long Prompt × Short/Long Maximum Output：
`128/4096` × `64/512`，Natural EOS，不強迫輸出長度。
Test 的 Long Prompt 改7168，仍滿足 `p+M≤8192`。

48 Episodes =4格×6個 Seed Blocks×2個互補 Action Sequences。
每對共用 Offered Tape，四格共用該 Seed 的 Arrival Tape；
Action 獨立於 State 事前隨機化，Dwell 循環2／5／10s，
另一 Arm 使用互補16／32。初始必須是實際 Backend 已 Drain 的同條件狀態，
不是只看 Gateway Count=0；Arm 順序在 Split 內隨機化。

每 Episode90s 到達＋最多20s Drain。Train 的 Capacity Scout 先取得 λ*，
Train／Calibration／Validation 使用0.6→1.1→0.6λ* 的低高低負載，
辨識飽和與恢復；Test 使用新的 Burst 時間／1.3λ*，同時包含較長 Prompt。
這是聯合 OOD Stress，不足以分別歸因兩種 Shift；若失敗須另做單因子實驗。
λ* 只用 Train Scout 決定，不從 Test 反調 Offered Load。

Train24、Calibration8、Validation8、Test8，整個 Episode／Pair 分離。
必須依序完成 Fit→Calibration→Validation→Freeze，再解封 Test。
Template：`results/mpc-research-cpu/native-id-design-manifest.json`。
它尚未填真實 λ*／Payload／Host Identity，也沒有執行權。
48場是 Pilot，不足以驗證0.1% Tail Risk 或 Native 性能優勢。

此設計用因子對照、互補輸入、不同 Dwell 和共用到達 Tape 取代巨大 Full Grid，
是資訊效率導向的選擇，**不是已證全域最優的 Experimental Design**。

### 6.2 必收狀態與事件

| 層 | 必收資料 | 用途／不可誤稱 |
|---|---|---|
| Offered／Client | ID、排程／送出、HTTP Status、First/Last Content、終止／截尾、DONE、錯誤、Usage | 全 Offered Goodput；Late Failure 不能算成功 |
| Gateway | Run/Boot/Time Namespace、Sample Time、R、Budget/Version、Request Marks、Received/Admitted/First/Released、拒絕 | 可觀測 Reservation／Residence；不是 Backend Membership |
| Control | 事前分派、Operation ID、Send/Ack/Applied At、CAS、Expiry、Pending／Fallback | 以實際套用輸入建模，另報 ITT／第一階段執行率 |
| Engine | 對應 Request ID、Enqueue/First/Terminal、Running/Waiting、Executed Prefill/Decode Tokens（若有） | 名稱與單位必須用實際0.30 Schema 驗證；缺漏標 unavailable |
| GPU | KV 使用量、Memory、Utilization、Clock/Power/Temperature、Throttle，帶 Sample Time | 低頻輔助觀測，不當成每次 Request 的精確 GPU Cost |
| Host | Timing 分段、CPU Quota/Throttle、Lag、記憶體／GC、錯誤 | 區分模型、觀測與控制器開銷 |

先以 Gateway 的 R／Phase Residence 建最小模型；Engine 感測不可得時，
可以研究端到端 Admission 因果效應，但不能聲稱已辨識 Prefill/Decode 物理排程。
KV 只有在感測新鮮且 Ablation 有增益時才加入；不能補零當已觀測。

模型先嘗試守恆 `R_next=R+admitted_cost-released_cost`，
再擬合 Phase Hazard／Residence 與 Service Cross-effects。
只用 Action 前 State 估計動態；ITT 不篩掉未送達 Action 或拒絕 Requests。
EOS 用低載全接納樣本和右截尾分析，避免把 Admission 選中的 Outputs
直接當全體分布。自然 EOS 是否隨 Batching／Budget 改變也是待驗假設。

### 6.3 真實 HTTP 與資格 Gate

- 使用單 Gateway Worker、單 ID Writer；Load Generator、Observer、
  Planner 分開 Process，避免求解卡住排程到達。預先建立 Payload／連線。
- 同一 Monotonic Domain 才比較時間；不同 Domain 先拒絕，不作跨 Boot 減法。
  Client 和 Gateway 儘量同 Pod／Time Namespace；GPU Clock 不直接減 Host Clock。
- 驗證 Fragmented SSE、DONE、Error、Late Failure、Reject、Disconnect、
  Timeout、Duplicate、Orphan Backend Work。SSE Chunk 不等於一 Token。
  只有合格 Token Count／Timestamp 才報 TPOT；否則只報 Chunk Gaps 和明確標示的
  Client-delivery Mean TPOT，不冒稱 Token-gap P99。
- 新 HTTP Load Fidelity 預先要求 Posting Lag P99≤10ms、P99.9≤25ms，
  對應0.5s SLO 的2%／5%；端到端 Scheduled TTFT 仍包含全部 Lag。
  這是新 Gateway 實驗合約，**不是放寬舊 Envelope 的5ms Gate**。
  若重用舊 Collector，舊 Gate 完整保留，不重評舊失敗。
- Raw Evidence 必須100% Offered IDs 有結局或顯式 Censoring，
  無重複、漏 Join、跨時域比較、未解 Pending 就發新命令。
  無法完成／缺 Sensor／欠資格時判 Inconclusive 或 Failed，不刪該 Episode。
- Observer Overhead 用固定 Workload 的成對 OFF/ON Qualification 先量測；
  5% Upper95 Bound 不通過或樣本不足，就不能聲稱低擾動驗收。
- Native Guarded Write 必須實測 Delayed Ack、Timeout、CAS Conflict、
  過期未套用、Restart／Boot 變動；Fresh GET 本身不是 Fence。
  Skip No-op POST 但保留 Assignment Log：Template816次分派，
  約454次含初始／Drain 的獨立 Write，再為 Qualification 預留，
  在1024 Outcome Cache 上限前停止，不能靜默清空冪等性紀錄。

安全停止：任何 OOM／不可恢復錯誤、未解 Pending、跨時域不一致、
未達資格的感測、Backend 未在 Drain 上限內清空，停止新負載並保存資料。
KV≥90%持續三個合格 Sample 作保守停止觸發，不聲稱這是 OOM 機率界。
費用／時間即將超額也停止；不能為湊48場延租。

### 6.4 辨識與模型驗收，不與政策優勢混淆

必報1／3／5s Rollout Error、First/Terminal Residence、Action Contrast、
Brier／Risk Reliability、Residual Coverage、OOD Error、State Ablation，
與 Persistence／簡單 Phase／Prompt-Backlog Baseline 比較。
Reject／Failure／Unresolved 保留於所有 Offered 指標。

預先把「Validation 主要 Forecast MAE 至少比最佳同資訊簡單 Baseline
低10%、且 Risk Brier 不惡化」當實用候選 Gate，而非數學定理；
達不到就保留簡單模型。樣本太少／Interval 太寬就報未決，
不聲稱等效或條件 Coverage。不能用 Test 修係數或換成功定義。
完整 Observe→Generate→Solve→Applied 路徑另以100ms／99%完成率作
下一階段目標，同時保留原 Solver100ms 判定；不把外部開銷藏起來。

Pilot 不比較新調 MPC 是否「贏」。只有模型具可驗證 Action Ranking、
State 支持與真 HTTP 資格後，才另凍結 Fixed／Feedback／One-move／MPC
的公平 Confirmatory Benchmark，另行估算 Replicates 與付費預算。

## 7. 成本、授權與限制

最多單卡2.5計費小時、**申請全包上限 US$5，尚未獲授權**。
先前查閱的 Runpod 公開 RTX4090 Pod 價為US$0.89/h，
2.5h GPU-only 為US$2.225，僅規劃參考，不是現有機器報價或可用性保證。
來源／限制：`results/mpc-research-cpu/gpu-budget-reference.json`。
最低儲值、最低計費、稅、Disk、Network 全部須納入核准總額。

| 預算項 | 分鐘上限 |
|---|---:|
| Setup／模型與版本核對 | 20 |
| HTTP／State／Action Qualification、Train Scout | 15 |
| 48×(90s+20s) | 88 |
| 證據匯出 | 10 |
| Fit／分析／故障預備 | 17 |
| 合計 | 150 |

模型下載／版本不符／Qualification 失敗可能用完預備時間。
先預檢既有 Image／Cache；無法容納就少做場次並報缺口，不保證完成48場，
更不保證2.5h 內取得最終性能勝負結論。未經明確授權不租、不續租、
不假設之前租機仍可用。

## 8. 裁定：確認、推論、假設

**確認：** 凍結 CPU 的 MPC 負面結果；10次超時只在 burst；
其他兩類無超時仍輸；預測／Truth Service Law 不同；
多步常提高模型分數卻不改首步；現有 Raw Bridge Mode 與 MPC Mode 衝突；
HTTP Collector 尚缺部分 Native Label；本次226項準備／守護回歸通過。

**有依據但未隔離的推論：** State Aliasing、錯誤 Residence／Interference、
非平穩 Forecast 與 Replanning 可能共同使多步的模型內收益落空。
計算工作量是 Deadline 的主要候選機制，Host 排程／Throttle 的額外比例未知。

**待 Native 驗證：** Budget 的因果效果、實際可操控延遲、
EOS／Remaining Work、KV 增量價值、OOD 校準、HTTP 命令資格，
以及在那些條件下 MPC 是否值得超過簡單控制器的成本。

下一步不是盲調參或擴大 Solver，而是先解決介面／Label 資格，
再以受授權、受預算限制的激發實驗取得可辨識資料。
