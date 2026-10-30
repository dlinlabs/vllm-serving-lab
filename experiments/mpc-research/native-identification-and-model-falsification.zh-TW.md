# PR33：原生系統辨識與候選模型反證

## 裁定

**不是「MPC 已經沒有勝率」，也不是「Native MPC 已經勝過 Fixed16」。**
本輪已完成真實 RTX 4090／Qwen3-4B 的兩批動作介入與獨立低併發辨識，
並找出一項必須先處理的物理模型錯誤：
**不能把 HTTP first-residence 直接當成獨占、逐請求串行的 GPU prefill service。**
這個解讀與已量到的吞吐量存在可重現的數學矛盾。
沒有停在反證：接著完成 CUPTI 資格與 built-in profiler 原生實驗，
取得真實 GPU kernels 與 batch annotations，確認 CPU range／GPU span／kernel busy
必須分開；但短 profiling smoke 不等於無 profiler 的批次模型 calibration。
在矛盾解決前啟用該模型的 MPC，只會把錯誤預測接上真實控制；
因此本輪沒有偽造 calibration attestation、繞過 permit 或執行 Native MPC 驗收。

原產品／CPU MPC 與舊負面結果未改。本輪新增研究程式、測試、真實原生證據；
沒有 Push、Merge、修改 Git remote、PR 留言、`gt done` 或重啟舊 Gas Town workers。
工作分支仍為 `research/pr32-native-id-integration`。
所有結論僅限本部署，跨模型／跨 GPU 的勝負沒有被證明。

## 1. 已執行的原生實驗

沿用先前 qualification 的 RTX 4090、Qwen/Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`、BF16、vLLM 0.30.0、
Torch 2.13.0+cu130、正常 graphs／async、prefix cache off、
max sequences32／batch tokens2048。沒有換成 eager 或放寬 deadline。
16／32仍是 reservation-cost budget，不是請求數。

本輪啟用的是已修 default-off 的 observer：
`29fc1ebd875ccc1a53b9584fd0f3b002c518e3653c9b4238877b67b7ec13802e`；
它已實際用於以下 GPU 實驗，不再只是 CPU 測試的版本。
產品 implementation hash 保持：
`b4f234020105f69f608da45dba7317a8bb632ecd84aeb0e310362740e5fa083a`。

| 實驗 | 獨立 Train seed blocks | 設計 | Offers | Natural EOS complete | Rejected | 其他失敗 |
|---|---:|---|---:|---:|---:|---:|
| 第一批介入 | 160001–160004，4 blocks | 4 cells × 2互補16／32序列；每 arm90秒 | 8740 | 6487 | 2253 | 0 |
| 較低負載介入 | 160005–160008，4 blocks | 相同配對設計；較低參考率；每 arm90秒 | 5052 | 4589 | 463 | 0 |
| 單請求辨識 | 160009–160010，2 blocks | 4 cells，每格1筆 prelude＋3筆 fit；逐筆前後 drain | 32 | 32 | 0 | 0 |

四格為 prompt128／4096 × max output128／256；自然 EOS、temperature0，
prompt template 後的 native input tokens 與名義值完全相等。
介入的每個 arm 都包含30秒低負載、30秒較高負載、30秒 recovery。
前一批參考率取自 C32 scout，後一批取自 C16 scout；這些是粗略 cohort
capacity estimates，不是已確認的穩態容量。
每對 arm 共用 offered tape 與 prompt recipe，使用原 protocol 的互補序列，
記錄實際 applied acknowledgement，而不是假定排定時間已生效。

兩批共13792 offers＝11076 complete＋2716 rejection。每筆都保留，
包含所有拒絕、暖機／prelude 與被排除於特定模型 fitting 的資料。
每 arm 開始／結束有兩次 drained evidence，ledger 連續、無 dropped/write errors。
第一批／低負載批最大 applied lag 分別78.76ms／52.03ms，無 assignment retry；
不是把控制延遲當作 solver deadline，也沒有放寬100ms command TTL。

**不能把16個 arm 當16個獨立 workload seeds。**
兩批總共只有8個介入 seed blocks；配對序列第一次採用不同動作後，狀態就分岔。
可分析整段指派序列效果，不能逐時間點相減宣稱 same-state action causal effect。
這也不是 Fixed16／Fixed32／Feedback／One-move／MPC 的正式公平比較。

### HTTP 指標與觀測資格

本研究先固定診斷定義：TTFT 從 scheduled arrival 起算≤500ms，
request-mean TPOT＝(terminal−first content)/(output tokens−1)≤50ms；
單 token 的 mean TPOT 定義為0。拒絕在 all-offered good count 中得0。
這是研究用複合 SLO，不是客戶 SLO，也不是逐 token TPOT P99 資格。

| 批次 | TTFT-only good | 複合 good | Native samples |
|---|---:|---:|---:|
| 第一批 | 6366 | 6364 | 10323 |
| 低負載批 | 4522 | 4521 | 10367 |

這些是帶 drain 的 cohort counts，不是可直接相除而宣稱穩態吞吐量的數字。
`audit_interventions.py` 從完整 rows 重算，拒絕 summary 與 raw 不符、
錯 clock、缺失初始 assignment、epoch 改變或 incomplete outcome。

第一批只有離線 ledger effective timestamps，不能證明 controller 當時已收到它們。
因此低負載批每次 Native sample 前另外讀 Gateway state，保留兩個 client receipt
timestamps。10367筆均通過 source／clock／age≤0.5s／width≤0.05s 的 join；
其中8502筆有完整 physical work join，1865筆仍有 Gateway 已接受但 Native
尚未出現的 work，**不能填0或當作完整 MPC state**。
這些是從實際 capture 重放驗證，不是當時已發出 online forecast 的證據。

NVML monitor 在低負載批後段才加入：只有最後兩個 arm 分別19／90筆 window
內樣本，不能稱全批 clock／power／temperature 已受控。完整254筆 timeline 與
實際範圍保留；單請求實驗在它啟動後執行。

## 2. 預測模型：哪些有效、哪些沒有勝過簡單基線

### Residence 模型

Opus 獨立實作6維非負模型，以 leave-one-seed-block-out 留出整對 arm，
預測尚未 first-content／terminal 的 admitted requests 剩餘 residence。
Scout＋第一批介入共11712 offers；732個很快 first-content 的 requests
未進 first-residence target，數量有明確記錄，沒有將它們當可預測成功。

兩個 target 各只有4／8 folds 優於最佳 baseline；短 prompt terminal 的 persistence
通常較好。最差 fitted terminal bias 約−250ms，不是−25ms。
這是 retrospective Train 診斷；有效時間戳不等於線上 consumption availability，
也沒有證明因果 action model、Native calibration 或 MPC 效能。

### Native decode token-progress 模型

以實際相鄰 Native snapshots 中仍在 decode 的 requests，辨識：

`E[Δoutput] = 1000 Δt / (θ0 + θD D + θP Pk)`

其中 D 為正在 decode 的數量，Pk 為 observed prefill prompts 的千 tokens。
得到 θ0≈12.31ms、θD≈0.049ms/request、θP≈8.73ms/kilo-prefill-token。
**θP 是共存 prefill 對 decode 的 slowdown，不是 prefill duration coefficient。**

資料為11719 snapshot pairs、79046 survivor intervals。Grouped held-out
平均 token MAE：constant1.674、persistence1.029、候選1.043；
候選對 best-baseline MAE 只贏1／8 folds，不能宣稱預測已全面勝出。
候選對 constant 的 Poisson deviance 則改善6／8 folds。

已知限制：實際有7661個消失 request
作 censoring 記錄、4523個 prefill intervals 不在此 target、15個末端仍 active。
觀測 support 為 D1–31、Pk0–8.192k；超出應 abstain。
本資料沒有 zero decode increments，可能受 utility RPC sampling 節奏影響；
不能據此斷言真實系統沒有 stall。Δt 是事後已實現的 snapshot gap，
所以這不是已通過固定未來 horizon 的線上預測；描述性 coverage 也不是統計保證。

## 3. 關鍵反證：HTTP 延遲不能直接當串行 GPU service

先前11712 offers 只找到52筆符合 isolation 規則的 first-residence，全部是長 prompt；
只有一個 prompt level，截距與斜率不可識別。程式現在明確輸出 parameters=null，
而不是隨意補一個 slope。

新增低負載與單請求資料後，在16796個 offered records 中：
4499未 admitted-complete、220個預先定義 warmup／prelude、
11906個 shared reservation、3個 observed shared native 被分別記帳；
168筆符合 observed-isolated first-residence fitting。
用 seed-balanced nonnegative regression 得到 HTTP proxy：

`first_residence ≈ 0.03796779 + 0.00005423180 × physical_prompt_tokens 秒`。

它包含 backend send 到 client first content 的 transport／frontend／first-token
開銷，不能宣稱純 GPU prefill。即使用它做 first-residence prediction 尚可，
也不能自動移植成控制模型的串行 service time。

若把它完整放入目前候選的 exclusive FIFO prefill law，p128 每筆需
44.90946ms。從空系統開始，90秒最多完成：

`floor(90 / 0.04490946) = 2004` 次 prefill，

EOS completion 不可能更多，decode 只會進一步縮小上限。
但已量到：

| 原生 episode | 90秒內 natural EOS completions | 該解讀的上限 |
|---|---:|---:|
| train-160001-1 | 2449 | 2004 |
| train-160001-0 | 2056 | 2004 |

2449筆對應109.9833秒串行 work，缺少19.9833秒；要用 boundary 解釋，
至少需445筆已完成 prefill 的 carry-in，與 initial drain 矛盾。
這裡只算90秒內 terminal 的2449／2056，不使用後續 drain 的2459／2066。
`serial_capacity_screen()` 自動重算並拒絕該解讀。

Astra（候選作者）獨立核對上述推導，明確同意條件式反證：
它否定的是**這些 HTTP proxy 係數的獨占串行解讀**，不是所有串行模型、
所有 prefill 模型，更不是 MPC 本身。
模型回覆只是審查依據；真正反證來自空初態、固定 window 的原生 counts 和容量不等式。

不能把37.97ms直接扣掉、把 slope 調小到能贏，或將 batch 效果塞進任意係數，
再拿同一份 Train 結果宣稱解決。需要另外識別 transport 與批次 GPU service。

### 接續實驗：不是只提出 profiling 計畫

先在原 Pod 實際跑 CUDA matmul profiler smoke，取得1筆真實 CUPTI kernel event，
而非只有 CPU annotations；同步只在這個獨立 smoke 中使用，沒有插入正常 serving。
然後啟動相同模型／graphs／async 的獨立 loopback vLLM diagnostic，
使用原生 `--profiler-config`，關閉 stack／shape capture、忽略 frontend profiling。
這一輪刻意啟用 profiler，**不與 unprofiled benchmark 混為同一環境**。

兩筆 prelude 後，prompt128／4096 × offered microburst1／4，各重複兩次；
共22筆標準 Chat HTTP complete，全部是預先指定 max8 的 `finish_reason=length`。
它們是測量資格探針，不是上表的 natural-EOS 辨識資料，也不加入其 goodput 或 EOS fitting。
觀測每組前後兩次 native drain，保留 HTTP IDs／usage／state／collector source。
第一次 collector launch 因 log directory 尚未建立而在 HTTP 開始前失敗；
建立目錄後才執行，沒有覆寫既有 outcome 或修改測量條件。

實際取得37112個 GPU kernel events、84個帶 detailed token counters 的 GPU batch
annotations。`analyze_profile.py` 以 External id 對應 CPU/GPU annotations，
拒絕重疊的 ambiguous batch spans，並以 device-stream containment 歸屬36272個
kernels；另外840個保留為 batch annotation 外工作，不任意塞進 prefill。

| 實際 batch 特徵 | Batches | GPU span 中位數 | Kernel busy union 中位數 | CPU annotation 中位數 |
|---|---:|---:|---:|---:|
| Prefill1，query128，decode0 | 4 | 67.66ms | 10.36ms | 68.39ms |
| Prefill1，query2048，decode0 | 8 | 115.22ms | 112.46ms | 57.29ms |
| Prefill3，query384，decode1 | 2 | 48.53ms | 24.31ms | 48.90ms |
| Prefill0，decode1 | 32 | 9.42ms | 9.38ms | 2.77ms |

這些數字顯示 CPU return、GPU interval span、GPU active kernels 不是同一個量。
GPU span 含 gaps；kernel busy 也不是可直接相加的 exclusive per-request demand。
不能將不同 run 的45ms HTTP proxy與這裡的10ms相減，就宣稱已辨識35ms transport。
Microburst4也不等於真正 prefill batch4：實測可能先排1筆，再將3筆與decode混排。

**仍缺：**這份 trace 沒有資格化 request-to-batch identity／monotonic trace conversion，
只有兩次重複、decode background最多4、且 profiler 會擾動 timing；
沒有證明 unprofiled 批次 service law、因果 decomposition、完整 action support 或 calibration。
因此沒有拿它直接替換 MPC 係數。這是新的實測進展，不是性能勝出。

## 4. 已實作但未部署的 Native MPC challenger

`native_mpc.py` 是純函式研究候選，不持有 actuator 權限：
state adapter 將 Gateway charged cost 與 Native physical tokens 分開，
綁定 model／cost／observer／training-source identity；
使用同一組情境比較8個3×1秒16／32 plans，再以 Fixed16 continuation 與完整 drain
計分，100ms wall-clock deadline，失敗回16。

目標是 all-offered 的複合 good completions；拒絕不得分，
未來 prefill 造成既有 decode 的 TPOT 惡化也影響評分。
未知 physical work、EOS survival support、超出物理 support、未完成 drain、
stale／clock mismatch 都不回傳可部署的成功推薦。
synthetic tests 有 workload-dependent binding action，但不是 Native 性能證明。

目前尚缺可接受的批次 prefill 模型、fixed-horizon prediction qualification、
獨立 Calibration、完整 runtime／normal actuation permit 及公平 HTTP comparison。
因此 native_calibration_accepted、performance_accepted、actuation_authority
均維持 false；沒有用 IdentificationPermit 冒充 adaptive MPC 授權。

## 5. 下一個有資訊價值的實驗，不是盲調 MPC

### 最優先：可對應批次的真實執行量測

需要將 request ID、batch ID、實際 scheduled prompt／decode tokens 與真實
GPU execution interval 對應，另記 backend receive、first-token-send、client receipt。
本輪已檢查 pinned `GPUModelRunner.execute_model`：真正 model forward 與稍後
`sample_tokens`／async copy 分開，CPU function return 不代表完成。
CUDA event／profiler placement 仍須對 async／graph streams 完整資格化；
不能用 CPU API return、scheduler computed_tokens 或 blocking synchronize
製造「看似精確」但改變原系統的 service time。
mixed batch 的總 GPU duration 也不能不經辨識就全部算給 prefill。

已有 CPU trace-parser tests 與上述真實 CUPTI smoke／batch trace；
下一步補 opt-in／default-off 的 request/batch join、event completion、
observer overhead、drain 與故障資格，建立無 profiler 的低擾動量測。
若無法可靠區分 transport 與 service，應保留不可識別結論，而非補假資料。

### 辨識 pilot 提案，尚未執行

固定本機 GPU／模型／所有 runtime flags，開新 protocol version 和新 Train seeds，
不動原160021–160024 Test。以4個獨立 blocks 隨機化：
prompt128／4096 × prefill microburst1／4筆 × background decode0／8筆，
共32個 condition episodes；記實際 batch 組成，不把名義 burst size 當實際 batch。
先驗證 action succeeded／排程曝光、完整 all-offered outcomes、
硬體時間線與 transport／GPU work 是否可分離，再決定能否 fit。

預期20分鐘內的 bounded pilot（含啟動／drain／smoke，非完成保證）。
若 provider 單價為每小時P，20分鐘新增租用時間成本約P/3；
目前價格與帳單未知，不能捏造美元實支。這不是已發生的花費或自動續租授權。
若量測資格失敗就停止該 pilot，保留失敗，不透過升 timeout 或改 metric 繼續。

### 之後才做正式 controller 比較

模型在 Training／Calibration／Validation 過關後才封存 candidate 與一次性 Test：
Fixed16、Fixed32、簡單 workload-aware feedback、One-move、MPC，
相同 offered tapes、seeds、warmup、window、完整 drain 與硬體條件；
實際支付 GET／solver／persist／POST／fence wall time。

需要先以 pilot variance 做 paired-block power／precision 規劃，
不能保證任意4或8個 seeds 足以證明持平。
預先定義 goodput 的優越性／實用等效 margin 與 rejection／SLO guard；
不得看到 Test 後更改 margin、改 primary metric 或反覆調同一 Test。
若 MPC 和 One-move／feedback 相同，應承認多步規劃沒有新增價值。
全 fallback16、從不 binding 或固定32改名 MPC 均不算成功。

## 6. 交叉審查、執行紀錄與重現

| 執行者 | 實際模型 | 本輪工作與限制 |
|---|---|---|
| Mayor | GPT-6.1 Sol | 介入／單請求實驗、receipt adapter、audit、容量反證及裁定 |
| `8eb9618b-8392-4775-99c8-60768dbc9dad` | Claude Opus 5.5 | 獨立 residence／decode-progress 模型與 tests，未操作 GPU |
| `1884fd38-6581-4330-a752-c9f4266aa418` | GPT-6 Astra | retrospective predictor 審查，指出 input availability 限制 |
| `e434d55b-dc6d-46a9-9567-1f52593a4f6b` | Gemini 3.8 Flash | 較低負載／initial assignment 審查；無證據的 fsync／timeout 推測已撤回 |
| `145880b6-7115-4423-8f77-1bcd23aed6e8` | GPT-6 Astra | 純 Native MPC 候選，並核對容量反證；無 actuation／Native acceptance |

以上是 bounded Copilot research agents，模型由 runtime `read_agent` 記錄確認；
不是新啟動的 Gas Town Polecats，也沒有走 Polecat completion／`gt done`。
原始四模型三輪 Gas Town 報告與事故紀錄仍保留。
候選作者曾不必要地 broadcast 喚醒舊 Copilot reviewers，已明確制止；
那些回覆不作本輪 code verification 或額外研究成果。

本輪 focused regression：`python -m pytest -q experiments/mpc-research/native_observer`
實際50 passed。GPU／HTTP 的驗收依 raw artifacts 與 strict audits，
不是靠這50個 CPU tests 推論成功。

原始 scout archive 不變。本輪新增：
`results/mpc-native-4090-scout/native-identification-train-evidence.tar.gz`

SHA256：
`b86094f1e7d14343efd89741a21430fdd623c0f11381b2e674e4a2353501129f`

包含兩批完整 interventions、單請求實驗、Native frames、Gateway telemetry、
實際 collector source、logs、partial NVML、精確 PID 關閉證據。
實際 credential substring 掃描通過；credential config／SSH private key 不在 archive。
各 collector 舊版 `collector-measured.py` 保留，不以後續 source 冒充已量測版本。

後續 profiler 獨立封存為
`results/mpc-native-4090-scout/native-profiler-evidence.tar.gz`，
SHA256：
`13a2e00907c666a692faaee957eb6688990d62d16de45fb688a675612cbff9a2`。
包含 CUPTI qualification、原始 compressed trace、profiler table、22筆 HTTP、
啟動 configuration／source／logs 與最後 resource-stop；
credential 掃描也檢查了解壓後的 trace。

解壓至私有 evidence 目錄後，在 repo root 以既有 Python3.12 環境重現：

```bash
python -m pytest -q experiments/mpc-research/native_observer
PYTHONPATH=. python experiments/mpc-research/native_observer/audit_interventions.py \
  --directory /path/to/evidence/native-id-train --output /tmp/first-audit.json
PYTHONPATH=. python experiments/mpc-research/native_observer/audit_interventions.py \
  --directory /path/to/evidence/native-id-low-train --output /tmp/low-audit.json
PYTHONPATH=. python experiments/mpc-research/native_observer/audit_interventions.py \
  --isolated --directory /path/to/evidence/native-isolated-train \
  --output /tmp/isolated-audit.json
python experiments/mpc-research/native_observer/prefill_identification.py \
  --evidence results/mpc-native-4090-scout/native-evidence.tar.gz \
  --interventions-dir /path/to/evidence/native-id-train \
  --interventions-dir /path/to/evidence/native-id-low-train \
  --interventions-dir /path/to/evidence/native-isolated-train \
  --output /tmp/prefill-combined.json
python experiments/mpc-research/native_observer/analyze_profile.py \
  --directory /path/to/profiler-evidence --output /tmp/profile-analysis.json
```

所有自己啟動的服務均已按 PID 精確 SIGTERM：Gateway6444、vLLM6521、
monitor7377；8000／8080關閉、GPU compute process 為空。
後續 profiler vLLM8405也已精確 SIGTERM，再次確認8000關閉、GPU為空。
**沒有關閉 Pod，租金可能仍持續。**
最後結論是「原生資料與可否證模型已實質推進，但 MPC 性能目標仍未達成」，
不是宣稱持平、勝出或跨硬體無勝率。
