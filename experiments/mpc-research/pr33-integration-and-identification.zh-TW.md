# PR33：整合、辨識資格與下一階段裁定

## 1. 已完成與未完成的事項

PR32 已以 merge commit `765166081d0af5499c63ecff53b257f91261d3de`
合併至它原本的目標分支 `analysis/cap-miss-causes-v1`，**不是 main**。
沒有 squash、rebase、force push 或重寫既有提交。

新 Draft PR：https://github.com/dlinlabs/vllm-serving-lab/pull/33

新工作分支：`research/pr32-native-id-integration`。
整合提交 `b9015b7944990fbab160846ad996219294ae1481` 保留既有研究提交
`de36581`、`3234459`、`ac7c452` 及四模型三輪原始報告。
本輪由 Mayor 本機完成，未重啟 Polecat、未呼叫 `gt done`。
只發布使用者此次授權的新 PR 分支；這不追認先前已揭露的違規 Push。

**尚未證明 MPC 勝出或實用等效。** 本輪不改 MPC 的服務係數、成本權重、
控制 horizon、100ms 求解期限，也不反覆使用舊 Test seeds 調參。
沒有執行 Native GPU 或真實模型 HTTP 實驗，沒有啟動付費資源。

另一條 pending-v2 研究尚無可核對的確切提交或原始資料。
本機 pr32 Mayor 工作樹及分支未找到該實作，不能宣稱已整合、重現其成果，
也不能把它的 pending 問題接成舊合成 benchmark 落差的已確認原因。

## 2. 對負面結果的獨立裁定

以下是既有凍結合成實驗，不是新版本 GPU 結果：

| 策略 | 平均 successful goodput（req/s） |
|---|---:|
| Fixed16 | 5.5778 |
| Fixed32 | 2.3833 |
| Feedback | 3.8722 |
| One-move | 4.5444 |
| MPC | 4.4222 |

來源：`results/mpc-research-cpu/policy-benchmark.json`、
`experiments/mpc-research/benchmark-results.md`，研究提交 `3234459`。
這裡的 16/32 是 **reservation cost budget**，不是同時執行 16/32 個請求；
也不是 vLLM `max_num_seqs` 或每輪 prefill token budget。

### 已確認

- MPC 較 Fixed16 約低 20.7%；適用於該合成 serving law、工作負載及實作版本。
- stationary、shift 沒有 deadline fallback 仍然較差，所以超時不是唯一原因。
- 凍結 Test 中 10/180 次 deadline fallback 全在 burst。
- 六組 Development-only 已知未來/EOS 診斷中，模型與合成 Truth
  都把 Fixed16 排在 Fixed32 前；不能直接指控固定 cap 排序必然顛倒。
- 九個事前選定 checkpoint 的七個多步 open-loop 分數較高，但只有兩個
  首步動作不同；共享前綴的 workload 不能充當獨立樣本。
- One-move 與 MPC 的差異區間包含零，不代表已通過等效性檢定。

### 有根據的推論，但尚未完成因果消融

最大的嫌疑是 **在線狀態重建與未來 workload 假設共同造成的閉環偏差**：
舊 decoder 進度以簡化的 elapsed-age/服務常數估算，
沒有重建歷史 decode interference；未開始輸出的 prefill 工作也會重新估算。
未來 EOS、成功與 transport 假設又與合成 Truth 不同。
這些偏差可能改變「目前提高 budget 後，成本何時釋放」的判斷。
偏差方向不一定一致，不能只修改某個係數便宣稱找到 root cause。

多步規劃沒有優勢的合理機制是：多數狀態第一步相同、重新規劃不執行
完整舊計畫，以及多步 rollout 放大狀態與 workload 誤差。
目前沒有使用可信任的原生 trace 分別消融這三種機制。

Deadline 的直接瓶頸是大量 lifecycle event rollout。
既有 Development 診斷中，rollout 約占 95.2% 決策時間；
一個 burst 反例在約 100.081ms 時只完成 7 個完整計畫，
完成事件數 28,651，尚未完成整個 32-rollout 搜尋。
這是 cooperative deadline，不是 OS realtime 保證。
逾時捨棄部分搜尋並 fallback16 的規則沒有被放寬。

舊合成 plant 不會隨 solver wall time 前進，也沒有完整支付 GET/POST 延遲。
因此它仍不足以代表部署中的控制成本或證明跨 GPU／跨模型勝負。

## 3. 本輪的實作，而非另一套 CPU MPC

### 標準 Chat 與完整結果標記

`mpc_lab/http_capture.py` 使用標準 `/v1/chat/completions`，
沿用 `telemetry.has_sse_content` 驗證 JSON content。
新增嚴格 UTF-8、SSE 分幀、第一／最後 content event 時間、finish reason、
usage、DONE、request-ID、timeout/cancellation 記錄。
取消仍透過 recorder 保存結果並傳遞 cancellation；
完整 offered manifest 缺任何請求都不能作為有效實驗。

Gateway admission 503 與 upstream 503 分開判定，避免把服務失敗算成拒絕。
SSE chunk/event 數 **不當成 token 數**；usage 不構成 token timestamp 證據，
`token_timestamps_qualified` 保持 false。
因此這仍不能驗收 native TPOT P95/P99。
Capture 預設關閉，沒有 GPU 啟動器或自動執行 CLI。

### 精確成本佔用 trace

Gateway 在 MPC 與 telemetry 都啟用時，於既有 state lock 內記錄
`mpc_ledger`，含 clock/run/model/cost/source identity、連續 sequence、
reservation cost、budget/version 及完整性。
Admission、first content、release、budget mutation 均留下同一順序的 frame。
其 scalar 量測不等於 KV bytes 或 GPU 工作秒數。

`mpc_lab.identification.reserved_cost_at` 要求初始完整 snapshot、
連續 ledger、最終 snapshot watermark 和零 dropped/write-error。
藉此重建精確 `R(t+1)`，**不將較晚的 polling snapshot 冒充 1 秒真值**。
Trace 開銷仍須在 Native 實驗中量測，各比較 arm 必須使用相同 instrumentation。

### 辨識專用權限，不偽造 calibration

`IdentificationPermit` 僅接受：

- 明確批准、相同 run/clock/identity 及有限到期時間。
- `native-id-unfitted-v1` / `unfitted-not-validated` 身分。
- `identification_assignment`、沒有 MPC plan 的預定 16/32 成本 budget。
- 500ms 內完整 state、原有 CAS、100ms TTL、到期與 durable pending fencing。

不填假的 calibration hash，也不授權正常 MPC 決策。
GET 的開始／回應／sample 時間與實際 command/ack 被保留。
Native GPU 支出許可與 transport permit 是不同事情；
目前只有 Mock/ASGI 測試中的人工許可，沒有真實資源許可。

### Drain 與故障資格

Gateway ledger 清空不代表引擎已清空。
`drain_evidence` 同時要求可信任、同 clock、fresh 的引擎
running/waiting/orphaned 訊號、正確兩端 run identity 和無未知 pending write。
未知／過期／跨 boot 量測一律不宣告 drained。
一般 Prometheus GET 回應時間不能冒充引擎實際採樣時間；
原生 sensor adapter 的時效與 orphan 證據仍須實機驗收。

## 4. 凍結統計及因果定義

新協定生成器：
`experiments/mpc-research/native_identification_protocol.py`。
所有 JSON 均標記 approved=false、execution_enabled=false。
生成 action/split 規格，不生成或檢視 Held-out arrival tape。

**保持 48 episode，但改為 24 個獨立 arrival/prompt seed blocks，各兩個互補 arm。**
每個 block 只屬於一個 workload cell 和 split：
Train 12 blocks、Calibration 4、Validation 4、Test 4。
各 cell 為 prompt128/4096 × max-output128/256；自然 EOS。
各 split 的 cell 平衡；同一 block 兩 arm 的 offered tape 相同，
但每個 arm/offer 使用唯一實體 32-hex request-ID，保留共同 logical offer ID。

這修正舊 48 場方案中 Validation/Test 各僅一個共同 arrival seed 的問題。
**四個 Validation/Test blocks 仍是 pilot，不足以宣稱穩定泛化或條件 tail coverage。**
Calibration 只有四個獨立 blocks，不能聲稱已有有限、可靠的 90% block-level
conformal bound；沒有資格把它包裝成 stochastic MPC 的統計保證。

主要 forecast 指標固定為 `R(t+1)` 的 MAE / 32。
每個 episode 平均決策，再平均配對 arms，再平均獨立 seed blocks。
1 秒以外的 horizon、first/terminal residence 與 good count 都是 secondary；
不能事後挑最有利的 horizon。每個 split 分開計算。
Missing decision、重複資料、未完成真值或跨 split seed 一律拒絕主要分數。

Baseline 僅以完整的 Train leave-one-arrival-seed-out folds 選定，
凍結 selection hash、模型與資料版本後才讀 Validation。
Selection hash 只封存選擇結果，不是模型或資料資格證明；
各 fold 的 data/source/cost/parameter artifacts 須另行封存與驗證。
目前沒有 Native 訓練資料，也沒有已選定、已驗證的 Native baseline model。
10% Validation MAE 改善只作探索性篩選，不作統計通過證明。

Brier 事件是「完整成功 HTTP 且 scheduled TTFT≤0.5s」，
預測在 POST 前產生，不使用真實 future arrival/EOS。
分母涵蓋所有 offered requests，拒絕與錯誤不消失。
及時 first content、但尚無完整終止的右截尾案例保留兩種 label 的 loss bounds。
Brier 是 secondary，不能用拒絕所有請求的好分數取代 successful goodput。

互補 arm 比較 **整段指派序列的 ITT** 與實際 delivery first stage。
第一次動作後狀態分岔；不得把後續點對點相減稱為 same-state C16/C32 因果效果。
局部 action ranking 另需足夠 pre-action state support、指派機率、
實際套用 lag 與可靠 outcome；不支援的狀態保留未決。

## 5. GPU 設計、成本與停止條件

單一部署限定 RTX4090、同模型與完整軟體／tokenizer／scheduler 身分。
不是同型不同主機或不同 GPU 的移植性研究。

先以 Train-only、每 cell 的 Fixed16 scout 選 rate reference：
0.5/1/2/4/8/16 req/s，短 offered window 加 drain；
以 successful goodput 最大的最低 offered rate 定錨，全零則停止。
之後凍結參考值，採 0.6→1.1→0.6 倍的負載與 2/5/10 秒 action dwell。
未來 disturbance-shift／混合 workload 的確認需另立未使用的資料。

必收：完整 offered schedule/prompt/max-output、Gateway admission/cost/release，
引擎 running/waiting/KV/remaining-work 可觀測代理、真實 token/terminal 事件，
action 指派與 CAS ack、GET/POST/solver 時間、
GPU clocks/power/temperature/memory/utilization，以及 trace loss/censoring。

48×(90s offers+20s drain)=5,280s。
另保留 setup/scout/export/contingency 3,720s，總上限提案為 9,000s（2.5 小時）。
**預算提案 US$5 all-in，尚未授權；不是實際花費或價格保證。**
價格、最低充值、storage、下載、啟停計費須在批准前確認。
資訊不足、trace 不完整、drain 失敗、未知寫入、無有效 action variation、
或到達預算上限時停止，保留全部資料；不承諾硬做完 48 場。

目前缺有效 Native sensor/HTTP qualification 與批准的付費實驗時段。
先取得可辨識資料，不拿這一輪 pilot 證明 MPC 贏。

## 6. 後續 iteration 的成功定義

先從 Train 資料辨識工作成本釋放與 prefill/decode interference，
在凍結 Validation 檢查多步 rollout、OOD 與風險校準。
有可靠 action/state support 後才做 Fixed16、Train-tuned best fixed、
Feedback、One-move、MPC 的同硬體同 offered tape 比較。

獨立 confirmation 的樣本數由 Development/pilot 配對變異、
事前 power 和改善／等效 margin 決定，Test 解盲前一次凍結。
不能持續測同一 Test 直到「贏」；調參後要用新的確認資料。

Superiority 要 successful-goodput 配對、處理多重比較的下界為正。
實用等效的提案是相對 goodput ±2% 的 TOST 90% CI，
另加事前凍結的拒絕、SLO、安全與 overhead 界。
**±2% 尚不是已驗收的容忍度；其他界與 sample size 尚待 pilot 支持。**
CI 包含零、純 fallback16、budget 從不 binding、只換名的 Fixed16，
以及忽略 solver/HTTP wall time，均不算 MPC 的成功證據。

若充分辨識與公平確認後仍無優勢，保留負面結果並採簡單控制器；
不以跨模型共識替代實驗，也不承諾任意 GPU／模型皆能勝出。

## 7. 本輪驗證索引

完整 CPU 證據：`results/mpc-native-integration-cpu/qualification-junit.xml`
與 `qualification-pytest.log`：2014 passed、1 native skip、1 既有 warning，
265.14s。它們不代表 Native GPU 或真實模型 HTTP。
這次完整回歸之後，獨立 wheel 匯入發現缺少 `telemetry`；
已加入封裝清單，另外檢查修正後的 wheel 匯入、內容與相關 source manifest。
精確後續數量與來源檢查記錄於同目錄的最終 verification manifest。
封裝修正後 source-manifest 子集 30 passed；獨立 wheel 匯入成功，
43 個 Python source files 與工作樹逐位元符合。
Wheel 匯入不是 Native 主／客端 identity 資格；
實驗兩端仍須核對實際載入的完整部署來源與同一 implementation hash。

MockTransport/ASGI 覆蓋標準 Chat 到 Gateway admission/first/release ledger，
UTF-8 分片、錯誤／缺 DONE、取消、timeout、request-ID、503 分類、
跨 clock／stale/orphan drain、辨識權限範圍、trace gap/loss、
cohort/seed leakage、右截尾 Brier 與 Train-only baseline freeze。

原凍結結果、失敗紀錄、source identities 與舊 protocol 保持原樣。
Current pending-native prefill design 只刷新本輪直接改動的 source hash，
不改歷史 qualification、5ms posting scope 或驗收標準。
