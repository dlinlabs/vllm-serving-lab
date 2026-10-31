# Native v3 第三輪：證據裁定與下一個可反證模型

**第三輪審查已重現，但原生 MPC 優勢與十小時實驗尚未驗收。**
本文件記錄 Mayor 的技術裁定，不以四模型投票代替實驗。
既有長時辨識仍使用凍結的 `6c11fe7`；沒有更改正在執行的 workload、
SLO、cap、split 或 outcome 定義。

## 來源與驗證

四份報告、程式與測試從以下本機提交逐檔 byte-verify 凍結。
模型身分另以各研究 session 的 backend usage records 核對，而非只看 profile。

| Worker | 實際 backend | R3 凍結提交 | 交付物 |
|---|---|---|---|
| amber | claude-opus-5.5 | `da31690bff0a63daaa7a296e277d80105580bb86` | 原生 marked-release 模型、Train adapter、數值與來源回歸 |
| jade | gpt-6-astra | `121234f8c99bcc1680d657b54e9eb7eb7b772911` | joint-response planner 介面、state consistency、late-input fencing |
| pearl | gemini-3.8-flash | `491c42b7f093d76732e91a052c844e194719e023` | 實驗方案、物理／duration／統計 helpers、交叉反例 |
| flint | gpt-6.1-sol | `475eb855dada59cb40d8c96d09368befcedadc92` | 完整 offered outcome、實際配置 binding、exact block inference |

Mayor 單一 runner 重現 **142 passed、107 subtests passed**。
首次使用 `--import-mode=importlib` 造成兩個 sibling-module collection errors；
改用既有預設 import convention 後通過，沒有修改測試或安裝依賴。
這是 CPU 實作／反例證據，**不是 native calibration、solver deadline、
HTTP MPC 閉環或統計效能優勢的通過證明**。
原始12個檔案與逐檔 source manifest 封存在
`results/mpc-native-v3/review-evidence/round3-frozen-source.tar.gz`；
archive SHA256 為
`8ce96cb02fbc55819904cba13c3ad283c60593af6874dae8c748ad1cba38f4b1`。
包含未通過 Mayor 科學驗收的原始 helpers，不會因封存而獲得 runtime authority。

## 已裁定的結果與仍未修復的缺陷

1. **物理工作分母已釐清。** 40 個 HTTP completions 中只有 20 個 LIGHT
   requests 有 physical labels，這 20 個皆觀察到一個 overhang；20 個 OFF
   requests 未量測。不能推導 0.5 的尾端機率，更不能把 OFF 當零。
   也沒有證明所有工作負載都只有一個 overhang。
2. **Opus 是 marginal release 診斷，不是 joint MPC model。**
   snapshot disappearance 保留 interval censoring；原始 cost ownership、
   來源／clock／receipt、ordered action sequence 及 joint support 都有實質修訂。
   但 midpoint censoring 不是 survival MLE；block envelope 不是 conformal
   guarantee；joint support 的平方級運算不能直接用在完整長時資料。
3. **Astra 的 same-first-action 行為不是已證明的 correctness bug。**
   R1 已明訂：最佳首動作相同時保留強 One-move 的 resolution gate。
   此策略可能放棄兩步模型預測的機會，尤其 abstention 保留目前 cap32、
   而兩個模型都建議16時，abstention 並不等於執行16。
   放寬此政策是可研究的替代策略，須獨立凍結與比較；不能當作無風險修 bug。
   不同 continuation 起點的 one／two families 未必 nested，open-loop
   score 更不保證每秒重新規劃後的真實閉環改善。
4. **Gemini 的 R3 passing tests 沒有排除所有反例。** Mayor 在凍結 R3
   實碼上再次執行：四個相同正效果的 sign-flip p=0.125；僅縮放至
   `1e-15` 就變成 p=1，原因是 absolute `1e-12` tolerance。
   `ExperimentDurationAccounting` 仍把 NaN、infinity 判為十小時 Qualified。
   兩個名為 One-move／MPC 的 compatibility aliases 實際仍是 heuristics。
   **這些 helpers 不准作為十小時、MPC 身分或最終統計驗收 authority。**
   已派出有明確停止條件的修正與 regression；原始 R3 證據不覆寫。
5. **Sol 改善了 audit binding，而非證明了獨立樣本。**
   實際 model revision、hardware/software、source/cost/scheduler、
   policy、SLO、window 進入 frozen comparison digest；exact rational
   sign inference 不依賴量綱。digest 仍不能認證派工隨機性、receipt 真實性
   或 workload independence；bootstrap CI 也不是有限樣本 coverage theorem。
   非顯著不等於持平，更不能替代 practical margin 與 multiple-baseline 判定。

## 真實 Train 提示的時間尺度，而非最佳 horizon

Opus 的 R3 報告提出 11 個 Train episodes、1,331 origins 的診斷：
1 秒 release labels 有 971 exact、330 interval，平均 normalized interval
width 約0.052；3秒接近全數已釋放，5秒沒有 exact unreleased labels。
1秒 midpoint-imputation sensitivity 平均約0.042、最大約0.40。
這些數字是 **Worker 的 Train 診斷報告**；Mayor 本次重現的是 CPU suite，
不是重新執行這份原生資料 fitting。不得將它升格為 held-out forecast 結果。

這提示原始 outstanding cohort 的長 horizon release 係數缺少辨識資訊；
**不是「所有5秒預測無效」或「MPC不可能勝出」**。未來新到達的 requests、
admission、共同 batch interference、terminal good 與 queue evolution
仍可能在5秒內具有動態。不能只為了讓 MPC 有事可做而刻意加長 output，
也不能只增加 horizon 期待模型改善。

## 下一個模型的數學邊界

這是 Mayor 選定的待驗證方向；不是已校準的四模型共識或已部署控制器。

- 精確 ledger 為 \(\mathcal L_k\)，reservation
  \(R_k=\sum_{i\in\mathcal L_k}c_i\)。保守 estimator 另輸出 running／waiting、
  visible prefill/decode composition、unknown-work fraction、request ages、
  remaining-output bounds、KV 與 recent batch/history observations。
  壓縮模型 state 不等於刪除 constraint handler 的原始 ID／cost ledger。
- Gateway 唯一已資格的輸入為 reservation **cost** budget
  \(u_k\in\{16,32\}\)，不是 concurrency、KV bytes 或 GPU seconds。
  Scheduler token budget、優先權與 prefill/decode 配額目前不是線上 actuator。
- 精確 stock-flow 必須滿足
  \(R_{k+1}=R_k+\sum_{i\in A_k}c_i-\sum_{i\in D_k}c_i\)。
  降 cap 不驅逐既有請求；因此 \(R_k>u_k\) transient 並非必然錯誤。
  新 admission 必須逐筆符合實際 cap／cost 規則，不以連續 release fraction
  提前釋放完整 cost。
- 待辨識 transition 為
  \(x_{k+1}=f_\theta(x_k,u_k,w_k)\)，observer 為
  \(y_k=h(x_k)+\eta_k\)。\(w_k\) 包含 future arrival/workload mix、
  EOS/remaining length 與 batch service variation。
  Shared batch variation 必須保留 joint correlation；不能把 marginal
  release CDF 獨立抽樣後當 joint good/release/tail guarantee。
- 暫以1秒 transition／兩個可決策 moves 作候選，保持相同 future law、
  scoring window 與 continuation 的強 One-move 對照。
  這是資訊效率假設，不是已證明 optimal sampling/horizon。
- 目標直接比較共同定義下的 successful good count，並在共同 SLO、
  rejection／safety 條件內比較；action variation 作次要 tie-break 或 Pareto
  診斷，不透過任意 cost weights 搜尋幸運勝利。
  最終主要 Goodput 邊界、實用改善 margin、樣本單位與 multiplicity
  必須在真正 benchmark／Final outcomes 前固定。

下一輪兩個模型獨立提出 joint transition 方案；另外兩個分別修復已確認
驗收缺陷、檢查真實 collector 的因果輸入與 outcome provenance。
至少交付一個可重現候選或具體不足證據，然後停止該 bounded 工作。
要求 one-step、**free-running** multi-step、Train block cross-fit 與 OOD；
direct horizon response 不冒充 recursive state transition。
原生 API 不提供的 stage、client scheduling、independent EOS 或 future
arrivals 不得捏造。Calibration／Validation／Final 不用來補救 Train 調參。

## 長時與資源快照

本輪 R3 保存時，Mayor replica 已 byte-verify **18 個完成 episodes／
2,160 秒 offered windows（0.6小時）**；這不是現在或最終總時數。
完整計畫仍為400 episodes／48,000秒，實際十小時尚未完成。
只使用既有單張 RTX4090；MPC仍未啟用。GPU billing credential／rate
未驗證，費用保持 unknown，不能拿舊 Pod 報價或停止 vLLM 假稱停止計費。

同一份 replica usage snapshot 中，四個 fresh research sessions 合計
**2,893.3051525 AI Credits**：Opus811.64126、Astra1496.9106、
Gemini329.7953925、Sol254.9579。以 `totalNanoAiu/1e9` 並依 API call
去重；不含 Mayor、舊 sessions、GPU USD 或 final invoice。
後續使用持續記錄，不能把此快照當最終總額。
