# PR32／PR33 Native MPC Master Research Report

## 目前裁定

本輪任務開始於原生辨識 commit `4d0a93394fabee29f0b8b7d7fd00c58cb28f843d`。
舊 CPU 負面結果、原生介入／profiling 與所有失敗證據保留。
**MPC 尚未證明優於 Fixed16、One-move 或合理 reactive baseline。**
HTTP first-residence 不能直接當作 exclusive serial GPU prefill service；
這個反證不代表所有 batch-aware MPC 都沒有機會。

本報告持續更新，不將研究提案、模型口頭共識、CPU 測試或 profiler smoke
混稱為真實閉環效能成功。

## 任務權限與分支

使用者已取消固定 AI Credits／GPU 時數上限，允許同一個既有 Pod 的自主實驗、
服務啟停、本機實作與提交；不得擴大付費資源種類、同時使用多台 GPU、
改寫公開歷史或操作其他 Rig。
工作分支是新建本機 `research/pr33-batch-dynamics-v2`，不直接修改原 PR branch。
原 PR32 已合併，PR33 是 Draft，開始時遠端 HEAD 為
`0588707949f651039d44a7a034c12bd35b8c8047`。
使用者最後另有「有結果請推至PR33」的條件式授權；
只有 Mayor 可在結果與內容核對後執行明確 PR33 ref 的發布，不授權 worker 自動完成推送。

## Round NV2-000：資源與完成路徑隔離，已執行

| 欄位 | 記錄 |
|---|---|
| Experiment ID | `native-v2-safety-preflight` |
| 起始 Commit | `4d0a93394fabee29f0b8b7d7fd00c58cb28f843d` |
| Model／Controller | 無效能模型修改；controller 未啟動 |
| Workload | 本機 bare receiver 傳輸探針；未向 GitHub 發送 push |
| 假設 | 正常 completion 的 primary／bare fallback 不能意外發布；閒置 Pod 應真正退出計費 GPU |
| 實測 | 兩個 Git common-dir 的 pre-push 均拒絕；receiver 無 refs；wrapper 拒絕 `gt done`／`git push --no-verify` |
| Provider | 既有 Pod `i243ycb94oaxy2`；GraphQL 報價 US$0.89/hour |
| 儲存 | volume0GB、container50GB；停止前六份最新 raw outcomes／telemetry 與本機 sealed archives 全部 byte-match |
| Pod 結果 | `podStop` 回覆 EXITED，後續 query 為 EXITED、runtime=null；不是僅停止 vLLM |
| 下一步 | GPU 暫停期間進行四個 fresh native Polecat 的獨立 CPU 研究；完成量測方案與測試才 resume 同一 Pod |

生命週期 credential 只保留在 repo 外 mode0600 檔案，不加入 report／archive／worker
任務內容。Pod 的 ephemeral model cache／venv 可重建；研究原始資料已另存本機。
停止後的 `costPerHr` 欄位仍可能是規格報價，不是仍在收取 GPU 費用的證據；
帳單結算與稅費未取得，不虛構累計實支。

### 防 Push 的實際機制與範圍

已查閱安裝版本 Gas Town1.2.1 的 `done.go` 與 `git.go`：
一般 completion 會先 push，再建立 MR；失敗會退回 `.repo.git` 或 Mayor repo 推送。
因此兩個 Git common-dir 都設置拒絕式 `pre-push`，不變更任何 remote URL。
測試只使用本機空 bare receiver，未執行真正的 `gt done`，也未接觸 GitHub push。

pr32 專屬四個 profile 保留原 model flags，另外加入 `exec_wrapper`：
將 worker 的 `git push`／`send-pack` 與 `gt done` 在 process boundary 拒絕。
新 dispatch 同時使用 local convoy、no-merge 與 raw bead，避免自動完工公式。
完成方式是保留本機 commit、產出 report／receipt、mail Mayor，之後由 Mayor 收斂。

這是對正常／意外 completion 的技術隔離，不是假稱能 sandbox 惡意同 UID agent：
明知而繞過 wrapper、指定 `--no-verify` 或改 hooks 的 hostile worker 仍超出此保證。
舊 Onyx 等四個事故 workers 不重用，不重寫或清除其證據。

## Ground truth 與尚未解決的缺口

已重新查閱 PR32／PR33 GitHub HEAD、commits／inline review、歷史事故、
最終 CPU 報告與最新原生辨識報告。舊報告內 US$5／2.5h／逐次授權限制
是歷史條件，已由本次使用者授權取代，不能繼續當作 GPU 停工理由。

原生已有13792 intervention offers、32 natural-EOS isolated probes；
另22筆刻意 max8 的 profiler HTTP probes 不能加入 natural-EOS dataset。
840個 trace kernels 在 batch annotations 外，不能任意當作 prefill；
GPU span 含 gaps，kernel busy、CPU return 與 request latency 必須分開。
低負載批有1865個 Gateway-accepted／Native尚未出現的 work marks，不能填0。
原型的 source／cost／clock guard 與正常 actuation qualification 不能被省略。

接下來必須比較 queueing/fluid、batch-aware state-space、hybrid、learned transition、
piecewise nonlinear 與 probabilistic service 結構。
每個候選都需 one-step、multi-step、OOD 與 action-relevant 評估；
不把 correlation 當 action 因果、不以 training fit 宣稱可控制。

## 四模型研究與公平性規則

已建立四個 fresh native Polecats，模型分別為 Claude Opus5.5、GPT-6 Astra、
Gemini3.8 Flash、GPT-6.1 Sol。A/B/C/D 是主責重點，不取消共同核心問題：
每個模型都須獨立提出完整 state／action／disturbance、dynamics、observability、
identifiability、MPC objective／horizon、安全機制及判別實驗。
Round1 完成之前不交換設計。Round2 互審，Round3 依實驗或可重現反例修訂；
未解分歧保留，不能用投票代替證據。

profile 名稱與 argv 不是實際模型證明；必須逐一讀 usage records 的 model，
記錄 session、token／credits 可得欄位與各自報告 hash。
本輪已從各 session 的 `session.usage_record.data.usage.model`
確認四個實際 backend 模型，非僅 profile／argv。
原生 epic 為 `pr-native-v2`，caller-owned、merge-local convoy 為 `hq-cv-3977g`。

| Polecat | Native Bead | 實際 usage model | 狀態 |
|---|---|---|---|
| opal | pr-nv2-opus | claude-opus-5.5 | 獨立 Round1 已凍結 |
| topaz | pr-nv2-astra | gpt-6-astra | 獨立 Round1 已凍結 |
| garnet | pr-nv2-gemini | gemini-3.8-flash | 獨立 Round1 已凍結 |
| ruby | pr-nv2-sol | gpt-6.1-sol | 獨立 Round1 已凍結 |

Gas Town 強制將 `--base-branch` 當作 origin ref，未發布的本機分支不能直接用；
因此每個 worker 改以不同新建本機 branch 的 `--branch` 原生派工，
不建立假 origin ref、不改 remote，也不重用舊事故 identities。
其 startup sandbox 擋住 localhost3307，導致 `gt hook` 誤判沒有工作，
不代表 Host Dolt 或 Beads 遺失。Mayor 在 Host 驗證四個 Beads／hook，
再用原生 nudge 直接交付完整任務；metadata 由 Mayor 寫入。
不因此更改 global sandbox 或其他 Rig。
至少 Opus／Astra 在 startup 預設流程曾嘗試 `gt done`，均被 wrapper 拒絕；
這是技術屏障實際生效的紀錄，不能說只靠指令成功防止嘗試。

### Round NV2-001：四模型獨立提案與 CPU 診斷，已執行

四份提案及各自 source commit／逐檔 SHA 已凍結於
`results/mpc-native-v2/round1-freeze.json`；全部報告 ready 後才釋出互審。
整合只擷取各自三份研究檔案，排除 auto-checkpoint 混入的
CLAUDE／Gas Town hooks／slash-command 設定，保留原 Worker 分支與提交。
Gemini 原稿的十五處行末 Markdown 空白在整合時正規化，
不更動科學內容；原稿仍可由 freeze manifest 的 source commit 取得。

Mayor 使用既有 Python3.12 環境獨立執行四個新診斷的組合測試：
Opus17、Astra9、Gemini18、Sol8，共 **52 tests passed**。
這證明這些 CPU 範例在共同環境可執行，**不認證所有報告中的數學、
物理或統計主張**；尤其不是 Native MPC 性能勝利。
模型假設、claimed theorem、clock decomposition、成本降額約束、
survivor coefficients 的物理解讀仍須 Round2 反例審查。
目前沒有因四模型高層次用詞相似就直接裁定「共識已成立」。

下一輪同一批原生 Workers 各自審查其他三份 frozen report／prototype，
保留原 Round1，不回寫刪除錯誤主張，將反例與修訂另外保存。
GPU 繼續維持 provider EXITED，直到可執行的量測方案與資格檢查就緒。

### Mayor 並行工作：重用 PR32 的 passive batch capture

`pr-nv2-batch-bridge` 正在進行資格準備；沒有重建另一套 CPU MPC。
新 `batch_capture_bridge.py` 只包覆既有 `ExternalCapture`、
`scheduled_work`／`step_key`、`WorkerRecorder` 與38檔 installed-source guard。
預設不載入 GPU hooks；專用 opt-in 啟動才允許 frozen TRAIN170001–170012
的 off／light capture，沒有 synthetic request、live predictor、actuation、
scheduler replacement、profiler 或同步 GPU 等待。

量測為同一 batch identity 的 engine schedule／host commit 與
worker current-stream preparation／forward CUDA span；
**不把 forward span 認證成 kernel busy 或整條 GPU pipeline 完成時間**。
Worker clock／source／process epoch 在開始與結束核對；
pending CUDA events、overflow、missing terminal／batch IDs 均標 invalid，
原始回覆仍保留。結束回覆在 EngineCore 記憶體可重讀，須由 collector
先寫入／fsync 並核對 SHA，再 ACK 才可開始下一 trial；
這不是 process-crash durability 的保證。

CPU 組合驗證為 **80 passed、1 skipped**，包含新 facade18項及既有
pipeline／HTTP wire contracts。被 skip 的是實際 installed-vLLM
CompletionRequest 測試，原因是 CPU 環境未裝 vLLM；
不能將它計為通過或以此宣稱 Native HTTP 已資格化。
下一步為四模型批判後的 installed-source／真 HTTP／drain 資格，
再進行未開 profiler 的 batch-law Train 實驗。

研究用共同 SLO 沿用 TTFT500ms＋request-mean TPOT50ms；
真正逐 token tail 若沒有 timestamps 資格，明確標為不可測，
不能把 SSE chunk gaps 改名為 token TPOT P99。
正式 confirmatory protocol 的最小有意義改善與統計判定，
將在 development variance／power 設計後、最終 held-out 產生及查看之前封存。
原160021–160024 Test 尚未執行，不能納入調參。

## 實驗與資源 ledger

每輪 JSON 至少保存 experiment ID、commit／model／controller versions、
workload／GPU environment、hypothesis、intervention、results、failure analysis、
next action 與資源記錄。資料位於 `results/mpc-native-v2/`。
GPU 時間以 provider RUNNING／EXITED receipts 與 UTC 記錄估計，
報價估算與 provider 實際帳單區分；AI Credits 以 runtime usage 提供的量計，
若只提供 tokens／billing multipliers，就不杜撰成美元或任意 credits。

新增 Mayor-only `pod_lifecycle.py`，只有既有 Pod 的 status／stop／resume，
不提供 create、terminate、多 GPU 或任意 Pod 參數。
Resume 要求具 Commit／假設／workload／停止條件的實驗計畫；
Stop 要求近期 remote-to-local sealing receipt 與未變更的本機 artifact SHA。
Mutation intent 先 fsync，模糊回覆不盲目重送；
EXITED acknowledgement 不等於釋放，必須再確認 runtime=null。
九個失效案例單元測試通過；實際 status 再次確認 EXITED／runtime=null。
官方 `podResume(input:{podId,gpuCount:1})` schema 已核對，
但本輪尚未 resume，不能將 mock resume 描述為已重啟 GPU。
