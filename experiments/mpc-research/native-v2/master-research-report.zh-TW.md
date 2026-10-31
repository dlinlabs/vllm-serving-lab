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

將建立四個 fresh native Polecats，模型分別為 Claude Opus5.5、GPT-6 Astra、
Gemini3.8 Flash、GPT-6.1 Sol。A/B/C/D 是主責重點，不取消共同核心問題：
每個模型都須獨立提出完整 state／action／disturbance、dynamics、observability、
identifiability、MPC objective／horizon、安全機制及判別實驗。
Round1 完成之前不交換設計。Round2 互審，Round3 依實驗或可重現反例修訂；
未解分歧保留，不能用投票代替證據。

profile 名稱與 argv 不是實際模型證明；必須逐一讀 usage records 的 model，
記錄 session、token／credits 可得欄位與各自報告 hash。
本輪尚未啟動的 worker 不計為已執行的四模型結果。

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
