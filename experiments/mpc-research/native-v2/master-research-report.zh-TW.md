# PR32／PR33 Native MPC Master Research Report

## 目前裁定

本輪任務開始於原生辨識 commit `4d0a93394fabee29f0b8b7d7fd00c58cb28f843d`。
舊 CPU 負面結果、原生介入／profiling 與所有失敗證據保留。
**MPC 尚未證明優於 Fixed16、One-move 或合理 reactive baseline。**
HTTP first-residence 不能直接當作 exclusive serial GPU prefill service；
這個反證不代表所有 batch-aware MPC 都沒有機會。

本報告持續更新，不將研究提案、模型口頭共識、CPU 測試或 profiler smoke
混稱為真實閉環效能成功。

### Native v3 最新里程碑

**R13 實際量測資格與 R14 新反證（2026-10-31）：** 四個不同模型
R13 source 在新 Train15／16 開始前封存；Mayor 對 Source144 API 的
CPU 重跑實際為 **398 tests／196 subtests 全通過、零 skip**。
同一 RTX4090／Qwen3-4B／vLLM0.30.0 的真實 HTTP smoke 有
48秒、922 offered＝649 complete＋273 rejected、零 other。
3,346筆 Native observations 的 explicit sampling Gateway UUID 全部
可對上649個 accepted client IDs，actual ignore_eos 全為 False。
True ignore-EOS、controlled cohort 與 Native MPC superiority 仍未驗證。
兩個 Gateway epochs 的 retained aggregate ledger 共1,963筆；
status 回報 closed／零 dropped／零 write_errors，但不保證完整 suffix、
per-UUID lease ownership、認證或 fsync durability。
Mayor 另確認 Gemini 的「RealNativeCohortRunner」仍使用 virtual clock、
toy plant 與非實際 Gateway contract，不能部署或當成 Native 因果證據；
ignore_eos 不應令 request 越過 max_tokens 永不結束，Gateway sampling
UUID 也應等於 validated benchmark UUID，而不是要求兩者不同。
四模型已由新原生 Beads／caller-owned local convoy
`hq-cv-pkmce` 針對 birth-time model、stable-CRN exact solver、
真正 HTTP adapter 與 ledger closure 繼續獨立實作／互審。
Source144 的 identification03 正在執行，沒有 hotpatch 或外來 cohort traffic。
此版 progress snapshot 為11完整 episodes／1,320新秒，加舊15,960秒，
共 **17,280秒＝4h48m**；不計48秒 smoke，十小時仍未完成。
監控每120秒 off-Pod checksum replica／GPU resources／四 Worker usage，
結束／失敗後依新 PID fingerprint 停 owned inference，Pod billing stop
仍不可驗證，USD 未知；四 current sessions 在14:51:55Z 為
14,564.7298625 credits，排除 Mayor／舊 sessions／invoice。
詳見 [R13量測與R14反證](../native-v3/evolving-model-negative-and-observation-support.zh-TW.md#nv3-016r13-實際-native-量測與-r14-資格反證)。

**R12 封存與 Native 停止：** Continuation02 在13:28Z因一次 pre-header
HTTP failure停止，失敗raw/events/index完整保存；只確認約3.85ms無headers，
exception class當時遺失，不假定keep-alive或GPU root cause。
累積有效15,960秒＝4h26m，十小時仍未完成。Owned GPU processes／ports
停止已驗證，Pod保留且provider billing stop／USD未知。
R12 v9 event-order修復大部分v8預測偏差，但development仍未達5%；
fresh superiority gate尚未執行，
尚非MPC candidate。Astra另找stable-per-birth CRN correctness缺口。
新sampling lineage／typed transport error／exact ledger準備的Parent137tests通過；
舊R12 seals對新source的7base／9subtest failures保留並由R13修正，
不將歷史377tests當作新版本通過。Mayor已曝光Train12／13 summary，
下一fresh pair改15／16；Cal／Val／Final未開放。
四個原生R13模型任務／caller-owned localconvoy已啟動；GPU恢復與
真正closed-loop仍待source-qualified preflight。
詳見 [失敗與觀測資格](../native-v3/evolving-model-negative-and-observation-support.zh-TW.md#nv3-015r12-封存native-失敗與下一版資格)。

**R11 evolving model 與支援邊界：** 四模型先獨立封存，再開放互審；
Mayor 實際 323 tests／177 subtests 全部通過、沒有 skip。
Opus batch-aware v8 在 USED8／9 的 full G5 MAE7.67／7.70，
明顯劣於 fixed-mark v7 的4.11／4.20，Mayor raw byte-for-byte 重現。
Aware 比 blind 好，不能抵銷完整預測、rejection bias 與 coverage 的失敗。
Astra／Sol 的短期 stayer counter gains 不能補足三／五秒完整 cohort support；
Gemini mock −58tokens 不是 GPU 物理效應。新 hypothesis 集中於
intra-step release ordering 與真正穩定 cohort intervention，不盲調 MPC 權重。
13:06Z 已完成新11,160秒＋舊3,600秒，共4h6m有效offered windows，
十小時尚未完成，Native MPC 未啟用。
詳見 [evolving負面結果與觀測支援](../native-v3/evolving-model-negative-and-observation-support.zh-TW.md)。

**R10因果缺口與精確加速：** 原R9 release sensitivity raw已byte-for-byte
重現；兩個endpoint預測較強baseline好，但G5 bands只有.595／.656，
不選有利endpoint或宣稱risk。修正ABI／mixed-schema後Mayor286tests／165subtests
全部通過。Actual prefix benchmark在1111cuts、1,137,664scenario comparisons
完全等價，paired median加速1.28–1.31倍；deadline350→285，短高rate仍全超時。
全部offline first-action differences仍0，不是GPU改善。
Fixed-mark模型無法更新future admission對既有cohort／earlier births服務的影響，
已由source與fixed-CRN witnesses證明；physical effect magnitude仍待因果驗證。
已轉向四模型獨立batch-aware evolving prototypes；早期leaked／fabricated
diagnostic已拒絕，fresh12／13與Final未曝光。詳見
[固定marked因果缺口與exact prefix](../native-v3/fixed-mark-causality-and-exact-prefix.zh-TW.md)。

**R9真正optimizer與未通過的CPU gate：** Mayor實際重現全部1111個
frozen-v5 cuts；與Worker共同完成的29,300個plan means精確一致，
但100ms deadline依環境分別造成303／354次未完成。
完整family為808／757，first-action differences均為0，不能將deadline actions算相同；
不是Native MPC改善。Mayor一次cooperative callback實測262ms，沒有hard100ms保證。
四module CPUqualification5failed／258passed／165subtests，foreign-ABI
pipeline path正另版修正，不skip或重裝救分數。
Sol／Mayor獨立重現cyclic survivor kernel bias，原始v5結果保留，
新repair先preregister再用未使用Train10／11；同時四模型獨立檢查
future batching interference是否能影響已存在cohort的service。
截至12:12Z續跑8040秒、舊完整3600秒另列；十小時未完成，live729未替換。
詳見[真正solver重現與qualification](../native-v3/mean-optimizer-qualification.zh-TW.md)。

**新Train6／7未改動v5 replication：** 先以517f928預註冊，再freeze20episodes，
同Train1–3 fit／同source，1111origins的G5 MAE4.4234／4.2260，
分別低於最強同-mask5.9191／5.6450；paired4.3661／4.0481也低於zero。
但80%bands inclusion仍只有0.577／0.614，不能宣稱risk／MPC成功。
發現v5 descriptive mean_truth只保留每episode最後origin；primaryMAE／bias／
paired calculations不受影響，原始source／raw保留，另存corrected origin audit。
已選定mean-only admission/birth研究candidate，開始solver／真正HTTP-control
pipeline／online-input工程，沒有部署liveGPU。
詳見[replication](../native-v3/frozen-model-replication.zh-TW.md)
及[architecture ADR](../native-v3/experimental-mpc-architecture.zh-TW.md)。

**最新R8 fitted model：** Mayor241tests／152subtests通過，
admission／birth hybrid的五秒Good MAE4.08／同-mask additive5.93，
paired factual delta4.17／零差異5.91，raw byte-for-byte重現。
但一秒較差、Good bias−1.26、rejection bias+3，scenario bands under-cover；
不是MPC Goodput提升、GPU service law或plan-ranking保證。
另一個direct IO fit的負面結果也已重現；兩模型aggregate masks不同，不能直接投票。
真正callback bridge與公平CPUharness完成，Native整合／risk／MPC permit仍缺。
未改動v5的Train6／7 replication先預註冊，Final仍未曝光。
詳見[fitted birth-response與runtime工程](../native-v3/fitted-birth-response-and-runtime-components.zh-TW.md)。

**最新R7反證：** 四模型完整互審 source 已凍結，Mayor217 tests／146 subtests
通過，Native Trace plan-response raw JSON 與 Worker byte-for-byte 相同。
五秒完整 additive plan 的 Good paired-difference MAE8.15，
仍輸零差異6.90，雖 level MAE5.84 較低。這是不同 arm state 的 factual
diagnostic，不是 same-state action 效果，也不代表 MPC 普遍無效。
實際 Train 有全部32種五槽 pattern；conditional support／ACK law／
selected-policy coverage 仍未成立。R8已派工兩個 fitted model、真正receipt
bridge與公平benchmark harness，不看Final、不搜索weights/horizon。
最新已核對 continuation3,600秒與原long01的3,600秒分列，十小時未完成。
詳見[plan contrast反證與實作裁定](../native-v3/plan-contrast-falsification.zh-TW.md)。

**Native state adapter 已實作：** source／clock-first、running-only token
progress、unknown waiting work 與 Gateway whole-cost owners 分開保留。
67項 coupled tests 及17,909個 recorded-receive frames replay 通過；
沒有部署至 GPU，沒有把 output headroom 當成 EOS 剩餘工作或授予 MPC permit。
新 package module 改變本機 implementation hash，live729仍不變。

**最新R6訊號與邊界：** 四模型真正完成獨立提案後已凍結，
Mayor204 tests／139 subtests通過。Mayor另外重現Native Trace progress
診斷：1／3／5秒Good-count MAE約1.07／3.45／5.90，
history約2.70／5.69／8.76；五秒相對history改善約32.6%，
但相對更強aggregate hybrid約1.7%，不能歸因全部於progress。
這仍是已使用Train development資料的預測，不是MPC Goodput提升；
完整candidate-plan response、uncertainty與Native閉環仍缺。
四模型已開始R7比較causal hybrid kernel與尚未裁定的full-plan direct model，
不看Final或搜索horizon/weights。
詳見[progress預測與plan-response缺口](../native-v3/progress-prediction-and-plan-response.zh-TW.md)。

**最新模型反證與介面進展：** Mayor已byte-for-byte重現Opus R5第三個
Train development block：外生arrival修訂的5秒Good-count MAE9.84，
仍輸history mean8.76；oracle未來offers也只有8.78，不能只怪arrival。
本機`734d459`新增真正client attempt／first-content／terminal publication hook，
102項coupled tests通過；沒有倒填舊row或部署至正在GPU執行的729。
新的source／consumer／Native資格與joint kernel仍待驗證。
R6正在保留獨立性的token-progress／batch-dynamics研究階段，
尚非四模型共識或MPC勝出。
詳見[線上receipt與第三個Train反證](../native-v3/online-receipts-and-third-train-falsification.zh-TW.md)。

**最新長時結果：long01在30個完整episodes／3,600秒後因三次guarded
assignment未獲ACK而中止；十小時尚未完成。** 失敗partial與pending
證據已off-Pod保留；owned Gateway與vLLM已停止，Pod保留且billing停止
未驗證。其後`729b144`不放寬TTL的confirmed-hold修訂已通過真實
4 episodes／48秒／922 offered資格：16次applied ACK、32次fresh holds，
636自然API-stop完成、286拒絕；不是MPC效能比較。
Fresh continuation02已啟動，planned370 episodes／44,400秒，live Gateway
與原生服務已驗證；十小時實際完成尚未驗收。原失敗資料完整保留，
新增off-Pod／resource monitor及指紋核對後owned-service shutdown。
停止vLLM仍不等於暫停Pod計費，現有Pod保留、lifecycle credential缺失。
四模型R4已凍結，Mayor171 tests／128 subtests通過；原生Trace recursive
joint forecast在1秒有訊號、5秒Good仍輸history baseline，沒有MPC勝出。
詳見 [長時失敗、實際joint反證與受控恢復](../native-v3/long-run-failure-and-joint-model.zh-TW.md)。

目前本機工作分支為 `research/pr33-native-v3-ten-hour`。新 Pod 的 vLLM
0.30.0／Qwen3-4B／RTX4090 已完成 16-arm、40 自然 EOS HTTP qualification。
新版 physical／delivered accounting 有效，8 個原 strict-v2 失敗完整保留；
詳見 [可追溯修訂與封存 SHA](../native-v3/qualification-revision.zh-TW.md)。
四個 fresh native Polecat 已完成獨立 R1，Mayor 重現 65 項測試與
68 個 subtests，並已交付 R2 交叉批判。

`6c11fe7` collector 已啟動，凍結 1 秒 cap16／32 互補隨機干預、四種純 workload
及混合 workload，分離 16 train、20 calibration、4 validation arrival
seed blocks，不使用 final held-out。正式計畫的 offered windows 合計
48,000 秒，不以 max drain、setup 或 idle 補成十小時。
**尚未宣稱完成十小時或 MPC 優勢**。已執行兩個 bounded interface
smoke；後一個跨兩個 planned Gateway epochs，922 offered／650 自然完成／
272 拒絕、48 秒，source／cost／clock／CAS／HTTP／drain／owned-process
cleanup 均驗證。既有 1,024-operation ceiling 保留；每個完整配對、
drain 後切換 Gateway epoch，不重啟 native vLLM，也不忘記本 epoch 的
operation outcomes。長時資料逐個 completed episode 封存到 Mayor。

四個 backend 已完成三輪交叉審查；最新 R3 從各自提交凍結，
Mayor 重現142 tests／107 subtests。仍有已重現的 Gemini numeric／
duration／heuristic-alias 缺陷，因此未准作為驗收 authority；
Astra same-first-action 保守政策的機會成本保留為未決設計問題。
下一輪集中於短步、batch/history-aware **joint stock-flow** transition，
不將 marginal original-release 模型直接接成 Goodput MPC。
詳見 [第三輪證據裁定與下一輪模型](../native-v3/round3-adjudication.zh-TW.md)。
十小時與原生 MPC 效能比較仍未完成，不把正在運行的計畫當結果。

Mayor 已另外執行兩個完整 Train block 的凍結 R3 cross-fit。
batch-aware original-release normalized error bounds約0.033–0.052，
train-fraction baseline約0.090–0.111、fluid-history約0.150–0.181。
這是原生 traces 的初步預測訊號，不是信賴區間、完整 joint state rollout
或 MPC效能改善；5秒 original-release labels仍degenerate。
詳見 [固定兩個Train block的完整診斷](../native-v3/train-crossfit-diagnostic.zh-TW.md)。

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
| opal | pr-nv2-opus | claude-opus-5.5 | Round1–3 研究交付完成；未認證 Native 控制器 |
| topaz | pr-nv2-astra | gpt-6-astra | Round1–3 研究交付完成；未認證 Native 控制器 |
| garnet | pr-nv2-gemini | gemini-3.8-flash | Round1–3 研究交付完成；新版 evaluator 仍被駁回 |
| ruby | pr-nv2-sol | gpt-6.1-sol | Round1–3 與 collector source review 完成；Native 待跑 |

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

### Round2：四模型互審已凍結，32項反例可重現

原始四份報告與各自反例／測試已依
`results/mpc-native-v2/round2-freeze.json` 保存，Mayor 的共同 Python3.12
環境執行 **32 passed**。這代表反例可執行，不代表報告的每一句都成立。
已確認的核心問題包括：降 cap 不會即時釋放舊 reservation；
「settling time 大於 slot 就必然有 multi-step 優勢」不成立；
Gemini 的統計原型會接受重複 seed blocks、錯誤 solver 分母及缺失 rejection
guard，不能用來認證性能。Opus 的 knee 選擇先看過外層 held-cell，
warmup 則曾刪去仍影響後續請求的歷史，必須限縮原結論。

**Mayor 明確駁回未量測數字：** Gemini Round2 宣稱 facade 約4.5µs、
CUPTI 約450µs，以及零 serving jitter，實際只是 CPU fixture 的常數；
本輪 GPU 尚未啟動，這些不是測量結果，已要求 Round3 撤回。
其餘新批判也仍須限縮：fitted mean 不是 capacity lower envelope；
output increment 不是自動具資格的 GPU iteration clock；
兩個固定 cap 的整段分數相同，不能推出所有切換策略都無價值。
下一輪以逐條回應及修訂設計裁決，不以三對一表決或共識取代證據。

### 可執行的第一個 Native 資格關卡

新增獨立 opt-in plugin wheel、loopback-only launcher 與
`collect_batch_qualification.py`，重用標準 Chat collector、source38 guard、
drain 與 natural-output work audit。凍結方案為 TRAIN170001／170002、
p128／4096、cohort1／4、capture OFF／light，共16個短 arm，固定 cost
budget32；完整保留 startup、`stop`／`length`、HTTP失敗與無效 capture。
它只驗 request／scheduler／worker／commit joins、正常 graphs／async、
pending events 與 drain，不做服務係數、observer equivalence 或 MPC 性能驗收。

collector 先將原始 utility 回覆與 capture 寫入／fsync、重新核對 SHA，
才送 ACK；digest／epoch 錯誤不 ACK。四項新增測試（含無效 raw 的保留）
與18項 facade cases 合計 **22 passed**。Plugin wheel 已用宣告的 isolated
build backend 建置；無隔離建置曾因 CPU 環境缺 setuptools 失敗，沒有將
那次失敗記為通過。實際 GPU 安裝／HTTP 執行仍待後續證據。

**OFF 的範圍限制：** capture OFF 仍載入 wrappers／GIL 設定，不是 literal
disabled baseline。後續 observer-effect pilot 必須使用乾淨重啟的 disabled
部署，不能拿這16個 arm 宣稱零擾動或已定位 roofline knee。

資源記錄截至 `2026-10-31T01:03:12.570024+00:00` 的已完成 accounting prefix：
Mayor 加四 fresh workers 累計 **2323.653925 AI Credits**，實際 backend
分別仍為 Opus5.5／GPT-6 Astra／Gemini3.8 Flash／GPT-6.1 Sol；
原始 usage IDs 不輸出，prefix bytes／SHA 留在 round2 execution ledger。
這個 snapshot 含已入帳的少量 Round3，不是最終總額或精確單輪費用，
也不能換算為美元。01:07:49 provider 再確認 Pod EXITED／runtime=null；
本輪尚未新增 Native GPU 實驗時間，單卡報價 US$0.89/hr，實際 invoice 未知。

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
上述資格完成時尚未嘗試 resume；後續一次實際恢復被拒，見下節。
不能將 mock resume 或實際被拒的 mutation 描述為已重啟 GPU。

## Round NV2-003：修訂、獨立裁定與整合驗證

四模型已各自完成三輪，而不是 Mayor 假冒四份分析。Round3 原始提交、
報告／原型／測試凍結於 `round3-freeze.json`；Mayor 重現 **26 passed、
2 subtests passed**。Sol 另外檢查 `ab8c392` collector 的完整
`audit_capture(..., forced_outputs=False)` 與 exact request-ID joins，
沒有發現必須先判資格 smoke 無效的 critical defect；這仍不是 Native 驗證。

Mayor 不接受「測試通過＝設計正確」。新版 Gemini evaluator 仍能把同一
block 複製六次認定 superior，且把0 output token／NaN origin 計為 good；
三個反例實際執行並存入 `round3-mayor-adjudication.json`。原稿保留，
不以投票、刪除反例或放寬指標取得表面共識。

目前暫選 history-conditioned marked-reservation release／good-count 模型，
以 batch-aware labels 解釋誤差，並與 fluid、hybrid、learned、piecewise、
probabilistic 結構比較；它只是辨識候選，**不是已校準或已勝出的 MPC**。
完整數學邊界、固定主要 forecast metric、強 One-move 對照及下一組高資訊
實驗見 `decision-and-next-native-protocol.zh-TW.md`。

兩個舊 P2 在目前 source 與 regression cases 中均已有修正：
`prefill_lab/state_envelope.py:31–80` 將 specification、features、
dependency hashes 和 generated cost hashes 綁入 support，PreparedEnvelope
保留驗過的規格；`:126–136` 先判 clock-domain mismatch，再比較 uptime。
`tests/test_prefill_state_envelope.py:157–224` 包含跨 boot fallback、
相同 clock 的倒退拒絕、cost semantics／dependency 變動及 prepared-cost
測試。本輪沒有為這兩項重複修改產品程式。

### 資格工具的精確修正與完整 CPU 結果

新增 guard 禁止 active／未 ACK capture 期間執行重型 metadata 探測；
獨立 opt-in plugin 版本為0.1.1，wheel 可建置。
初次 root collection 暴露 Round1／Round2 Sol 模組同名衝突；
只正規化整合副本的 Round2 module／test 名稱與 import，原 Worker blobs
與科學邏輯保留，不以忽略整個研究目錄解決問題。

| 實際執行範圍 | 結果 |
|---|---|
| 修訂 bridge／collector 與既有 pipeline／wire | 85 passed、1 skipped |
| 正常 repository root CPU suite | **2206 passed、1 skipped、24 subtests passed，256.16秒** |

範圍互有包含，不能相加當作獨立樣本。唯一 skip 是 CPU 環境未安裝 vLLM
的 installed-vLLM protocol test；不是通過。完整 suite 有一項既有
Starlette/httpx deprecation warning。JUnit 原檔均保存在
`results/mpc-native-v2/`。沒有將 CPU suite 說成 GPU、HTTP 或 MPC 驗收。

## Round NV2-004：實際恢復被拒與資源安全阻塞

已先凍結 source `ab8c392` 與16-arm TRAIN plan-v1、備妥 source archive、
plugin wheels 和 fail-stop collector，才對唯一既有 Pod 發送一次 resume。
供應商回覆 business error；lifecycle helper 先保留 intent／unverified
receipt，沒有盲重送。01:13:34 再確認 EXITED／runtime=null；
01:18:13 及 **01:32:31 UTC** read-only 查得原 RTX4090 machine
`gpuAvailable=0`，Pod 仍 EXITED、GPU release 已確認。

因此**本輪新增 Native arms=0**，沒有新模型 prediction／closed-loop
goodput、GPU latency、solver-overhead 或 MPC 勝率結果。
容量為0與拒絕相容，但不足以認定它是唯一拒絕原因。沒有另租機器、
修改 key／remote、嘗試舊 SSH endpoint 或讓 GPU 等待 CPU 討論。
根因及後續 reconciliation 保存於 provider receipts／lifecycle JSONL。

任務開始至安全 preflight 約5分4秒，當時沿用已在運行的 Pod；以
US$0.89/hr quote 粗估此準備區間約US$0.075，**不是實際帳單**。
之後 provider EXITED，沒有新 GPU 實驗運行時間；不能說整項歷史任務
GPU 花費為0。未取得 provider invoice，也未把規格報價當成仍在計費。

截至 **2026-10-31T01:33:28.744429+00:00**，新的 immutable accounting
prefix snapshot 為 **3135.1236375 AI Credits**：
Opus942.8696、Astra1260.9647、Gemini172.8993975、Sol215.12431、
Mayor543.26563。逐筆去重、實際 backend 及 complete accounting
均見 `agent-execution-ledger-round3.json`。這是 mission-to-date，
不是單獨 Round3、美元或最終帳單；後續整理／發布／回覆不在該 snapshot。

### 下一個可驗證動作與發布範圍

既有資源可恢復後先跑 Gate A 的標準 Chat／source／join／drain 資格，
再跑 Gate B 的 disabled/OFF/LIGHT、LL/LH/HL/HH 與 same-action repeats。
先辨識 release／good-count、observer effect、EOS repeatability，
再進入新模型與 Fixed16／best-supported-fixed／reactive／強 One-move／MPC
的公平比較；不繼續盲調 horizon 或 cost weights。

原 plan-v1 未執行且保留。metadata 修正後的 source 必須另凍結新版計畫，
不能假裝舊計畫已測過新 source。現在已另外凍結 plan-v2，對應完整 CPU
驗證的 source `9820231228f6859cd75abbf8a1d04d770490ab28`，仍未執行。
`prepared-deployment-v2.json` 保存 archive／兩個 wheels 的 SHA，
且逐一確認 wheel module 與 committed source 完全相同；
產品 implementation hash 也實際重算確認仍為上述 `b4f234…`。
Final held-out 未查看；最小有意義改善、
獨立 block 數／power、CI 與共同 rejection／safety guards 尚待 development
資料後預先註冊，不能由目前 pilot 宣稱穩定 superiority 或 equivalence。

可發布成果是四模型證據、反例、數學／實驗裁定、資格工具及真實 CPU 結果，
**不是 Native MPC 性能勝利**。只由 Mayor 核對遠端 ancestry 後，
以明確 PR33 ref fast-forward 發布；worker guards 保留，不使用 `gt done`、
force push、Merge、Rebase、remote 變更或 main。實際發布以獨立 receipt
與 GitHub HEAD 核對為準。

**原生追蹤限制：** caller-owned／merge-local convoy `hq-cv-3977g` 確實存在，
但本次結案核對時 `gt convoy status --json` 回報 `tracked=null`、0/0。
`gt convoy add` 雖回覆加入四項成功，重新查詢仍是0/0；不可把它寫成
「convoy 已驗證4/4完成」。四個實際 Beads、派工、Worker source commits、
三輪成果與 usage 可逐一核對。Mayor 直接記錄各研究交付的本機完成，
epic／Native 資格仍被 GPU 阻塞；不為這個 metadata 異常修改其他 Rig、
global routing 或 Gas Town 程式，也不呼叫有 Push 風險的 completion。
普通 `bd close` 因 Mayor actor 與 Worker assignee 不符而拒絕；
Mayor 以行政 `--force` 結案僅限已核實的研究交付，保留原 assignee，
沒有解除 Native／MPC 驗收 gate。詳見 `local-completion-receipt.json`。
