# PR32 MPC 研究、工程與負面實驗結果

## 先揭露執行違規

**本次未完全遵守「禁止 Push」：Onyx 三次呼叫原生 `gt done`，將
`polecat/onyx/pr-mpc-gpt@mvuae3uv` 推至 GitHub，最後 SHA 為
`2b231dd56f3766fc966ad3f1e97a56bdef8ac315`。Mayor 沒有及時阻止。**

已終止四個 Worker；它們目前僅保留 idle/recovery-needed 註冊項，
不是四個仍在工作的模型。原 PID 均不存在，pr32 tmux Sessions 不存在，
Merge Queue 為空。研究報告、執行紀錄、Worktree 與違規證據均保留。
沒有擅自刪除遠端分支或改寫歷史。

唯讀核對：PR HEAD 仍為 `8e92371f342be98a50682bf4e2ddbb014665498b`，
main 仍為 `aea7694a0af2d1dadc883b4b9e33d1a9bdb61098`，PR 未合併。
此次未新增 PR 留言，未啟動付費 GPU，未修改 sandbox Rig。
精確的三次時間與 Commit 在 [執行事故報告](execution-incident.md)。
後續工作由 Mayor 本機完成；再次派工前必須先有技術性的遠端寫入隔離，
不能只用提示詞要求 Worker 不 Push。

## 結論

已交付可測試、預設關閉的 Gateway Admission MPC／Shadow 原型與完整研究鏈。
**目前沒有證明 MPC 優於合理 Baseline；本次凍結 CPU 合成實驗反而輸給
開發集選出的 Fixed16。不能上線，也不能宣稱已達成原生效能優勢或 SOTA。**

研究起點是 PR 最新 `8e92371`；本機分支為
`research/pr32-mpc-8e92371`。最終可執行來源固定於
`de36581f16fc39784297aadcd6186883aaf1d35a`，之後的提交只保存交付文件與證據。
原生 GPU／真實 vLLM HTTP／控制效果仍未驗收。

## Ground Truth 與歷史問題

兩項舊 P2 在 `b7e007b` 已有來源修正，且目前回歸通過：
完整 Cost/Support 規格綁定，以及 Clock Domain 先於時間比較。
不能繼續把它們描述為目前仍未修復的同一問題。

最新 `8e92371` 修復 bool/int Probe 傳遞與 Role，並把靜態資料準備移到發送前。
這只有 CPU 來源與回歸證據，沒有新的 Native Timing 驗收。
先前沒有 LIGHT Prediction、超過5ms 的失敗 Shadow Run 仍是失敗，
不能重評為通過；原始完整 Archive 未在本機取得的部分僅引用使用者摘錄。

歷史實驗與本次環境、Baseline、47 Commits／15 Comments 的初始 API 稽核，
以及實測和假設的界線在 [Current-State Audit](current-state-audit.md)。
RTX4090、Qwen/Qwen3-4B-Instruct-2507、vLLM0.30.0、
Torch2.13.0+cu130、Python3.12 是原生計畫的版本條件，
不是本次 CPU 環境正在執行這些 Runtime 的宣告。

## 四模型研究確實獨立執行

| 原生 Polecat／Profile | Usage Record 的實際模型 | 三輪研究 | 最後狀態 |
|---|---|---|---|
| obsidian／copilot-opus | Claude Opus5.5 (`claude-opus-5.5`) | 獨立提案、交叉反例、修訂，共3份 | 已終止，保留註冊 |
| quartz／copilot-astra | GPT-6 Astra (`gpt-6-astra`) | 同上，共3份 | 已終止，保留註冊 |
| jasper／copilot-gemini | Gemini3.8 Flash (`gemini-3.8-flash`) | 同上，共3份 | 已終止，保留註冊 |
| onyx／copilot-gpt | GPT-6.1 Sol (`gpt-6.1-sol`) | 同上，共3份 | 已終止，保留註冊 |

並非只查 Profile 名稱或 Process argv。逐一讀取各自
`session.usage_record.data.usage.model`，116／54／97／67 筆均只出現預期模型。
四份 Round1 完成後才分享同儕報告；Round2 對抗、Round3 修訂，
不是以多數決認定科學結論。全部12份報告、Session ID、原生分支、
原始 Events 路徑與 SHA256、報告 SHA256 見
`results/mpc-research-cpu/agent-execution-ledger.json`。
原始 Session Events 不直接提交，避免不必要暴露工具內容。

Mayor 重現8項確定性反例，並重跑交叉批判的檢查。
撤回的說法包括：平均 Release 當成逐路徑界、無 KV 觀測卻聲稱 Chance Safety、
硬 Terminal Drain 必然可達、Stationary 就表示 Myopic 最佳、
把相關事件當獨立樣本、把 GET 當作 Late Write 的 Fence。
完整採納、否決與未決分歧見 [Debate and Consensus](debate-and-consensus.md)。
模型共識不是 GPU 證據。

## 選擇的架構與數學範圍

選擇**有限候選、共同情境、非線性事件／生命週期預測的 Gateway Admission MPC
Challenger**，而不是預先認定 Robust／Stochastic／Learning MPC 必勝。
理由是現有可用的動作、資料與安全驗證能力；不是已證效能最佳。

狀態包含帶時間／階段的 Outstanding Request Marks、完整 Reserved Cost、
Awaiting Prompt Backlog 與過去到達統計；每個狀態的可觀測性、
延遲／噪聲、因果用途與待做 Ablation 在 [System Model](system-model.md)。
KV Occupancy、真實 GPU 壓力、未知 EOS 與 Engine 進度沒有被假裝成已知。

控制輸入目前只有 Weighted Admission Budget `u ∈ {16,32}`；
這不是物理並行請求數16／32，也不是 Engine Token Budget B2048。
Gateway 不具備本次可動態使用的 Prefill/Decode Token 排程、
KV 配額或 Waiting Queue Capacity 介面；那些需要 Runtime/Scheduler 工作。

`f` 是 FIFO Prefill、Decode Interference、Terminal Release 的事件近似；
`h` 是同鎖、帶 Run/Clock/Cost/Model 身分與時效的 Gateway 觀測。
到達、Prompt/Output Mix、EOS、Service Scale 是擾動或不確定量。
現有係數與 Poisson/獨立 Marks 情境是合成假設，未經 Native 辨識。

每1s 決策，3個控制步、8個候選計畫、4個共同情境；
固定5s 評分 Cohort，另10s Tail，控制期後使用宣告的 Budget16。
主要目標是情境平均的「已解析、最終成功且 Scheduled TTFT≤0.5s」件數下界；
所有既有與評分窗內 Offered Requests 固定納入，拒絕、失敗、未解析不算成功。
因此不能靠全拒絕取得完美成績。Tie 優先 Hold、再少切換，不任意混合量綱權重。

初版並沒有完整多 SLO 的 TPOT/E2E 最適化與 GPU 利用率約束：
TPOT/E2E 是副指標，GPU Utilization 未知。後續權重與風險限制須依真實 SLO、
量綱、Pareto 與獨立 Calibration 設定，而不是從本次 Test 追分。
情境 Reward 上下界只是有限 Tail 的 Censoring 界，不是 Native 風險 Coverage。

Admission 不超過適用 Budget；降 Cap 不驅逐既有 Reservation。
Clock-first、完整身分、CAS、Expiry、Operation ID 在同鎖驗證；
Timeout 代表「是否已套用未知」。獨立 Opt-in Actuator 先持久化 Pending，
重啟後仍禁止新 Write，直到有效 CAS 前進或同時域 Expiry 證據；
單一 Fresh GET 不是 Fence。CLI 永遠 GET-only，Native 授權不存在。
沒有宣稱硬即時、Recursive Feasibility、Queue Stability、KV Chance Safety
或跨負載 SLO 保證。完整方程、限制與假設見
[Mathematical Specification](mpc-mathematical-specification.md) 與 [ADR](architecture-decision-record.md)。

## 實驗與驗證結果

| 證據層 | 已完成 | 不能推論 |
|---|---|---|
| 未改動 Baseline、Python3.12 | 1723 passed，1 native skip | GPU／HTTP 通過 |
| 最終完整回歸、Python3.12.16 | **1976 passed，0 failed，1 skipped，304.02s** | 原生控制效果 |
| Package | Wheel 七個 MPC Module 均與來源位元組一致 | 已完成 Native 部署資格 |
| 凍結合成 CPU Policy Benchmark | 5 Seeds ×3 Workloads ×5 Policies，75 Episodes | 真實 vLLM 效能／安全 |
| 新 Native GPU／真實 vLLM HTTP／Live MPC | **未執行、未驗收** | 任何效能優勢 |

唯一 Skip 是未安裝 vLLM 的 Installed Schema Case；另1項既有 TestClient
Deprecation Warning。整合時的 Metadata 與合成 Clock-grid 失敗均保留，
修正沒有放寬驗證器、100ms Deadline 或刪除不利資料。
詳見 [Implementation and Test Evidence](implementation-and-test-evidence.md)。

CPU 的開發集選 Fixed16、Feedback512/2048；Test10001–10005 事先凍結，
未使用 Test 調參。平均 Goodput：

| Policy | Good req/s | 平均 Rejection |
|---|---:|---:|
| Fixed16 | 5.5778 | 57.984% |
| Fixed32 | 2.3833 | 47.429% |
| Feedback | 3.8722 | 56.599% |
| One-move | 4.5444 | 58.312% |
| MPC | 4.4222 | 58.189% |

MPC 比 Fixed16 少1.1556 good req/s，約低20.7%；
5個 Seed Block 的探索性95% Bootstrap 區間為[-1.4167,-0.9667]。
對 One-move 為-0.1222，區間[-0.4000,+0.0333]，沒有證明多步優勢或等效。
雖比這個 Feedback 高0.5500，不能選擇性忽略更好的 Fixed16。
MPC 有10/180 Deadline Fallback；每 Episode 的 Controller P95 平均69.65ms，
One-move 為21.23ms。均非 GPU 數字或硬即時保證。
完整 Throughput、TTFT/TPOT/E2E、失敗、拒絕、振盪、Recovery 與統計限制見
[Benchmark Results](benchmark-results.md)；不混淆平均 Episode Quantile 與 Pooled Quantile。

## 已證、未證與下一個高資訊量實驗

已證的是來源回歸、合成生命週期／Reservation／傳輸失效處理、
目前 CPU 比較的負面結果，以及歷史 Development Trace 的 Factual Predictability。
歷史 Prompt Backlog 的高 AUC 不等於 Control Action 的因果效果；
成功 Output 皆128 的資料也不能辨識 Natural EOS。

未證的是 Native Action 因果模型、Residual-life 分布、Online Sensor Delay、
Prefill/Decode 雙向干擾、OOD Calibration、真實 HTTP 命令資格與 MPC 優勢。
最大的風險不是再選哪個 Solver，而是錯誤或不可辨識的 Intervention Dynamics、
Gateway/Backend Membership 不一致，以及未建模的傳輸／失敗路徑。

停止對這份 Held-out CPU Test 做排行榜調參。不宣稱 Fixed16 在真實4090
最優，也不宣稱所有 MPC 都無效。先保留簡單控制器與 One-move，
以隨機且受支持的 Budget 激發、變動 EOS／Workload、飽和恢復、
完整 Client Labeling 做因果辨識；嚴格分開 Train／Calibration／Validation／Test，
確認模型能在未見的 Action/Workload 上預測，再決定是否值得增加 MPC 複雜度。

建議的下一階段是**需另行授權的單台 RTX4090 Identification Pilot**：
最多2.5計費小時；48個約90s Development Episodes 僅在 Setup/Drain/剩餘預算
確實容納時執行。需要供應商報價與明確金額上限
`2.5×小時報價＋核准的儲存／網路／最低計費`，不假設舊租機仍可用。
Pilot 不保證足以完成最終 Confirmatory Benchmark。

具體提出**總額最高 US$5、最多2.5計費小時**的待授權 Pilot。
2026-10-30 查閱 Runpod 公開價目表，RTX4090 Pod 標示 US$0.89/小時，
GPU 本身2.5h 的參考值為 US$2.225；不是實際可租機器或全包報價。
儲存、網路、稅費、最低計費與最低儲值必須另查並納入核准總額；
超過 US$5 或未獲明確授權就不啟動。沒有選定、付費或啟動任何供應商。
公開來源與限制記錄在 `results/mpc-research-cpu/gpu-budget-reference.json`。

先驗 State／Action／Fallback／Single-worker 真 HTTP／Clock，才啟用控制；
若重用舊 Envelope Collector，仍須重過原來5ms Gate，不能放寬舊門檻。
Gateway-only MPC 不必等固定 Membership Envelope Oracle，應事前定義自己
的 HTTP／Offered-load 時序合約。完整計畫、停止條件與限制見
[Limitations and Next Experiment](limitations-and-next-experiment.md)。

**原生辨識與最終效能驗收仍被資源／授權 Gate 阻擋；Epic 不宣告完成。**
模型共識、CPU 綠燈與合成 Benchmark 都不能取代這個成功標準。

## 十項交付對照

| 交付 | 檔案 |
|---|---|
| Current-State Audit | `current-state-audit.md` |
| Mathematical System Model | `system-model.md`、`mpc-mathematical-specification.md` |
| Identification Report | `system-identification-report.md`、`system-identification-plan.md` |
| Competing Controllers | `competing-controller-comparison.md` |
| MPC Specification | `mpc-mathematical-specification.md` |
| Architecture Decision | `architecture-decision-record.md` |
| Debate and Consensus | `debate-and-consensus.md`、`proposals/*-round{1,2,3}.md` |
| Implementation/Test Evidence | `implementation-and-test-evidence.md`、`mpc_lab/`、`tests/test_mpc_*.py` |
| Benchmark Results | `benchmark-results.md`、`results/mpc-research-cpu/policy-benchmark.json` |
| Limitations/Future Work | `limitations-and-next-experiment.md`、`execution-incident.md` |

表中的研究文件相對於本報告目錄；程式碼與 Results 相對於 Repository Root。
「有報告」不表示尚未執行的原生實驗已完成。
