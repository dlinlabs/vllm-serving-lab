# Native v2：技術裁定、數學邊界與下一個實驗

## 1. 裁定摘要

**尚無 MPC 勝過 Fixed16、強 One-move 或合理 reactive baseline 的真實 GPU 證據。**
這不是「所有 GPU／模型上 MPC 都沒有勝率」的證明。
本輪完成真正四模型、三輪獨立提案／互審／修訂，以及可執行的量測資格工具；
沒有把 CPU fixtures、模型共識或既有 profiling 當成新 Native 性能。

暫選研究方向是**支援域內、history-conditioned 的 reservation release／good-count
機率模型**，並行用 batch／commit labels 解釋誤差；不再把 HTTP residence
當獨占 GPU service，也不先建造高維 kernel digital twin。
這是下一個辨識候選，**不是已辨識完成的控制器或已證最優架構**。

既有資源的恢復 mutation 被供應商拒絕；之後 read-only 確認
`i243ycb94oaxy2` 為 EXITED、runtime=null，原 RTX4090 machine 的
`gpuAvailable=0`。可用數為0與拒絕相容，但不足以排除其他拒絕原因。
沒有盲重送、另租機器或修改憑證／remote；GPU 相關工作暫停，
而非用更多 CPU 調參取代必要實測。

## 2. Ground truth 與可追溯證據

| 項目 | 已確認範圍 |
|---|---|
| PR32 | HEAD `8e92371f342be98a50682bf4e2ddbb014665498b`；merge `765166081d0af5499c63ecff53b257f91261d3de`；base 是 `analysis/cap-miss-causes-v1`，不是 main |
| PR33 | draft、同一 base；本輪發布前查得 head `0588707949f651039d44a7a034c12bd35b8c8047`；後續發布以 GitHub／remote 實際 receipt 為準 |
| 舊 Native 資料 | 13792 intervention offers、32 isolated requests，另22個刻意 cap-stop 的 profiler probes；三種族群不可混算 |
| 舊 GPU 環境 | RTX4090 24564MiB、Qwen3-4B-Instruct-2507 BF16、revision `cdbee75f17c01a7cc42f958dc650907174af0554`、vLLM0.30.0、Torch2.13.0+cu130、Python3.12.3；不是本輪新執行 |
| Serial 反證 | 特定 exclusive HTTP-demand 解讀預測90秒至多2004，實際 drained-start arm 完成2449；不反證所有 fluid／queueing／MPC |
| 舊 synthetic CPU | Fixed16 5.5778、MPC 4.4222 good/s，約低20.7%；不是 GPU 勝負，也不能只怪 solver |
| 四模型真實性 | usage backend 逐筆驗證 Opus5.5／GPT-6 Astra／Gemini3.8 Flash／GPT-6.1 Sol；source commits、prefix SHA 與 accounting ledger 保留 |

歷史原始依據見 `../native-identification-and-model-falsification.zh-TW.md`、
`../native-gpu-qualification-and-scout.zh-TW.md`；本輪見
`results/mpc-native-v2/round{1,2,3}-freeze.json`、各輪原稿與 Mayor adjudication。
產品 implementation hash 仍為
`b4f234020105f69f608da45dba7317a8bb632ecd84aeb0e310362740e5fa083a`；
目前改動為獨立研究／資格工具，不是上線 MPC。

## 3. 多模型裁定：保留證據，不製造共識

| 主張 | 裁定及理由 |
|---|---|
| 降 cap 後必須 `R <= current_cap` | 錯；舊 reservation 不被驅逐。應檢查 exact admit/release 與歷史最大 cap |
| 長 physical memory 必然使 multi-step 勝出 | 錯；binding、可改變的決策、continuation、模型與資訊都重要，已有可執行反例 |
| 兩個固定 cap 整段成績相同，因此切換策略無價值 | 錯；異質時段與 switching 可被平均掩蓋 |
| `<5%` first-action disagreement 足以判定提升很小 | 錯；必須同時量 binding、受影響 offered mass 與每次差異的後果 |
| NNLS mean busy law 是 Native capacity lower envelope | 未證；通過必要條件不代表可排程、可達容量或 SLO |
| snapshot delta-output 是精確 GPU iteration clock | 未資格化；可作 effective commit-rate proxy，須另外核對 participation／async／step |
| Astra unresolved-tail intervals 導致 under-admission | 不是對實際原型的反例；它是會計 diagnostic，不是 admission policy |
| Gemini R2 4.5／450／1200µs 與零 jitter | 明確撤回；只是 CPU fixture 常數，沒有 GPU 測量 |
| Gemini R3 已修好 superiority evaluator | 仍駁回：同一 block 複製六次仍宣稱 superior；0 output token／NaN origin 仍被記 good，Mayor 已重現 |
| 混合 workload 的 short TTFT 與 long-prefill backlog 無關 | 未證；舊 homogeneous p128 不能證明 short-behind-long 行為 |

最後兩項的 CPU receipt 在 `round3-mayor-adjudication.json`。
所有原型／原稿保留且標示未接受；不把 passing tests 等同其定理或統計認證。
Gemini 修訂中的「不顯著便證明 Native MPC 不可行」、弱 One-move、
未辨識 KV safety 與重複0.5good/s門檻也不採用。

## 4. 最小數學模型與不可假裝已知的部分

令 `H_k` 為決策截止前**實際收到**的、有來源／clock／epoch 的歷史。
`x_k` 不是宣稱完全可觀測的 GPU Markov state，而是：

- exact marked reservation ledger：每筆原始 cost、admit／first／release 狀態；
- `R_k`、已套用 cap／version／pending command；
- 可用的 prompt mix、admitted-not-first age bins、decode／progress bins、
  arrival history，以及 missing／unknown-work indicators；
- 經 ablation 才保留額外 Native features；drain 後 CUDA spans 只能是 labels。

目前可控 `u_k in {16,32}` 是 Gateway **cost cap**，不是 concurrency、
KV bytes 或 scheduler-token budget。內部 prefill/decode scheduling 不是現有
Gateway actuator；更廣 cap grid 也尚未 Native-qualified。

精確守恆部分：

```
R_next = R_k + sum(original_cost of actual admissions)
             - sum(original_cost of actual terminal reservation releases)
admit iff R + original_cost <= actually_applied_cap
R(t) <= max(R_initial, sup_applied_cap_so_far)
```

使用 serialized projected check/add，所以沒有任意 `+c_max` overshoot allowance。
cap 減少不撤銷舊請求；client terminal、engine EOS、reservation cleanup
不能不經核對就視為同一時間。GPU／HTTP clock 不可直接相減。

未知部分暫寫成：

```
(release trajectory, complete-good counts) ~ P_theta(. | H_k, assigned cap sequence,
                                                   arrival law, deployment support)
```

`w` 含未來 arrival/mix、EOS、GPU／host jitter。沒有將 frozen experiment tape
偷當線上可知未來。Randomized sequence 識別的是宣告 workload 下的 policy ITT；
不自動識別 mediator causal slope 或每個新 rolling state 的反事實 transition。
若四個 sequence cells 都不足，退到 action-only estimates，不拼接偽動態。

MPC 候選的目標是同一 offered cohort 的**預期完整 SLO-good completions**；
reject/error/cancel 為0，未結束保留 interval。共同 TTFT≤0.5s、
request-mean TPOT≤0.05s，one-token TPOT=0；SSE chunk gaps 不是 token P99。
沿用既有 `native_observer/audit_interventions.py:14–18` 的標記：
TTFT=`first_content_at-scheduled_at`，request-mean TPOT 在 output>1 時為
`(terminal_at-first_content_at)/(output_tokens-1)`，不是偷偷改用較短的
`last_content_at` 邊界。Output 必須為正整數、時間有限且同 clock；
完整 natural-EOS、錯誤及 censoring 的分類與原始 rows 一起保留。
先以支援域、actuation fence／expiry、觀測完整性及共同比較約束篩可行動作，
再最大化 good count；不靠任意混合 weights 或拒絕全部獲得漂亮 latency。
未校準 joint uncertainty 前不宣稱 chance-constraint／KV hard safety。

`D=5s`、兩個1s可變 slot 是**起始辨識設計**，不是已獲證明的最優 horizon。
Strong One-move 使用同一 window、arrival law、uncertainty、tail，
continuation 必須從 development 的 hold／fixed／reactive 選定；
不是只計第一秒，也不能刻意用差的 Fixed16 backup。實際都同頻重規劃。
需要實測 full-path observation→solve→persist→POST→fence latency，而非只計 solve。

## 5. 模型結構比較與主要辨識指標

| 結構 | 合理用途／目前限制 |
|---|---|
| Queueing／fluid | 合理低維 baseline；不能只用 total tokens 決定 whole-cost release |
| Batch-aware state space | 有物理解釋力；batch／commit scope 與線上 availability 尚須資格化 |
| Hybrid／semi-Markov | 可以表達 prefill/decode／EOS；目前不能假裝所有 hidden mode 已可辨識 |
| Learned direct transition／response | 暫優先小型 stratified／regularized predictor；須有 randomized support、block validation 與 OOD abstention |
| Piecewise nonlinear／hinge | profiler descriptive fit 值得檢驗；knee、未開 profiler 的係數和下界未證 |
| Probabilistic service／release | 能保留 EOS／release ambiguity；joint residuals、censoring、policy dependence 尚待新資料 |

先固定 origin 發布1／3／5秒 forecast，再比較 one-step／multi-step；
不能輸入 realized future gaps、未來 output length 或事後重建 state。
主要預測 target 為「origin 已 live cohort 在各 offset 累計釋放的原始 cost」。
以三個 offset 的 normalized absolute error 等權平均：
`mean_h |predicted_release(h)-actual_release(h)| / R_origin`；
`R_origin=0` 單獨報告，不以零工作樣本灌低誤差。
另報 first／terminal good-count、TTFT deadline Brier／calibration、
coverage、action ranking 與 OOD；forecast MAE 改善不是 MPC goodput 改善。
Baseline selection／feature selection 僅用 Train，validation 與 final 分開。
只有兩個 pilot seed blocks 時不宣稱10% MAE提升已泛化或具有統計保證。

## 6. 先執行的兩個有界關卡

**Gate A：已實作的 join／drain smoke。** Frozen16 arms：
TRAIN170001／170002、p128／4096、cohort1／4、enabled OFF／light、fixed32。
標準 Chat、natural EOS、max128，完整保留 startup／length／失敗。
原始回覆與 capture fsync／核 SHA 後 ACK；
collector 再核 exact offered/native ID sets、usage、natural-output work audit，
`forced_outputs=False`，LIGHT 才要求 step evidence。無效 raw 不丟。
Metadata 不得在 active／未ACK capture 時執行重型探測。
這不是 literal-disabled overhead、calibration 或 MPC 驗收。
原 source `ab8c392` 的 plan-v1 因恢復被拒**未執行**，不可寫成有16次 GPU結果。

**Gate B：只有 Gate A 成立後才執行。** 兩個 fresh independent TRAIN blocks，
每 block 的同一12s mixed short／long tape，四個 LL／LH／HL／HH cap sequences；
2s／3s 下令、4s 後共同 continuation。Literal disabled／enabled OFF／LIGHT
分乾淨 service epochs；每 sequence 再有一個 LIGHT 同條件重複，共32 episodes。
順序隨機且跨 block 平衡；全部實際 assignment／application／fence 保留。
每 arm 最多30s drain，overflow／pending／missing work 立即停止物理擬合。

此 pilot 的目的：observer effect、same-action EOS repeatability、switching
effect、固定 origin release/count predictability。不是五控制器決賽；
兩 blocks 不能證 equivalence 或 superior，也不能把 repeats 當獨立 seeds。
後續才擴 independent blocks、校準誤差／共同 continuation，
並資格化較完整 cap grid；只看16／32不能聲稱找到全域 best fixed cap。

成本以目前 US$0.89/hr quote：
Gate A 約5–10分鐘含已備妥環境的啟動，約US$0.07–0.15；
Gate B offered+最大drain=22.4分鐘，另四次 matched startup/warmup 若各2–5分鐘，
約30–42分鐘、US$0.45–0.63。這些是**有條件估計，不是預算上限或 invoice**；
cache／環境恢復未知時重新記錄，不讓無需求 GPU 等待 CPU 討論。
每輪封存並核 remote/local bytes，再確認 provider EXITED，而不只 kill vLLM。

## 7. 公平閉環與尚未達成的驗收

Fixed16、development-chosen best-supported-fixed、reactive、
strong One-move、MPC 必須同 GPU／模型／software／trace／offered load／warmup、
window／SLO與非控制 scheduler settings。共同報 goodput、accept/reject/failure、
TTFT／request-mean TPOT／E2E quantiles、KV／NVML、solver／full-path cost、
deadline fallback、oscillation與recovery。不得只展現 conditional accepted latency。

Stationary、burst、prefill-heavy、decode-heavy、mixed、shift、
saturation/recovery、unseen mix 均需覆蓋；不能先假設舊同質資料能預測混合干擾。
最終 primary estimand、最小實用 effect、block randomization、信賴區間、
rejection／安全 guard與多重比較處理，要在看新的 final outcomes 前另行凍結。
四個舊未見 Native seeds 的 exact two-sided floor 是0.125，不能假稱足夠；
至少需更多 fresh independent blocks，數量由 development variance／precision 決定，
不是達到六個便自動有 power。最終成果不足時保留「未決」，不是等效或永遠不可行。

**仍未完成：** 新 Native source／Chat／worker join 資格、新 GPU dynamics calibration、
one/multi/OOD forecasts、真閉環 MPC、fresh held-out優勢與部署驗收。
當前最高優先級是恢復可使用的既有資源後執行 Gate A／B，
而不是繼續擴 horizon、調 weights、借舊 theta 當 prefill duration，或宣稱 MPC 已勝出。
