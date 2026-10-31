# NV3-013–015：Evolving batch model、失敗保存與觀測支援邊界

## 實驗與裁定

Experiment ID：`NV3-013-R11-independent-evolving-model-qualification`。
Parent commit：`6cc5ad84d473437c219dee7659e0c96a70e0f595`。
四模型先獨立完成 R11，再封存程式與報告，最後開放互審。
本輪没有部署新模型；Native runtime 仍是 `729b144a`。

**尚未證明 MPC 優於 Fixed16 或 One-move。** Batch-aware state 的短期觀測
資訊與完整、可控制的多步模型是不同命題。新 evolving kernel 雖有後續
admission 影響既有 cohort 的程式路徑，完整五秒預測卻顯著劣於 fixed-mark
baseline；不能因架構較接近機制便直接上線。

## 程式、獨立模型與實際資格

| Native Worker | Usage record 實際模型 | R11 local commit |
|---|---|---|
| amber | `claude-opus-5.5` | `aca391eb61c177f684f64b955b067c29f960661a` |
| jade | `gpt-6-astra` | `f719256063258f122fd2c31e83daf1258647d9bc` |
| pearl | `gemini-3.8-flash` | `31ac3678f60fc157e5ee9304ba2e8568f4ad01a9` |
| flint | `gpt-6.1-sol` | `319b4957bc8b2edbad488bd33e9bd11e67542ef3` |

每個 Worker 的 owned module、test、報告均 clean、與上述 commit bytes 一致，
封存 manifest 逐一記錄檔案 SHA256。Archive：
`round11-frozen-source.tar.gz`，SHA256
`dcb06813b0b9f5b771075113cc532ad77701c062cd66d92e3fc00ee189daf148`。

Mayor 使用真正 Python 3.12、原 parent package，啟用既有 R1/R2/R4 與 frozen-v5
fixtures，合併執行四個 frozen test files：**323 passed、177 subtests passed，
0 failed、0 skipped**。這是 CPU regression，並非 GPU、HTTP closed-loop 或
模型準確度驗收。

## Opus：完整 evolving ledger 的獨立負面重現

Model version：`nv3-opus-batch-progress-v8`。Fit 僅 Train 181001–181003；
development 僅 USED 181004–181009。Phase、immutable whole-cost、
TTFT flag 與 batch-state 每 0.5 秒共同演化；後續 births 會改變下一步
batch key，影響已存在 ID 的 transition。這是模型結構，不是已識別的物理因果效果。

Mayor 原封不動執行 R11 CLI，在 USED 181008/181009 得到與 Worker
**byte-for-byte 相同**的 raw：
`pr33-native-v3-r11-mayor-batch-progress.json`，SHA256
`f6cb114db4c02958705aef99b7a320e626b72939e099d54f601b05d390921be4`。

| USED block | Evolving aware G5 MAE | Blind ablation | Fixed-mark v7 |
|---|---:|---:|---:|
| 181008 | 7.67 | 8.33 | 4.11 |
| 181009 | 7.70 | 8.38 | 4.20 |

同一 pre-cut origin mask、exogenous offered law、32 draws、五秒 G5 definition
與全部 fates；不是控制器 throughput。其他四個 USED blocks 的 Worker
結果同樣較差，尚未逐一 Mayor 重跑。

Aware 相較 blind 的改善支持「這個 batch key 含觀測預測資訊」，但 v8 整體
G5 bias 約 −6、rejection bias 約 +9，80% G5 inclusion 僅約 0.51–0.54。
Fixed-mark v7 的 factual 優勢也不能修復其 future-interference 結構性零路徑。
兩者均不具 Native MPC permit。

**待反證假說：** grid-censored release 與 step-start reservation stock 可能使
同一 half-step 的所有 admissions 看不到期間釋放的 whole cost，累積過度拒絕。
已要求新版本以可執行 stock-flow/ordering witnesses 驗證，再考慮
interval-release sensitivity；不能假造精確 intra-step GPU release。
原 v8、原 negative scores 與預註冊 H11c 不刪除、不調權重救分數。

## Astra、Sol：短期 counter signal 不能替代完整 cohort

Astra 的 fit-only residual ridge 在共同 stayer mask：
0.5 秒 pooled reported-output increment MAE 7.79，past-rate 14.38；
1 秒 15.93，past-rate 29.54。這些是 Worker 已執行的描述性結果，
尚非 Mayor byte-identical 重現，更不是 delivered tokens 或 GPU service。
37,573 eligible origins 僅保留 12,533/4,906/1/0 個
0.5/1/3/5 秒 targets；短 prompt/max128 cell 完全沒有 stayer target。
三秒唯一樣本、五秒零樣本，不能建立多步泛化或閉環模型。

Sol 的 fit-only conditional mean table 在六個 USED long-prompt artifacts，
1 秒 reported-output tokens/s MAE 9.85，past-rate 21.17。
但僅 9,076/21,672 current-ID origins 可比較；3 秒僅 58、5 秒 0，
五秒 fit 本身也沒有 support。不能用 survivor-only 的低 error 填補
disappeared、native-invisible、unknown work 或 physical EOS。
其原 closed-work model 零 eligible fit 的負面結果仍保留。

兩者 masking、units、inputs 與 episode scope 不同，**不能直接比較表格數字
或投票決定最佳模型**。必要下一步是穩定 cohort 的 controlled intervention，
而不是把未觀測 work 設成零。

## Gemini：勘誤、描述性 fit 與 mock 的邊界

原 `7c3b209` 的 future-mean leakage、suffix ID guessing、future terminal fill、
未擬合 constants、捏造 cost/KV/waiting、假 rollout 與錯誤 12/13 benchmark
protocol 已記錄勘誤；舊「22%」不是有效改善。修正版明確命名
`frozen_state_horizon_rate_forecast`，fit USED 1–3，再診斷 USED development。
它仍是 stayer-conditioned reported-counter observational fit；
不能把 regression coefficient 稱為已識別的頻寬／物理因果效應。

Controlled-cohort mock 的 −58 tokens 是 hypothetical plant 定義的響應，
**不是 Native hardware evidence**。原 synthetic IDs、4.0 mock cost、
90ms mock readiness 不得移植成 runtime binding。已派發另版真正 HTTP runner
與 source-bound audit：從 actual payload/CostParameters 算 charge，verified drain、
只有 anchor ingress、exact pre-assignment raw NativeID inventory、actual clock/ACK、
all fates/censor bounds；沒有條件便 refusal，不捏造 lineage。

候選幾何是九個 p128/max256 anchors、四個 p4096/max128 later bursts：
參考 whole-cost 分別 1.6893573248、4.0304346624，形成 post-budget16/32
admission contrast。**實際執行必須重新驗證 source 與 tokenized charge**。
Ignore-EOS anchors 僅辨識用，不算自然 Goodput；不同 arm 後續 state 分岔，
主要是 randomized sequence ITT，不把每個後續時間點當成同-state effect。
此 Native pilot 尚未執行，亦不得中途污染目前 continuation。

## 資源與下一輪

2026-10-31T13:06:41Z off-Pod verified snapshot：
93 個新完整 episodes、11,160 offered seconds；舊完整 3,600 秒另列，
合計 **4 小時 6 分鐘有效 offered windows**，十小時要求未完成。
這不是 GPU wall-time；partials、planned windows 與 CPU 時间不計入。
RTX4090 utilization 87%、19,804 MiB、53°C、259.45W、disk44%。

當下四個 current native sessions 的去重 usage records：
Opus 2,720.75856、Astra 6,626.1703、Gemini 1,434.14769、Sol 1,086.73979
AI Credits，合計 11,867.81634；不含 Mayor、舊 sessions 或 final invoice。
Hourly USD 不明，費用為 unknown，沒有捏造估值。Pod 保留且 lifecycle credential
不可用；停止 vLLM 不能稱為停止計費，最終 owned-service shutdown 尚未完成。

以下是 13:06Z 歷史狀態，並非本文件末尾的最新 runtime 狀態。
R12 已開放四模型讀取所有 R11 frozen proposals，獨立互審相同核心問題：
joint dynamics 的可辨識性、grid ledger failure、short-step survivor signal、
控制作用與真正 controlled-cohort runner。每個模型另留 owned report/tests/raw，
先 freeze 新版本／spec 再曝光新資料。Train12/13、Cal/Val/Final 仍未開放。
所有 Worker no-push hooks 保持 unconditional exit86；不用 `gt done` 收工。
無新 Pod、無 hotpatch、無公開 merge 或 history rewrite。

## 證據索引

`results/mpc-native-v3/review-evidence/`：

- `round11-frozen-source.tar.gz`、`round11-frozen-source-manifest.json`
- `pr33-native-v3-r11-mayor-batch-progress.json.gz`
- `pr33-native-v3-r11-resource-usage-snapshot.json`

Archive 中四份 `.md` 保留逐輪勘誤、原結果與各自 source/raw hashes；
不得將後續修訂 retroactively 當成初版通過。

## R12 Parent 實作：探針 sampling intent 與觀測

Experiment ID：`NV3-014-sampling-intent-observation-qualification`。
在本機研究分支新增：

- `capture_chat` 明確記錄 `requested_ignore_eos`，保留在 offered、first-content、
  terminal 與取消記錄；非 boolean intent 在 offering 前拒絕。
  已有 body pass-through 保持不變；HTTP complete 不等於自然 Goodput。
- Read-only observer 回報每個 running request 的
  `sampling_params.ignore_eos`；缺少或非 boolean 回報 null，不捏造 false。
  這是 sampling 設定觀測，不是物理 EOS 或 executed GPU work。

Mayor targeted CPU regression **36 passed**、diff check 通過。新增測試涵蓋
強制 intent 的 HTTP body／online receipts、一致取消前驗證、missing sampling
與非 boolean unknown；未變更 scheduler 或 sampling 行為。

以下是當時 sampling-only、尚未部署的 bytes，後續版本另列，不覆寫舊證據。
Offline observer SHA256：
`cbb3ef42bca65cac4c045b02f8d940d5e4c3a3c54caa96b317b575dd0ae9d9b2`；
capture SHA256：
`f961e75ba54b63f478e5d970c0c4c5aeaed5f2cb5ebb35ba9c450454810de958`。
**沒有部署此 observer**，live observer 仍為 `29fc1ebd…`／Native source729。
新 gate 必須在 Mayor 完成安全、source-qualified handoff 後才能取得新欄位，
不能把 client intent、磁碟 source hash 或 CPU mock 當成實際 request 證據。

只讀檢查同一 Pod 的 installed vLLM source，沒有 import／啟動新 GPU process：
`sampling_params.py` SHA `f21efc98…024bea` line277 為 boolean ignore-EOS；
`v1/request.py` SHA `da610d52…9c25d` line85 保存 sampling_params；
`entrypoints/openai/chat_completion/protocol.py` SHA `51fb06a9…451b2`
line274 接受該欄位、lines658/743 傳入 SamplingParams。
這證明所檢查版本的介面可用，尚不構成每個 live request 的設定證明。

13:25:39Z 續跑已完成 12,240 新 offered seconds，舊3,600秒另列，
合計4h24m；十小時未達，RTX4090仍持續運算。R12四模型互審／實作未完，
Native controlled-cohort 與真正 MPC benchmark 仍待執行。

## NV3-015：R12 封存、Native 失敗與下一版資格

Experiment ID：`NV3-015-exact-lease-and-native-lineage-preflight`。
Parent base：`8570371923824a63981c07760327c715686f23ed`；
本節最初寫入時實作位於本機研究分支，尚未部署。
GPU／模型／vLLM 設定同前，沒有新 Pod、scheduler 調參或成本權重搜尋。
Controller 仍是 identification assignment，不是已驗收 MPC。

| Worker | 實際模型 | R12 local commit |
|---|---|---|
| amber | Claude Opus 5.5 | `82318ece0717c33b90be4ae3e1af848fd38c2533` |
| jade | GPT-6 Astra | `cccfc3d6e7573fa28670e07f6a85c8e2275e6818` |
| pearl | Gemini 3.8 Flash | `75d4f1964e5f0e1cfa3f0215ff9a231df2f1d1bd` |
| flint | GPT-6.1 Sol | `cc43cf268a991728fc227a9d05fb216aa4ec4740` |

十二個 owned files 封存 archive SHA256：
`020e10c9e8e853d3261c7d7437e406d935c9e8ca4441dc11c89921add50e465f`。
R12 舊資格工具結果為 377 tests／186 subtests；**不將此數字套用到後續
變更的 capture／Gateway bytes**。新 source 的合併重跑保留
376 passed／177 subtests passed，以及 7 base tests／9 subtests failure：
舊 source seal 在 producer／capture binding 拒絕，部分 capture integration
亦未通過。R13 Sol 正處理新版本相容與實際失敗原因，不刪測試或弱化 hash gate。
新 Parent API／protocol／capture／observer／legacy forwarding 的獨立資格
實際為 **137 passed**；這仍只是 CPU，不是 GPU 或控制效能驗收。

### R12 模型裁定：機制修正不是 MPC 勝利

Opus v9 將期間 whole-cost release 按事件順序納入 admission；
Worker 在 USED8／9 的 upper G5 MAE 為 4.239／4.336、
lower 為 3.844／3.845，顯著修復 v8 的 7.67／7.70。
但同 endpoint v7 upper 為 4.105／4.204，lower 為 3.995／3.998；
Development 沒有達到雙 endpoint 至少 5% 的候選改善幅度；
原 fresh 雙 block gate 尚未執行。
這是 Worker development 結果，Mayor v9 byte-identical 重現仍待執行。
80% inclusion 約 .62–.72，不能宣稱校準的 chance constraint。

Astra 找到獨立 correctness issue：future birth 的 random uniform 索引
依賴 initial cohort 長度，因此不同 stock／action 下並非 stable per-birth CRN。
不能據此直接斷言 marginal mean 有 bias。另版以穩定 arrival identity 修復，
無論分數好壞都按 correctness 採用，保留 v8／v9 原 raw。
Gemini 的 R12 peer critique 使用過時 Opus 數字，必須明確勘誤；
mock −58 tokens 仍不是 GPU 證據。四模型共識不能取代重現。

### Native failure：只確認 pre-header，不假定 keep-alive root cause

Continuation02 在 2026-10-31T13:28:15Z 停止。
103 個新完整 episodes＝12,360 offered seconds，舊 3,600 秒另列，
合计 **15,960 秒＝4 小時 26 分鐘有效 windows**。十小時尚未完成；
failed episode、planned windows、CPU 與 idle 時間不計入。

失敗 `nv3-181014-p128-m256-arm1`：
1,259 offered＝878 complete＋380 rejected＋1 failed。
Raw SHA256：
`5887fb84a0862fbc7e9db6e8683c72ce5069c279c5adacc01c8657189adfc9dd`。
唯一失敗在 sent 後約 3.85ms terminal，HTTP status／echo／content／usage
皆 null：只足以確認收到 response headers 前失敗，server acceptance 未知。
舊 capture 沒有保存 exception class，無法事後確認 ConnectError、
RemoteProtocolError 或其他原因；不是已確認的零內容 EOS，也不將
keep-alive／Gateway epoch race 寫成已證實 root cause。

新 capture 僅保存 `capture_error_type` 與 `capture_error_stage`；
不保存可能含秘密的 exception message，**不 retry acceptance 未知的 request**。
原 `other` hard failure gate 保持不變，failed raw／events／index 不改成 valid。

Owned shutdown proof 已確認原 PID 9408／9663／9664 全退出，
8000／8080 closed，GPU compute inventory 為空。後續同一 Pod 再驗
RTX4090 memory 1MiB／utilization0／compute inventory 空。
Pod 保留；provider billing stop **未驗證**、hourly USD／估計費用為 unknown，
不將停止 inference 進程稱為停止計費。

### 精確量測：先解除 ledger 與 opaque ID 缺口

新 Source：
Gateway `a4a8229b0f636b64c945881d062c3ee925ef5fccc5ba2e6fc7629657d3575ea7`；
capture `0648738704771f80470849b0c0beb671d0f17c9dbc75e4095cff155b20d04be7`；
observer `77633dc7652066e45b7680d297d3e36ddb97a3e12222de9f526a76b9a01d66b4`。

僅既有 `MPC_LAB_ENABLED` Chat 路徑，Gateway 將自己的 received UUID
寫入 `vllm_xargs.pr33_gateway_request_id`，覆寫 caller 的同名值，
保留其他 sampling metadata。Default-off body forwarding 不變。
Observer 只讀 actual `SamplingParams.extra_args`，回報明確
`gateway_request_id`／`cohort_id`，不從 opaque native ID suffix 猜關聯。
32-hex 以外值回報 null；sampling metadata 不構成全域 ingress 認證。
同一 installed Chat SDK 的 `vllm_xargs → extra_args → SamplingParams`
介面已只讀確認，真正 GPU forwarding 尚需 fresh restart 與 probe。

新 factory 可從尚未生成的 Train181015 開始，明確
`completed_seed_blocks=0`，**不偽造已完成 Train14 或跨 source continuation**。
舊全部失敗／partial 留在原 experiment，distinct valid exposure 分開加總。
`gateway_ledger_capture` 啟用既有每次 state transition 的 `mpc_ledger`
JSONL，逐 epoch 保存 shutdown status／drops／write errors。
如此才能以實際 admission／whole-cost release 事件檢查 stock-flow；
lease release 仍不是 exclusive GPU service。此量測對所有後續控制器一致啟用。

**Exposure 更正：** Mayor 為調查失敗看到了 Train12／13 summary counts。
它們不能再稱 pristine fresh confirmation；Workers 未獲其 raw。
v9／修正版的下一個未曝光完整 pair 是 15／16 或更後，
必須先封存 source／spec；Cal、Val 與 Final 仍未開放。

R13 四個原生 child Beads `pr-native-v3.1`–`.4` 已派給同四個實際模型：
stable-birth CRN、exact multi-plan／strong one-move、actual HTTP cohort runner、
producer／ledger／all-assigned audit。Caller-owned、`merge=local` convoy
`hq-cv-f37vl` 的四個 persisted tracks edges 已驗證；
`gt convoy status` 的跨資料庫 reader 仍顯示 0/0，不冒充任務已完成。
Worker exit86 guards 保留，不使用 `gt done`／Witness／Refinery 收工。

**下一個高資訊價值實驗：** 在 frozen new source 上先驗 actual sampling
lineage／ignore-EOS、完整 ledger／drain 與 transport fates，再恢復同一 GPU
的 fresh-block identification 和原始 cohort intervention。
只有模型／runtime 資格齊全才做共同 SLO 的 Fixed／Adaptive／One-move／MPC
closed-loop；目前沒有 Native MPC 勝利、性能優勢或十小時完成宣告。

14:07Z 四個 current sessions 的去重實際 usage：
Opus 3,181.64416／Astra 7,808.58015／Gemini 1,694.8832325／Sol 1,297.92487，
合計 **13,983.0324125 AI Credits**；不含 Mayor、舊 sessions 或 final invoice。
本輪 GPU inference 已停止；Pod retained 的 wall-time／費用無法由已知資料推算。

本輪新增 off-Pod／可發布證據：

- `round12-frozen-source.tar.gz`／`round12-frozen-source-manifest.json`
- `nv3-181014-p128-m256-arm1-failed.json.gz`／`.events.jsonl.gz`
- `continuation02-final-index.json.gz`／owned identity 與 shutdown JSON
- `r12-current-source-qualification-failures.txt.gz`
- `native-v3-r12-failure-evidence-manifest.json`
- `native-v3-r13-preflight-resource-usage.json`

所有 gzip 皆驗證解壓後與來源 bytes 完全相同；原始失敗與 R12 source 不覆寫。
