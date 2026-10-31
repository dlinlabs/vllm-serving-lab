# NV3-013–015：Evolving batch model、失敗保存與觀測支援邊界

## NV3-020：R16 written seals與實際 HTTP 邊界

四個實際原生模型 R16 base sources 封存為12檔；Gemini actual-schema
follow-up `5aecb10c2bd2af543a3e0ad19ceec9cac4f5cc77` 另封存3檔，
沒有覆寫原 R15／R16 negative fixtures。Mayor 的主資格命令實際為
457 passed／252 subtests passed／4 skipped，134.78秒。
4 skipped 全是缺少 `NV3_R15_FREEZE`／`NV3_JOURNAL_REPO` 的
`R16JournalTests`；補足 exact R15 與 historical a4ec 後，僅執行該4項，
4 passed、0 skipped、0.27秒。故461項必要測試均已執行通過；
保留兩份原輸出，不將它們改寫成另一個「單次零 skip」receipt。

| Model | R16 local seal | Mayor scope |
|---|---|---|
| Opus／amber | `90b852058598f247a74fcaa53edfdfa3c8779b5f` | J1／J2、owner support、H16 Q1–Q5 |
| Astra／jade | `454c374a3b379fc2d9b59aedfc5d8eb52979d5a9` | offline labels ≠ precut state、actual observer gate |
| Sol／flint | `719d79a20c8f30230e5603145cbb00086c66de92` | actual a4ec journal、actual eight-field/max512 reference |
| Gemini／pearl | base `7212b3828c34cdf477a240dc38534694468cfdd8`; follow-up `5aecb10c2bd2af543a3e0ad19ceec9cac4f5cc77` | historical source refusal、no-phase／UUID／actual charged geometry repair |

### 接受新 owner label，但拒絕 online-state 升級

Opus 的 J1 是真實 source hole：telemetry disabled 仍會增加
`mpc_sequence`；final flush swap 後抵達的 cleanup rows 可能留在新 buffer，
原 status 卻能 closed／remaining0／SHA 對應短檔。J2 是 `mpc_mark=None`
仍可能接受 ID-less admission。Verifier 能拒絕缺失，不等於 producer 已修復。
Mayor 在另一個 source candidate 本機修正，commit
`d1bd1e5f6262eda1ee29867689a810a475797c73`，Gateway SHA256
`23aef8b8b903eda99ea325fd8ece09d998c174df39f2d1230ed5b9cc37e4ce3a`，
119項 CPU regression 通過，0.87秒。關閉時阻止新 admissions／control，
再次 flush late rows；seal 記錄 written final sequence／row count、
producer final sequence／unflushed count，對 gap／未寫入／残留 fail closed，
拒絕 NoneMark／writer-off admission。Feature 仍 default-off。

新候選位於 `mpc-native-owner-written-journal`。Historical
`mpc-native-owner-journal` 是 detached a4ec，Gateway SHA
`21e3b335450a141e0c9c84d15a5f6c46850f9bb219b296a1f7b9929c6caa81f7`，
供 frozen R16 tests 使用；不是新 source alias，沒有修改 sealed tests
或 Source144 parent bytes。119項是新候選 CPU receipt，不取代舊116項、
J1 failure 或原 source refusal。沒有 product deployment／hotpatch。

Astra 以相同完美 offline seal，重現 producer cut 的 owner cost20、
consumer 在較早 received watermark 卻仍為 empty；final seal 不能
回溯證明 pre-cut 已收到 rows。Sol 的自洽 forged UUID＋reseal 也能
通過 conditional offline integrity，但 authentication／fsync／online-cut
flags 保持 false。精確 owner lifetime label 與可用於線上 MPC 的狀態
是不同命題；尚無任何 Native／MPC permit。

### 真實 observer schema修復，不是因果窗口資格

Source144 observer 只有實際8個 request fields，沒有 `phase`。
原 R15 gate 依賴 fictitious phase，所以有效實際 frames 會被拒絕；
mock 加 phase 才會通過。Gemini follow-up 移除此條件，使用
actual prompt／computed／positive output／qualified max512／True EOS、
來源與 domain／freshness／exact client↔Gateway UUID bijection，
並按實際 tokenized Gateway charges 預檢 geometry；112項 owned CPU tests
也已納入 Mayor qualification。保留原 frozen failure witness，
不將舊 R14 serial-dispatch numerical faults說成 R16 hardware execution。

5aec follow-up 仍用 consumer `self.clock()` 作 `t_apply`，
gate counters 作窗口起點、survivor mean 作效果點估計、自動 approved
一小時 permit、receipt-relative sleep 作 burst offset。Server apply10、
receipt10.2 會把目標推到11.2而非11；missing owners 的 survivor mean
不能代替 all-assigned interval。R17 四模型正在 bounded CPU 實作／
反證 genuine ACK／domain／explicit permit／immutable apply+1秒、
counter brackets／received-prefix uncertainty／強 count16 baseline。
CPU schema 與來源修復不等於已執行 Native causal pilot。

### 分開 fresh reservations與資源

正式 reservation JSON：
`prospective-h14-h16-source-epoch-reservations.json`。
H14 保留 Source144 natural-workload Train181041／181042；
H16 Q1–Q5 measurement-only 用不同的 Train181043／181044，
只在新 owner source／core／protocol 完整 qualification 與來源封存後
才可開始。原 driver 仍獨占 backend，不注入 cohort；forced ignoreEOS
pilot 不代替 natural H14。USED15／16、Cal17–36、Val37–40 不重標、
不回灌 forecast 或挑有利端點。H11c fail／H12 unidentified interval／
H13 NOT_CANDIDATE 仍保留，下一個 exact-release forecast 要另預註冊。

16:58:49Z finalized ledger 核對新70完整 episodes／140 artifacts，
8,400秒＋舊15,960秒＝24,360有效秒（6h46m）。僅檢查 metadata／
artifact ledger，沒有讀取 Cal／Val outcomes。Ten-hour notifier／
acquisition／owned shutdown monitor 持續；十小時與 Native MPC
superiority 未完成。GPU USD hourly price／provider billing stop 未核實，
不把停止 owned inference 說成停止 Pod 計費。

17:00:01Z 四個 current native sessions 的 deduplicated usage：
Opus4,257.4581、Astra9,720.1067、Gemini2,458.62366、Sol1,744.91716，
合計18,181.10562 AI credits；不含 Mayor／舊 sessions／帳單，不能當作
整體費用或 USD。實際 usage models 仍分別是 `claude-opus-5.5`、
`gpt-6-astra`、`gemini-3.8-flash`、`gpt-6.1-sol`。
完整 source archive、兩份資格輸出、三份 canonical raw、119項 receipt、
local-only J1／J2 patch 與資源 snapshots 由
`results/mpc-native-v3/review-evidence/round16-mayor-required-coverage-and-evidence-manifest.json`
列出 bytes／SHA；gzip evidence 解壓後保持原 bytes，不 strip failure logs。
R17 caller-owned／merge-local convoy 是 `hq-cv-ll4h5`；R16 local handoffs
已 close，但不代表 Native acceptance、`gt done`、Refinery merge 或 GPU shutdown。

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

## NV3-016：R13 實際 Native 量測與 R14 資格反證

### 假設、版本與已確認結果

本輪檢查 explicit sampling metadata 能否在真正 vLLM／Gateway 接上
client receipt，並確認舊 CPU qualifiers 不會被誤認為可部署的 Native
intervention。Serving source 保持
`144cef32a9b51c3a74355ff67f5eafdfac6e075f`，implementation
`ed8a6a537b7b663f44f4d6ed42d9679fe88048b4077c25dd778b6e8cf418a014`，
model `native-id-unfitted-v1`／calibration `unfitted-not-validated`，
controller `prespecified-cap-sequence-v2-confirmed-holds`；
calibrated=False、mpc_active=False。沒有修改正在執行的 source。

R13 四模型十二個 owned files 的 byte-exact archive SHA256 為
`2d4307670e627730784f607cff2c72c7ba168aff1f07fd61c9b1f798ad9dace2`，
在 identification03 開始前封存。Mayor 以 Python3.12、完整歷史
fixtures、四 frozen modules＋current capture／observer 執行：
**398 passed、196 subtests passed、零 skip，39.87秒**。
第一次保存命令誤拼 test filename，零測試執行，未算通過；
訂正命令後的完整 output 另存，不覆寫已發布 R12 真實 regression failures。

Mayor 另以同 frozen R13 source、PYTHONHASHSEED=7、Train1–3 fit 與
USED8／9，獨立執行 `--replica-stable-crn ... 181008,181009`。
完整81,045bytes JSON SHA256
`b5703b1a652090271ad327fafc73a3e80cb899cc22f528739221b28f71cfd779`
與 Opus 原報告 **byte-for-byte 一致**，不是只比摘要數字。
Stable per-birth CRN 是 correctness 修正，development full-cohort
預測仍未通過共同門檻，不能當成 MPC victory；fresh15／16 尚待完整。

真實 GPU／HTTP smoke 的硬體與設定沒有變更：同一 RTX4090，
Qwen/Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`，vLLM0.30.0，
BF16／TP1／maxseq32／batchtokens2048／memory0.8／prefix cache OFF，
既有 batch hooks enabled、observer capture OFF，並非 pristine hooks-disabled。

| 實際觀測 | 結果 | 能支持的結論 |
|---|---:|---|
| offered window | 48秒／四 episodes | observation qualification，不是公平控制器 benchmark |
| HTTP fates | 922＝649 complete＋273 rejected，零 other | 所有 offered fates 有紀錄 |
| Native observations | 3,346 explicit Gateway tag matches | 649 unique accepted IDs 全部觀測到，沒有猜 opaque native ID suffix |
| actual SamplingParams.ignore_eos | 3,346筆皆 False | False 路徑已驗證；True 仍未測 |
| aggregate MPC ledger | 1,696＋267＝1,963 rows | retained bytes 可重現；不能推出 per-UUID lease ownership |
| Gateway reported status | 兩 epochs closed／dropped0／write_errors0 | reported clean closure，不是 authenticated complete suffix 或 durable ACK |

四 episodes 的 initial／final drain 皆通過；smoke 全649 complete
為 natural API-stop labels，但未做 MPC／Fixed16／One-move 比較。
21個 public smoke files 以 tar SHA256
`cdcb249b397c9bab2c11a8beeb05dc360eaa9444cc1db872319f824dc4a8eafa`
off-Pod 保存，每個 member SHA256 獨立重驗。Private control 未封存。

### Source 與實驗資格反證

Sol 的 R13 截 suffix counterexample 仍成立：closed status 沒有 final
sequence／row count／JSONL digest binding。Generic aggregate ledger 沒有
event UUID，不能以 adjacency／FIFO 宣稱 exact owner；flush 不是 fsync。
1,963＝3×649＋16 applied commands 的 count agreement 是待檢查 witness，
不是本文已證明的 full-run closure。R14 將檢查獨立 final Snapshot.sequence
與 all-assigned fates；若 raw 沒有 final watermark，保留 prefix-only，
下一 source 才加入相應 producer contract，絕不 hotpatch Source144。

Mayor 直接查 R13 Gemini source 發現仍有部署阻礙：
`RealNativeCohortRunner` 使用 transport.advance_clock／virtual contention
plant、非實際 `/native/state` API、sequential client.post，
以 content="prompt" 加 fictitious prompt_tokens 代替實際 tokenizer
qualification，並未接真正 GuardedActuator／standard Chat model payload。
Plant 更令 ignore_eos=True request 超過 max_tokens 後不終止；
**ignore_eos 只忽略 EOS，不能忽略 max_tokens**。
它要求 sampling Gateway UUID 不在 client IDs 也不符 Source144：
Gateway 覆寫 body spoof tag，但使用 validated received benchmark UUID，
實測 join 應相等。Pipeline unconditional provenance／ITT／censor flags
和 hardcoded「四 burst 全拒／全收」也不能用作 live proof。
這些是 CPU mock／部署資格錯誤；原報告 CPU-only scope 與原 source
保留，沒有把 mock −58tokens 當成 GPU performance。

R13 revised5×128/max512 與舊9×128/max256 anchor geometry 尚未取得
實際 lifetime／survival 證據，也沒有四模型共識。Source arithmetic
不能證明4.69秒存活；forced samples 不作自然 Goodput。

### 下一個有資訊價值的實驗與資源

新原生 Beads `pr-native-v3.5`～`.8` 與 caller-owned／merge=local convoy
`hq-cv-pkmce` 已派給 amber／jade／pearl／flint，實際模型依序為
claude-opus-5.5／gpt-6-astra／gemini-3.8-flash／gpt-6.1-sol。
四者都獨立檢查同一 birth／lease／batch interference 問題，再分別實作：
event-resolved birth lifetime competitor、stable-identity exact prefix＋
強 One-move、真正 supplied-client streaming HTTP adapter、
full-run ledger witness／all-assigned protocol audit。
不增加 MPC weights，不從單一 legacy first-action difference 宣稱 causal value。

Source144 identification03 為 fresh-source continuation，Train15／16
接 Cal17–36／Val37–40，planned31,200秒。Original H11／H12／H13 sources
及 gates 先封存；15／16 完整後只執行原 gates，失敗不改5%規則或挑 endpoint。
Mayor 已在封存後看過部分 Train15 summaries，R14 新模型不得再把
15／16 稱作 pristine unexposed testing。Cal／Val／Final 不供調參。
本次 GPU process／driver／120秒 replica monitor 已實際啟動；
11 complete episodes／1,320新秒＋舊15,960秒＝17,280秒＝4h48m
是此版 snapshot，不是最終總時數。48秒 smoke 保守不計主 campaign。

Native03 擁有 backend，cohort traffic 不可平行注入；真正 True
ignore-EOS／controlled cohort 在安全完整 boundary 或本次 driver 完成後
執行，不能停掉 partial 後當作 valid。12h driver／13h service timeout
獨立於 local monitor；結束／失敗後依 PID26392／startticks101822168
及 command hash 精確停止 owned inference，禁止 reuse 舊 PID。
這不等於停止 Pod billing：provider lifecycle／hourly USD 未取得，
保留 Pod 的既有授權不代表免費。

14:51:55Z 四 current native sessions deduplicated usage 為
Opus3,346.58492、Astra8,029.45735、Gemini1,792.5978225、
Sol1,396.08977，合14,564.7298625 credits；
排除 Mayor／舊 sessions／provider invoice，不用 generic cost 當 credits。
十小時主 campaign、fresh gates、真正閉環 superiority 仍未完成。

### 可追溯證據

`results/mpc-native-v3/review-evidence/`：

- `native-v3-r13-frozen-source.tar.gz`／`native-v3-r13-source-manifest.json`
- `native-v3-r13-mayor-current-source-qualification.txt`
- `observation-smoke-144cef3-public.tar.gz`／對應 member manifest
- `native-v3-r13-actual-observation-qualification.json`
- `native-v3-r13-stable-crn-mayor-replica.json.gz`
- `native-v3-r13-r14-native-worker-usage-snapshot.json`

原始 R12 regression、失敗 Native14、此前負面 model／deadline 證據全部保留。

## NV3-017：Source 邊界、split 與完成流程隔離

Train15／16 的20完整 episodes／40 raw-events files 已全部 off-Pod
checksum 驗證並封存，加 index／checksum ledger 共42個 members。
Tar SHA256 `8f228207c3e2af28973e727e6e630e8d0ae29c0e910b23b991d658e8136f0c16`，
index SHA256
`a7e088fb410ef29a032b2bcd4e8a0dd3d59bd7b1d33a38b7bc2e8146950ca54b`。
Preregistered R13 math source 與 gates 在資料前封存，
但 original `bind_source()` 僅支援6c11／729、共同 coreb4f234。
對 Source144／coreed8 的實際呼叫正確返回
`ValueError: replica_source_binding`，沒有計算任何 fresh forecast。
這是 consumer identity boundary，不是模型失準或可以忽略的錯誤。
另版 Source144 strict input adapter 必須先封存並驗證實際 core／collector／
protocol／cost／support／domain／all assignments／raw-events hashes，
再重用 frozen `episode_from_replica`／`bind_assignments`／model math。
不得 monkeypatch frozen registry、假造729 identity、改 gates 或刪除失敗。
本次前瞻性 input qualification 同時保留 instrumentation epoch 差異。

Mayor 對 frozen R12 v9 另行執行同 PYTHONHASHSEED7、Train1–3／USED8–9，
JSON SHA256
`804edf3341bd8a0885bd1b93780a1b8de541110e54d1349f1eef1120ab65def0`
與原 Worker raw byte-exact 相同；v10 b5703 的獨立重現另列 NV3-016。
二者仍是 development prediction 證據，不是 GPU／MPC 勝利。

**兩項重要語義：** Source144 `gateway.py:239–246` 的 first-content
只更新 first_at／sequence，cost reservation 持有至
cleanup `529–533`／`finish_request:439–453` 的 HTTP terminal；
不能把三個 aggregate rows 當成三次 lease changes 或提前 release。
Decode anchors 仍會占 full request cost，但真實 max512 survival 尚未測。
Cost budget16／32 也不等於 FixedConcurrency16／32；目前 runtime maxseq32
固定、cost-limited長 prompt 的 state coverage 未必支持16 running requests。
因此 current budget-policy identification 不能冒充公平的 FixedCap16 比較。
四模型已被要求獨立檢查真正 cap actuation／strong baseline／instrumentation
所需介面，不以改名或調權重解決。

Source144 protocol 已固定 Train15／16、Cal17–36、Val37–40。
R14 先前「fresh Train17／18」提案與這個 split 不符，已要求保留為
不適用提案並修訂 prospective gate：新 Train41／42 或未曝光的正式
Validation protocol，不能將 Cal 重新命名為 Train／candidate selection。
Original H11／H12／H13 全規則保持；R14 新模型不得把15／16再稱 unexposed。

**完成流程的實際隔離：** Host 狀態檢查發現 pr32 Witness／Refinery
均仍在運作，Refinery queue0；Sol 再呼叫 `gt done` 被 unconditional
wrapper 返回86，其 Witness escalation 沒有授權研究完成流程。
Mayor 先驗證 pr-witness／pr-refinery pane PIDs、cwd／startticks，
只 SIGSTOP／SIGKILL 這兩個 owned supervisor process，沒有調用會
「完成 in-progress merge」的 graceful refinery stop。
再 `gt rig park pr32` 設 local/ephemeral parked status，daemon 不自動重啟；
兩 supervisor 已 stopped，四 Polecat sessions 仍在，GPU experiments 未停。
沒有修改 Git remote／history、沒有 main merge，兩 unconditional
pre-push hook SHA2565e9dd500… 不變。Park 可在未來本機 wisp cleanup 消失，
所以 unconditional Git／gt-done barriers 仍是必要防線，不能只靠口頭提醒。

15:13:54Z snapshot：21完整 episodes／2,520新秒＋舊15,960秒，
共18,480秒＝5h8m；十小時仍未完成。四 current native sessions credits
15,235.053575，排除 Mayor／舊 sessions／invoice；USD 未知。
Cal／Val／Final raw 沒有在本次 evidence bundle 或本機分析中開放。
Native03 仍執行，失敗／結束的 owned GPU shutdown monitor 不變。

新增可追溯檔案位於 `results/mpc-native-v3/review-evidence/`：

- `native-v3-source144-train1516-frozen.tar.gz`／member manifest
- `native-v3-r13-source144-original-loader-refusal.json`
- `native-v3-r12-within-step-release-mayor-replica.json.gz`
- `native-v3-completion-supervisors-stopped.json`

## NV3-018：R14 資格與原規則 fresh 確認裁定

Experiment ID：`NV3-018-original-H11-H12-H13-source144-confirmation`。
Parent publication commit：`79fe17f29b000364ce9712987293902dcac8fbab`。
GPU serving source 仍是
`144cef32a9b51c3a74355ff67f5eafdfac6e075f`，沒有 hotpatch。
GPU／模型／software 與 NV3-016／017 相同；controller
`mpc_active=false`、calibration `unfitted-not-validated`。

### 來源、方法與原規則

四個 R14 native local commits：
Opus `e01b15a6741b1c40cff87f5f50609a244f7b2530`；
Astra `473d4a611f88715d75432cde2df75fb7e80ad683`；
Gemini `585cfb31b3b67162dd34829fa9788ad0c1a793c0`；
Sol `598ad1584fc778aa87627ddf22778c92bff6b16d`。
十二個 owned source／test／report files 已封存，archive SHA256
`856eba1d3987656767941601ba70abf2dda4abe71d3feee39fbf5a928e971a08`。
Mayor 對四 modules 與目前 capture／observer 的完整 fixtures 實際執行：
**419 tests／219 subtests passed、零 skip，56.36秒**。
Astra Worker 自己的90 tests 有27歷史 optional skips；Mayor 補 fixtures
後的零 skip 是另一次實測，沒有抹去 Worker skips。

原 H11／H12／H13 各自用原 frozen model、原 Train1–3 fit，
strict Sol Source144 bridge 僅投影完整 Train15／16；九個 helper AST
一致，model／normalization registry 執行前後相同。
所有共用的歷史 numeric scores 在 H11→H12→H13 精確相同。
沒有 alias729、調成本／weights／seed／門檻或新資料 refit。
確認資料20 episodes／40 raw-events files，共2,400 offered 秒；
兩個獨立 seed blocks，不把554／552個 overlapping origins 當成獨立重複。

原 R11.4：H11a 為 aware−blind transition log loss 兩 block 都小於0；
H11b 為 aware G5 MAE 小於 blind；H11c 為 aware G5 MAE 不高於
0.95×v7，且 rejected5 bias 不差於 v7+0.5。
原 R12.4／R13.4：每 endpoint 的 rejection bias／G5 MAE 比 aware 好，
且 G5 MAE 不高於同 endpoint v7 的95%，兩 block 都成立。
若只有一個 endpoint 通過，原 stop rule 是「interval 內未辨識」、
**不是 pass**；兩 endpoint 都不完成確認則不是 forecasting／MPC candidate。
Comparator 是 v7 `r8_admission_birth`／`r8_admission_birth_lower`，
不是較弱的歷史 `r7_full_plan`。

### 全部 endpoint 結果，沒有挑選

G5 是原 diagnostic 的五秒 good-count prediction MAE，不是 req/s goodput。

| Block | v7 upper | v7 lower | H11 aware | H12 upper | H12 lower | H13 upper | H13 lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| 181015 | 4.026825 | 3.927614 | 7.605858 | 4.194043 | 3.725346 | 4.164049 | 3.758557 |
| 181016 | 4.371300 | 4.268089 | 7.624693 | 4.435418 | 4.036251 | 4.421181 | 4.037340 |

H11 transition aware−blind log loss 為−0.059282／−0.052159，
aware 也優於 blind：a／b pass，但 c 的 MAE 與 rejection bias 都 fail。
因此「batch key 有資訊」成立，完整 v8 forecaster 不成立。

H12 的 a／b 在所有 endpoint／blocks 成立；lower 相對同 endpoint v7
改善5.1499%／5.4319%，upper 退步4.1526%／1.4668%。
只有 lower 完成5% gate，故依原規則 **UNIDENTIFIED_WITHIN_RELEASE_INTERVAL**，
不是 forecasting pass，更不是 MPC 勝利。

H13 a／b 同樣全成立；lower 改善4.3043%／5.4064%，upper
退步3.4077%／1.1411%。Upper 兩 block 都 fail，lower 的 block15
未達5%；兩 endpoint 都未完成全 block gate，裁定 **NOT_CANDIDATE**。
H12 lower block15 的3.725346與 H13 的3.758557 分處5%門檻兩側，
是必須保存的 Monte Carlo stream／threshold 敏感性；不能挑較幸運的
舊非 stable pairing 或 endpoint 當成改善。Stable CRN 修復是 correctness
要求，不依較有利 forecast score 決定採用。

Nominal80% G5 band inclusion：
H12 upper .655893／.654564，lower .710213／.687420；
H13 upper .666722／.663183，lower .706848／.685604。
這些不是已校準的80%機率保證，不能授權 chance constraint／native MPC。
完整一／三／五秒與所有其他 metrics 保存在原始 JSON，不只保留表內結果。

### 求解器與真實 HTTP 邊界

Astra R14 在同16 USED8／9 cuts、兩 endpoint 的
32,768 ten-metric scenario vectors／3,072 means，exhaustive／prefix EXACT。
強 One-move／MPC first-action differences 全為0；原唯一差異位置兩者
現在都選16。100ms deadline failures exhaustive20／prefix11，
兩個 p128-m128 cuts 連 One-move 都未完成，不能將 unknown 當作相同。
Paired median speedup upper1.723834／lower1.656545，為 Worker
Python3.14.4 CPU profiling，不是 GPU、hard-deadline guarantee 或 goodput。
完整 raw SHA256
`082b18401b8a7668bb03510f848f51106d622354a8a08b5c08659f54868208de`。

Mayor 對 frozen Gemini R14 再次讀碼發現：supplied real client 仍可能
用 mock binding／transport，failed drain／invalid CAS 後仍 dispatch，
empty cohort／缺欄位可能 vacuous gate，burst await full completion 而串行，
horizon 也不是 actual apply+1秒。Remaining>90 token 不是物理 survival bound。
此版尚不可 Native pilot；CPU419全通過只證明既有測試，沒有把它部署。
R15 原生 `.9`／`.10` 由 Gemini 修 genuine fail-closed runner，
Sol 以 actual-shaped client 獨立反例／驗收，不能共用 mock 當 GPU 證據。

### 下一個有資訊價值的工作與安全／資源

Caller-owned／merge-local convoy `hq-cv-w6kc7` 追蹤 `.9`–`.12`；
Opus／Astra `.11`／`.12` 獨立量測 tagged Native／Gateway owners、
whole-cost stock、latent pending／re-offer proxy 與真正多步資訊價值。
Aggregate ledger prefix 可檢查 stock timing，不可假造 per-UUID terminal
ownership 或 final watermark。Cost budget16／32 仍不等於 FixedConcurrency16／32；
需要明確且實際套用的 cap interface／state coverage，不能改名冒充 baseline。
原 H14 用17／18的提案保留為 INVALID；prospective Train41／42 不改。
15／16 在原 gate 算完後已 USED，R15 不可再稱 pristine；
Cal17–36／Val37–40／Final 沒有用於本輪模型分析或 selection。

15:55:08Z verified ledger prefix：39完整 episodes／78 raw-events files，
4,680新秒＋15,960舊有效秒＝**20,640秒＝5h44m**；48秒 smoke 不計。
十小時尚未完成，Native03 acquisition／off-Pod replica／owned shutdown
monitor 持續；沒有外來 cohort traffic 或 source hotpatch。
15:55:24Z 封存的四 current native sessions credits 為16,170.26603：
Opus3757.1746／Astra8701.983／Gemini2122.54587／Sol1588.56256，
排除 Mayor／舊 sessions／invoice，不是美元帳單。
GPU hourly quote／provider lifecycle credential 不可用，估計USD仍未知；
停止 owned inference 不等於停止 Pod 計費。
Witness／Refinery 仍 stopped；四 Worker tmux sessions 保留，
兩 pre-push hooks SHA2565e9dd500…不變，沒有 `gt done`／Merge／其他 Rig 操作。

本次新增 evidence 在 `results/mpc-native-v3/review-evidence/`：
`native-v3-r14-frozen-source.tar.gz`／manifest、
`native-v3-r14-mayor-current-source-qualification.txt`、
三份 `native-v3-h11/h12/h13-original-source144-fresh1516.json.gz`、
`native-v3-original-fresh1516-adjudication.json` 與
`native-v3-r14-four-worker-usage.json`。
H11 raw SHA256 `a81c6c8c…426fade`，elapsed70.52秒／RSS810,032KB；
H12 `1cfc71f8…6c0be3`，132.23秒／RSS896,996KB；
H13 `12291743…d32d91`，218.37秒／RSS897,588KB。
完整 SHA 在 adjudication；compressed files 解壓後逐一重算確認。

## NV3-019：R15 measured-state 缺口與 R16 觀測修復

Experiment ID：`NV3-019-R15-state-stock-and-causal-HTTP-qualification`。
Parent publication commit：`c38495b7be1cdeea56370c40fe0d8fcb20a870b2`。
Live source／GPU／model／software 與 NV3-018 相同，Source144 沒有 hotpatch，
`mpc_active=false`、`calibrated=false`。

### 四模型獨立成果與完整 CPU 資格

| Worker／實際模型 | R15 sealed local commit | 本輪證據範圍 |
|---|---|---|
| amber／claude-opus-5.5 | `9fdfe5770ccfc4ea9db0488486a834bdcde78c90` | USED15／16 owner／native tags、pending stock；48秒 public smoke journal |
| jade／gpt-6-astra | `c4f4844d48bf40d09e0306a2cac0489c54b365c4` | 既存32張 R14 tables 的 action-value 分解；2個預定 native prefixes；actual Gateway CPU witness |
| pearl／gemini-3.8-flash | `f9d5ad48fbc32ae6302f37bc4c84895c7300f91d` | Genuine supplied-client runner 的 provenance／gate／CAS／dispatch／window／censor 修復 |
| flint／gpt-6.1-sol | `6f861efa85ac46a9e74acff74d1b576cdaef91d3` | Frozen R14 actual-shaped HTTP 反例、all-assigned bounds、source bridge 不變 |

十二個 committed owned files 逐一核對，archive SHA256
`08fedfbc364a392c1efed9d84b721f7db3140e8b0f2db48d1a0d1a5ac9a4be1e`。
Mayor 完整 fixtures／actual Source144 Git source／capture／observer：
**438 tests／237 subtests passed、0 skipped，115.45秒**。
其中 Sol 先單獨實測100 tests／165 subtests／零 skip，11.23秒。
Astra Worker121 collected／94 pass／27 historical optional skips 保留；
Mayor 正確補 fixtures 後的零 skip 是另一次 qualification。
這些都是 CPU contract tests，不是 Native controlled-cohort 或 MPC 驗收。

### Measured stock／pending 缺口，不以參數補償

Opus raw used1516 SHA256
`27e416dba59469e440842b22ad8c08d611a9ef5bddd1637addd789fc5bbe8b51`；
smoke raw
`6804a05b35ff39075a652f6d8a3db87bfb91e47719dae508a4ab4d0fc8f642ee`，
Mayor 已取得完整 bytes 並重算確認。這是 Worker 的可重現 descriptive
readout，沒有新 forecast／fit／Native action execution。

| USED block | Pending later-owner proxy 漏掉 | Pending owner cost 低估 | 後續 owner 已在 cut admitted |
|---|---:|---:|---:|
| 181015 | 27.2% | 23.1% | 70.3% |
| 181016 | 27.8% | 21.7% | 68.3% |

Proxy 用過期 snapshot stock 篩 pending offers；snapshot 後已 release 的
owners 仍把 stock 膨脹，因此錯誤地 drop 真正後續 owners。
Rejection side 幾乎精確，兩 block 合計只有3個 rejections 被當作 Q。
這支持 stale stock／unseen releases 是此 proxy 的主要缺口，
不能乘一個 correction factor 或改 arrival law 當成已識別的修復。
Ground-truth future fates 只能做 retrospective audit，不得回灌 online state。
這也**不能**直接證明它造成另版 synthetic CPU MPC 與 Fixed16 的舊20.7%
goodput差距；資料與模型不是同一條因果鏈。

Tagged join：兩 block 無 rejected IDs 後來變 owner／native-visible，
約94% HTTP terminals 在 owner release snapshot interval 內；
沒有 HTTP terminal 早於最後 owner snapshot。Producer list 與 running+waiting
不完全一致的 frames 為1178／8933與1145／8917，unknown visibility 仍保留。

48秒 public smoke aggregate journal 與303／303 snapshot stocks 一致，
299／299 snapshot intervals 的 release multisets 一致；
643／643 releases 的時間可辨識，但只有82個（12.8%）能唯一辨識 owner。
相同 cost 的多個 departures 不能按 FIFO／adjacency 指派 per-owner lifetime。
Release position 在 interval 中近似均勻，median u=.512；
不能以較幸運 lower endpoint 救回 H12／H13。
以相鄰 completed event 對齊的 HTTP／Gateway lag median3.7ms，
不是已認證的 per-UUID lease pairing；所有 whole-cost decision checks 零違規。
兩 epochs sequences 無 gaps，但 status 沒有 final sequence／count／digest，
所以仍不能證明 retained suffix 完整。
USED1–16 sealed subsets 都沒有 journal，**H12未辨識／H13非候選維持原判，
沒有 rescore、midpoint choice 或拿 smoke 當 calibration**。

### 多步動作價值與真正 count-policy 的條件

Astra 完整18340-byte raw SHA256
`5c47a79b2b83e9e0134537771144d75a81cb54fce1b482554f67550de852f101`。
只分解既有32張 R14 tables，沒有新 forecast／profile／refit。
每 endpoint 的16 cuts 中，**4個兩 family 都選16、12個都選32**；
先前 disputed cut 的 conditional margin：
upper −.4375→−.1875，lower −.4375→−.25，tail bonus 沒有翻轉 ranking。
因此「model 的多步項完全沒有作用」太強；正確結論是目前 saved cuts
沒有 first-action 額外價值。原32768 exact、20→11 deadline failures、
兩個 One-move deadline-incomplete cuts 都保留，不能把這些 unknown 當成
100ms 可執行的相同動作。

Actual Gateway CPU witness：first content 持有完整 cost，downshift 不 eviction，
超 budget 立即503而沒有 real budget waiting queue，latent Q re-offer 尚未識別。
兩個預定 mixed-arm0 start+10s prefixes 的2／2與5／5 native tags 可對 owners，
各有 asynchronous pending_batches1，不能宣稱 full lifecycle／strict loader permit。
以 measured long-cell costs，從空狀態 budget32 最多只有7／6 long owners；
16個 long owners 至少需 budget64.48695／75.15362，或真正 count-policy16。
不能將 budget16／32 重新命名為 FixedConcurrency16／32。
Future action-value experiment 需真實 `16 < R+c <= 32` admission contrast、
actual CAS/apply 與 ITT clock，不能只比較 virtual plan。
v11 有 lag pool／stream／reordering／clipping，非 v10 exact ABI；
沒有盲目接入 prefix solver 或重新 fit。

### Genuine HTTP 修復與 independent negative reference

Sol canonical witness SHA256
`6465bcbb112bca0a39ae07716ea96d50375145e6571e55082994529dd9802b0b`，
535 bytes，Mayor exact SHA 核對。Frozen R14 在 real-shaped client 下仍能
failed drain／empty gate 後13POST，unknown／failed CAS 後4bursts，
serial full-SSE 把 end 改成 last-HTTP+1秒，九個 disappearances 被 zero-impute
卻宣稱 survival。**Actual capture 本身會遵守 future scheduled_at**；
問題是 serial await 破壞 common arrival/window，不是 capture 忽略排程。
Positive CPU reference 是4個 concurrent attempts、apply+1秒 sample 在
HTTP completion 前、all-assigned censor intervals，仍非物理／causal permit。

Gemini R15 已修 dynamic server provenance、source／drain／gate／CAS fail-closed、
concurrent burst at apply+offset、apply+1秒 sampling 不等待 HTTP finish，
end-survival／censor intervals 與 geometry action contrast preflight。
Mayor438全套包含這些 contracts，但**尚未在現有 GPU backend 執行 pilot**；
Native03 仍獨占 backend，不能注入外來 True ignore-EOS cohort。

### R16：隔離本機 owner journal，先取得 source-qualified 觀測

Mayor 在另一個 pr32 worktree
`/home/daniel_lin/gt/pr32/mayor/mpc-native-owner-journal`，
branch `research/pr33-native-v3-owner-journal`，提交
`a4ec4e386e68f6ea20b68a8557785fc5813c7b16`。
`MPC_OWNER_LEDGER_ENABLED=1` 為 default-off opt-in，需 MPC lab／cost_aware／
telemetry log+shutdown status。Schema-v2 rows 含 explicit owner ID／whole cost／
admission-first_content-terminal kind；control rows 不捏造 owner。
First-content 不 release；writer 停止與 final flush 後，status 綁 final sequence、
remaining owners、完整 file bytes／SHA；不一致或 leftover owners 標 failed。
**116項既有與新 CPU cases 全通過，0 skip，0.80秒**，包含 actual mock-HTTP
lineage、non-FIFO release／downshift、短 session final flush／closed epoch／
leftover owner failures／startup refusal，既有 snapshot strict contract 不變。
這不是 fsync／durability、external authentication、calibrated risk 或 MPC permit。
新 source code 仍只在本機隔離分支，沒有部署、Push 或改 active Source144。

先前把新 Gateway bytes 放在 historical R15 parent 時，5項 producer-source
checks 正確拒絕、2個 fixtures 未提供；原輸出完整保留，沒有說成 pass。
已將**僅 Mayor 自己的變更**移到獨立 worktree，原 parent恢復原 bytes，
補 actual Source144 fixture 後438／237／零 skip。沒有 override SHA／identity、
刪失敗、重寫 Git history 或拿 new-tag／old-data 作 source qualification。
四模型的新 caller-owned／merge-local convoy `hq-cv-opi13`、Beads `.14`–`.17`
正對相同 state／stock／lease／watermark／count-v-cost／clock／ITT／horizon
核心問題交叉批判；不是把 Mayor 實作當成四模型同意。
Prospective Train41／42仍需先凍結 source、包含完整 epoch journal／watermark，
重新設定可判讀的 primary metric、split、ITT 與停止規則，不用15／16重調。

16:25:23Z verified prefix：54完整 episodes／108 raw-events files，
6,480新秒＋15,960舊秒＝**22,440有效秒＝6h14m**；smoke不計，
十小時／真實 MPC closed-loop superiority 尚未完成，native acquisition 持續。
16:22:09Z actual four-session usage：Opus3942.79084／Astra9016.96195／
Gemini2221.8787725／Sol1612.34939，合計16,793.9809525 credits，
排除 Mayor／舊 sessions／invoice；GPU hourlyUSD／provider billing-stop 未驗證，
不將 service stop 冒充停止 Pod 計費。

新增 publication evidence：R15 source archive／manifest、完整438／237 output、
changed-parent-source refusals、兩份完整 Opus raw gzip、完整 Astra raw gzip、
Sol canonical witness JSON。重要負面結果、所有 endpoint／fates／deadlines
均保留；本輪發布的是辨識與觀測資格，**不是 MPC 性能勝利**。
另附 `native-v3-r16-owner-ledger-local.patch.gz` 與116項 qualification output，
僅為本機 prototype 的可讀 source／CPU 證據，不是產品 HEAD 整合或 Native 部署。
原始 refusal log／patch 的空白不改寫，以 gzip byte-exact 保留；
uncompressed SHA256 分別為45e0a33c…68662／d2a6b703…9862e。
