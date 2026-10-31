# NV3-013：Evolving batch model 的負面結果與觀測支援邊界

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
