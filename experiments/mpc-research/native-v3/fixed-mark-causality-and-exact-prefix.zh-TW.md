# NV3-012：固定 marked 模型的因果缺口與精確 solver 加速

**裁定：保留v5／v7作為descriptive forecasting baseline，
不將它直接升級為已有效的Native MPC plant。**
其預測改善是真的，但future admission對既有請求服務的回饋路徑不存在。
開始四模型各自的batch-aware evolving transition研究，尚未批准任何Native controller。

## Release sensitivity 的獨立重現

Mayor使用原始R9預註冊source `c58ba883...af35`，
重新執行完整Train8／9：raw SHA256
`d3f6d3ec5a49abd7b16b639d47cc29ca6739812fcbcf545d9b6fe72ff040a0a5`，
與Opus **byte-for-byte一致**；1112 scored origins／543 factual pairs。
兩個paired arms已有不同states，仍不是same-state causal action effect。

| 指標 | UPPER release convention | LOWER release convention |
|---|---:|---:|
| G5 MAE | 4.155 | 3.996 |
| Rejected5 bias | +2.66 | −0.56 |
| G5 nominal80% band inclusion | .595 | .656 |
| Rejected5 nominal80% band inclusion | .555 | .552 |

最強R7同-mask G5 MAE5.856；兩端均較低，但一秒仍輸R6／R7，
offers5仍輸R7。預註冊H9a／H9b達成，不代表識別出真正release time。
LOWER沒有因結果較好而被採用，兩端不被宣稱為Goodput的guaranteed bounds。
名義80% bands仍under-cover，沒有risk／MPC permit。

## R10來源與修正後CPU資格

四模型original R10封存archive SHA256：
`73a6f5b84f25a03cff46b25bc641c3bb10baa36866637ba91bda551f0b48e547`。
Opus `62512df...` 是eligible-uniform repair的**評分前規格**；
Astra `48e3331...` 實作exact prefix reuse；
Sol `11400f7...` 保留pre-outcome Gateway binding條件與未知Native lineage；
Gemini `ab2eb64...` 修正foreign ABI path，但合併測試仍發現v1／v2 schema mismatch。

Gemini另提交 `e2a1cdcb0b2d16cfbf800047ae71ec2d5e607d1c`，
修正source／consumer版本契約；原始失敗封存未覆寫。
Mayor以三個original R10 modules及此consumer fix，在qualified Python3.12、
所有R1／R2／R4及frozen-v5 fixtures啟用下：
**286 passed、165 subtests passed，沒有failed或skipped。**
這是CPU／mock-HTTP資格，不是GPU／HTTP閉環效能驗收。

## 真正的exact prefix acceleration

Astra曾將qualified Python3.12 measurement列為環境blocked；
Mayor用現有qualified interpreter實際完成測量，沒有安裝依賴或放寬100ms。
同used Train6／7、同kernel／draws／plans與origin masks，
1111 cuts上的 **1,137,664個scenario-level comparisons精確相同**。

| Block | Original wall median | Prefix wall median | Paired median speed ratio | Deadline original→prefix |
|---|---:|---:|---:|---:|
| 181006 | 64.11ms | 46.30ms | 1.309 | 173→137 |
| 181007 | 66.60ms | 51.77ms | 1.282 | 177→148 |

每個cut交替原始／prefix執行順序，每次重新建立draw／prefix caches；
full uncensored scenario audit在timed call外，不被偽稱為deadline內推薦。
Prefix最大實測136.43／151.65ms，cooperative callback checking仍非hard100ms保證。
短prompt／高rate cell的213次呼叫仍全部deadline。

這是solver運算改善，不是GPU Goodput提升。
完整offline audit涵蓋全部1111個分析cuts，**first-action differences仍為0**；
tail differences僅block6兩個、block7六個，其中包含零gain tie。
它補上R9被deadline遮蔽的offline ranking，但不能把deadline執行算作成功action。
Optimized-policy新states與真實閉環仍未驗證，不能推論所有hardware／model上MPC無效。

Raw SHA256：
`af4e5926ad2e89a6f733fbd63e6c0948ea0c73bef58cf6c7837a09d6eb7b9a30`。
Source SHA256：
`19e365413a35b6a81b4be932dba6f795b189168eea5a1ee677e2edf8b4cf7f2d`。

## 程式可表達性的證明，不是GPU效應大小估計

`simulate_plan`對existing cohort只在cut抽一次 `(release,good)`；
對birth只在admission依當時Gateway-owner load bin抽一次duration。
之後的admissions從未重抽或更新這些marks。因此在固定draw下：

\[
\frac{\partial T_{\mathrm{cohort}}}{\partial A_{\mathrm{future}}}=0,\qquad
\frac{\partial T_{\mathrm{earlier\ birth}}}{\partial A_{\mathrm{later}}}=0.
\]

Opus／Gemini／Sol已提供executable fixed-CRN witnesses；
該結論也可直接追溯至frozen source的heap／mark更新方式，而不是多數決。
模型能表達「admission前load影響該birth」，不能表達後續prefill／decode
競爭對既有請求的回饋。這足以阻止以factual MAE勝出宣稱causal MPC ranking可靠。
它是否為實際GPU效能落差的全部原因、效應有多大，仍需controlled intervention。

Opus在已使用Train1–3觀察15／15個keys中，較多post-admission load對應較慢
remaining client duration；pooled差約+.119s，prefill-heavy約+.365–.369s。
這是有confounding／EOS及client時序成分的observational residual，
不能當GPU service time或causal slowdown估計。

## 下一輪的拒絕條件與資料界線

四模型各自提出一個minimal evolving model，再交換critique；
source-bound actual observation、unknown waiting／not-visible work、
EOS uncertainty與whole-cost release分開建模。
現有native frames是running／waiting／pending／KV及reported per-ID counters，
不是已完整收集的per-step GPU batch composition或physical execution clock。
MissingIDs不能補成零工作，reported computed也不能直接當completed GPU tokens。

一個早期R11 prototype的baseline讀evaluation targets求mean、
未驗證ID substring join及disappeared-ID final-usage補值，
已被Mayor拒絕；其「22%」不被當成有效模型改善。
要求修正past-only baseline、source／clock與censoring、actual fit、
float progress及真正forecast protocol後，才可曝光新Train12／13。
既有12／13辨識資料不是八類controller benchmark，不能拿它宣稱5%Goodput勝出。

新Train10／11僅供已先封存 `62512df...` 的四版本kernel correctness比較；
修正採用依conditional-uniform correctness，不看MAE決定是否修bug。
完整20episodes／40checksums已封存，index SHA
`616f7ab9a62f50d387156498612a94cc37b0507c4a0a77ae03062fb8df75d4c6`；
此時才交Opus執行原始R10source，結果尚未宣稱完成。
12／13、Calibration／Validation／Final均沒有提前交給新model評分。
沒有hot-replace live729。2026-10-31T12:35:47Z續跑78episodes／9360 offered秒，
原完整3600秒另列，合計3小時36分；**十小時尚未完成**。
GPU仍在有效辨識；沒有第二GPU。Rate／Pod計費停止仍未知，不能將服務停止稱為停費。
