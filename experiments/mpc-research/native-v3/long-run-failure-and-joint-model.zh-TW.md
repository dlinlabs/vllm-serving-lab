# NV3-004：原生長時中止、joint rollout 反證與受控恢復

**長時辨識只完成30個有效120秒 episodes，即3,600秒／1小時。
未完成十小時，也未證明 MPC勝出。** 原始計畫400 episodes不能當作結果。

## 實際中止與安全處置

凍結 collector `6c11fe7` 於 `2026-10-31T09:27:47.977080+00:00`
安全中止，錯誤為 `bounded assignment failed; no unacknowledged action accepted`。
失敗 trial是 `nv3-181004-p128-m128-arm1`，在第50秒：

- 第49秒 cap32已取得 applied ACK，version50。
- 第50秒再次指派相同 cap32；三次 POST皆回傳 pending，前兩次後續
  reconcile確認 expired、observed cap32／version50。最後 pending intent
  保留，`not_after=1000674.910200009`。
- 這些結果不能改寫成成功 ACK，也不能算 solver timeout 或 MPC效能反證；
  此時仍是 randomized identification，沒有啟用 MPC。

30個完整 episodes、失敗 episode JSON／durable events及錯誤 index皆已
off-Pod保存；失敗 partial window不計入3,600秒。失敗 JSON內3,981筆 row
包含已完成及取消／未送出的預定 requests，**不等於全部實際 offered**。
不刪除失敗或從 planned offsets製造完整 measurement window。

Driver已停止自己建立的 Gateway。Mayor核對 `/proc/3487/cmdline` 後，
對自己建立的 vLLM PID3487送 SIGTERM，確認8000 endpoint關閉；原 detached
service退出0。保留既有 Pod，沒有新 GPU或服務採購。
**Pod billing停止仍無法驗證**；停止vLLM不等於停止計費。

### 最小恢復，不放寬門檻

Cap不是 lease，100ms command expiry不會自動撤銷已 applied budget。
因此重複指派相同budget不必再次發送 mutating POST：
新 collector在完整 clock／identity／cost／freshness驗證後，要求同 run、
同 version、同 budget與先前 applied ACK一致，另列 `confirmed_hold`。
這是「觀測已生效的既有budget」，**不是製造新command或ACK**。
Pending、stale、version drift、identity mismatch都不能建立hold。

真正改變budget仍保留原100ms TTL、CAS、pending fencing、三次 bounded
attempts及 fail-closed規則。沒有放寬TTL、SLO或snapshot門檻。
新增 journal開始／完成與pending wire command證據，以區分durable fsync、
client scheduling、transport與server延遲；目前不能只靠總elapsed推斷
這次expiry一定是fsync造成。

恢復使用fresh experiment output與new source，只略過經原始index驗證的
前三個完整Train blocks；從181004完整配對重跑，失敗 partial保留於long01。
剩餘370個120秒windows計畫44,400秒／12.33小時，**仍須實際執行驗收**。
23項 coupled CPU regression已通過；Native hold smoke／恢復長時尚未
驗收，CPU不能代替GPU執行。

## 四模型 R4 的實際交付

| Worker／backend | 凍結local commit | 交付與邊界 |
|---|---|---|
| amber／claude-opus-5.5 | `8c233e5ba8f8bd74bf8fa9b20098a2d189c41289` | 0.5秒joint aggregate stock-flow ridge、Train cross-fit與真正recursive rollout；scenario未校準 |
| jade／gpt-6-astra | `018ed9b6c4f03b2af309fad6f5ff46f5578bb91f` | exact-ID marked-event composer、共同future law與nested-family diagnostic；沒有fitted native event kernel |
| pearl／gemini-3.8-flash | `a6479e630b48d449baae471f9d60da744d5a1252` | 修復epsilon scale、NaN／Inf及偽MPC aliases；receipt統計authority仍需查核 |
| flint／gpt-6.1-sol | `ece0327af1b8be6b182c12a605a8a3de374b23e4` | 真實Train capability audit，保留online history／physical EOS缺口與source-bound release intervals |

Mayor逐檔 byte-verify凍結並重現 **171 tests、128 subtests**。
R4 archive SHA256：
`ec977749cbdb54b2b6bb16d0de584630459d9e8f0c5067c1ae5b6107e775bfa4`。
原R3缺陷來源仍封存，不覆寫歷史。Astra／Sol再次嘗試completion時仍被
exit86實體防線阻擋，沒有授權Worker push。

## Mayor獨立重現的真實Train rollout

使用兩個完整Train blocks181001／181002，20個JSON與20個events逐檔核對
後建立immutable subset。凍結Opus R4 module：
`670edf6627a53eae14ea80a47a88ffb4e0795254b606d90d3d58c3e13edfc2c8`。
Mayor實際執行cross-fit，完整JSON SHA256：
`b9f90e23ff22d79e373ac42ae562e444450af83310cc62852d4da66af891a7d0`。

後續good terminals來自cut後births的比例，1／3／5秒分別為
**19.42%／68.29%／80.95%**。所以current-cohort release幾乎5秒全數完成，
不代表5秒Goodput沒有動態：大部分good來自後續工作。

| Free-running good-count MAE | held-out Train181001：joint／history mean | held-out Train181002：joint／history mean |
|---|---:|---:|
| 1秒 | 1.80／2.72 | 1.65／2.62 |
| 3秒 | 5.29／5.59 | 5.55／5.70 |
| 5秒 | **9.84／8.47，joint輸** | **10.37／8.67，joint輸** |

Joint候選對reserved stock與部分TTFT-miss有訊號，但在5秒good、arrival
與部分rejection forecasts輸給合理history baseline。Nominal10–90% bands
在5秒實際inclusion約42%–75%，未具備risk qualification。
TPOT-miss targets近乎全零，不能推導完整TPOT-risk calibration。
這些是兩個Train block的預測診斷，不是獨立泛化CI或closed-loop勝負。

## 下一輪唯一模型假設及未決介面

正在執行的有界修正：將arrival視為由真正pre-cut all-offered history
產生的**共同外生future law**，不讓candidate cap改變predicted offered tape。
其他sampling／horizon／ridge penalty／metrics不變，以fresh Train181003
檢查，不碰Calibration／Validation／Final或搜尋cost weights。

另由不同模型交叉檢查 aggregate ridge是否能合法映射為original-ID、
whole-cost marked-event kernel；snapshot endpoints無法完整觀測在兩點之間
born-and-released的transient work，獨立clipped均值也不自動保證joint
count／cost／lifecycle consistency。Astra composer不是已fit的服務模型；
兩個模組不能只靠都叫joint就接成有效MPC。

Sol提出更嚴格physical-EOS endpoint與推論方案，仍是**獨立研究提案**，
不是Mayor已採用的Final protocol，也不是現在HTTP metric被悄悄重定義。
API stop、physical EOS、all-offered online receipt、完整source/policy
binding與共同metric必須逐一說明。既有Goodput／SLO不能為了任何模型
缺欄位或有利結果而調整。
