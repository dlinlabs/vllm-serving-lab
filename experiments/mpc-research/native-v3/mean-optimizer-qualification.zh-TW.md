# NV3-011：真正 solver 重現、CPU qualification failure 與下一個可判別假設

**目前結果：已實作有限 mean-only optimizer，但尚未得到多步規劃的實際
first-action 優勢；Native MPC 尚未部署。** 原本的預測改善與所有負面結果均保留。

## 四模型與封存來源

| Worker／實際模型 | R9 本機 commit | 交付 |
|---|---|---|
| amber／Claude Opus 5.5 | `76b1ad3dd2a1765be88ab592878f6044ad100307` | release sensitivity 預註冊、descriptive mean 修正與 regression |
| jade／GPT-6 Astra | `9fab778ea4fe508fa731d655e2dc1f394a92c6ec` | 真正 frozen-v5 32-plan optimizer、100ms deadline 與完整 timings |
| pearl／Gemini 3.8 Flash | `bbc11b9d1a9072ebada975b1b9bb8083d986fc4a` | actual capture／actuator／NativeState／receipt bridge mock-HTTP pipeline |
| flint／GPT-6.1 Sol | `64cde8a2392f7070d470dc4fc4ed666450c0b06e` | source-bound online cut adapter、未知 lineage／cost 明確拒絕 |

12個 owned files 與 manifest 封存於 `round9-frozen-source.tar.gz`，
SHA256 `1d049944c3e93b0767359bb67a613209e87e6688dbc17fa480f41c38f4718c76`。
Opus 此版本是**評分前的規格封存**，不是已完成的8／9結果。
模型 identity 來自各自實際 usage records，不以 profile 名稱或 Mayor 推論替代。

## Mayor 獨立重現真正 optimizer

同 frozen-v5 source、Train1–3 fit、已使用的 Train6／7與32 common scenarios，
執行32個 MPC plans及4個 nested One-move plans，固定100ms deadline。
這不是用假的 model callback 測 solver。

| 執行環境 | Origins | Full family complete | Deadline | Complete cases first-action differences |
|---|---:|---:|---:|---:|
| Astra 原始實測 | 1111 | 808 | 303 | 0／808 |
| Mayor qualified Python3.12 重現 | 1111 | 757 | 354 | 0／757 |

兩次執行的 **29,300個共同完成 plan mean 全部精確相同**；
共同完成的757個 full-family choices也相同。Kernel hash與source一致。
不同CPU／interpreter／即時負載造成deadline masks不同，不能挑較快的一次、
合併masks或把未完成actions算成相同。

Mayor兩block wall median為65.03／62.50ms，P95約100.31／100.20ms，
最大實測262.07／100.50ms；最大process CPU亦約262.07ms。
Cooperative callback checking **不是hard100ms execution bound**；
超時結果均保留為deadline，沒有candidate action或Native permit。
Mayor block6有13次連One-move family也未完成，不能移植Worker「全部完成」說法。

Astra所有短prompt／高rate cell皆未完成full family。七個正向model score
差異僅改未來tail plan，One-move與MPC都先選32；
不是GPU Goodput提升，也不是Monte Carlo差異的統計顯著性。
目前證據不足以將這個candidate直接升級為有效Native多步控制器。

## CPU qualification 沒有隱藏失敗

Mayor四個 frozen modules 合併測試為 **5 failed、258 passed、
165 subtests passed**。五個新pipeline cases失敗：harness將Python3.14
site-packages插入Python3.12，載入不相容NumPy binary。
Worker原本環境的63／63不能覆蓋這個獨立結果。

不重裝已存在依賴、不skip測試、不更換解釋器來掩蓋portable gate。
已派Gemini修正foreign-ABI path插入並另版測試；原始source／failure log保留。
這些是CPU／mock-HTTP結果，不是真實GPU／HTTP closed-loop效能驗收。

## 已獨立重現的數值問題

Sol提出、Mayor以**原始frozen source**重現 `kernel_draw`：
records為A=(release2,good1.5,terminal2)、D=(.5,None,.5)、B=(3,None,3)，
pending age1時D不合格。均勻起始index加cyclic next-alive產生A、B、B，
使兩個合格records機率為1／3、2／3，而非conditional uniform的1／2、1／2。

這會影響預測抽樣，不只是descriptive mean報表。
既有v5正向／負向scores仍是原實作的真實結果，不能偷偷改名成修正版本。
下一版先封存eligible-set uniform repair與regression，
再使用之前未曝光的完整Train10／11比較；不調weights／horizon／SLO救分數。
Pending attempt的實際admission status仍是另外的latent-state問題，
不能用重新admit近似宣稱已證明non-eviction。

## Fresh-data 與實際資源界線

Opus先提交 `76b1ad3...`，Mayor才封存Train8／9的20完整episodes、
40個checksum-verified artifacts與2400 offered seconds。
Subset index SHA `ec4d9160e45b41cd7143445192bcfa5abcdaae6dcb584c89d42e733421b103a3`。
兩個release endpoints都必須報告，不在評分後選有利endpoint。
8／9、6／7均為development；沒有讀Final或授予risk guarantee。

2026-10-31T12:12:11Z off-Pod index驗證continuation完成67episodes／8040 offered秒；
原long01完整3600秒另列。合計11640秒（3小時14分），**十小時仍未完成**。
單張已授權4090正在有效辨識，live source仍729b144；
沒有hot-replace controller、新增GPU或將CPU時間算成GPU實驗時間。
當次GPU utilization100%，磁碟使用43%；hourly USD rate與Pod計費停止仍未驗證。
停止vLLM不等於停止Pod計費；保留使用者已授權Pod的狀態明列。

目前四個sessions累計usage約9254.26 AI credits
（2176.31886／5269.83275／988.9979325／819.11008）；
不含Mayor、舊sessions、GPU USD或最終帳單。
Sol再次嘗試 `gt done` 被實體guard exit86擋住，沒有Push；
已再次要求只交本機receipt，guard沒有放寬。

## 下一個有資訊價值的判別

四模型各自獨立驗證：future admission改變batch composition後，
frozen模型能否改變已存在cohort的service／completion？
出生時已固定的marked duration能否反映之後的prefill／decode interference？
要求可執行counterexample與source證據，而不是先宣稱必定有效或無效。

同步推進exact semantics-preserving solver加速、verified pre-outcome lineage、
原始整筆charge與snapshot-race資格。若需新的batch-aware hazard／progress
transition，先明列可觀測state、辨識與對照，不回到serial HTTPservice假設。
所有新版本另行封存；Native比較須通過真正input／actuation gate，
不得把Fixed16 fallback或nested model utility當成MPC勝利。
