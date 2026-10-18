# Shared-client A／B 開發資料分析

分析日期：2026-10-17（America/Los_Angeles）。C 未執行、未納入分析。

結論：兩次 capture 完整且可重現，足以進行系統辨識。第一版 ridge 的 1–2 秒預測有開發價值，但 5 秒的 SLO 達標／超時預測不穩定，尚不能據此啟用 MPC 控制或宣稱模型通過正式驗收。

## 原始資料與完整性

- Runtime commit：`d0ccb2c0d3711e1b2a69bb4ec248c61961ce3586`。兩次記錄的 15 份程式／依賴檔案 SHA256 全部與該版本一致；git_dirty 為 true，但所記錄檔案內容一致。
- A／B 各自的 native CPU JUnit 均為 476 個案例，無 failure、error、skip。
- A／B 的 report.json 均由原始檔離線重現，所有啟用的完整性與時序門檻通過；B 的新機器 smoke 報告也獨立重現。
- A 的 306 個 snapshot、B 的 307 個 snapshot，各有八個狀態欄位完整對齊原始 lifecycle 重建，無不一致。
- 每次 7,646 份 request 診斷（含三個 warmup）均 cleanup_complete；shared mode 沒有每個 request 的 client 建立／關閉。
- A 的宣告為 shared-client-qualification-v1；B 為 shared-client-b-v1，宣告內容、trace、schedule 與 capture identity 相符。兩者均為開發資料。

| 指標 | A | B |
|---|---:|---:|
| 送入 requests | 7643 | 7643 |
| 成功完成 | 1976 | 2059 |
| 拒絕 | 5667 | 5584 |
| 失敗 | 0 | 0 |
| 成功完成且 TTFT ≤0.5秒 | 572 | 736 |
| SLO goodput（req/s） | 1.907 | 2.453 |
| p99 TTFT（秒） | 2.413 | 2.203 |
| 最大 arrival scheduling lag（ms） | 17.13 | 15.40 |
| 最大 budget application lag（ms） | 26.86 | 18.97 |

這些是描述性結果，不能把 B 較高的 goodput 歸因於反向 schedule。兩台都是 RTX 4090、driver 595.91.07、vLLM 0.30.0、Torch 2.13.0+cu130、context 8192、prefix cache disabled，但 rental host 不同，執行狀態與軟硬體環境仍可能混雜。A／B 使用同一段到達時間與輸入長度 trace，不能把跨實驗驗證視為全新流量驗證。

## cap 改變後的動態

- 同為第30秒 96→64：A 需約0.947秒，B 需約2.744秒，reserved cost 才首次降至64以下。這是首次跨越上限時間，不是已驗證的 settling time。
- B 第60秒 64→48 時，reserved cost 已只有43.28，因此不需要先等待舊工作釋放到48以下。
- B 第90秒 48→96 後，reserved cost 約47.70→95.68，等待首 token 的數量在一秒內11→19。
- cap 是 admission 工作量限制；它不會取消已在執行的 requests，降低 cap 和減少實際負載不是同一個瞬間。
- A＋B 合計涵蓋四個 cap 等級間全部12種不同方向切換，但只有18個非初始切換；大量0.5秒資料列不等於同樣數量的獨立控制實驗。

## gateway 停頓

B 的兩個最大 monitor 延遲：第123.244→123.588秒約343.21 ms；第231.121→231.354秒約232.86 ms。這兩段均與 loop-thread generation-2 GC 重疊，GC span 分別約264.59與247.78 ms。第一段 overdue 與該 gen-2 GC 約263.68 ms 重疊；第二段約176.22 ms。這是時間重疊證據，不是把整段停頓排他地歸因於 GC。兩次停頓沒有使這次 load/control fidelity 超過宣告門檻。

## 第一版 forecast baseline

固定使用現有程式：ridge alpha=1、訓練平均值補缺、訓練標準化、預測下限0；0.5秒 grid、1／2／5秒 horizon、1／5秒歷史視窗、engine 最大 age 3秒，不加 E2E deadline。每次600列，合計1,200列；鄰近列與標籤視窗重疊，不是獨立樣本。時間切分使用各 run 前2/3可觀測標籤訓練，後1/3評估，排除跨切分才得知的標籤。另做 A 訓練→B 評估與 B 訓練→A 評估。未使用 C，未呼叫正式 acceptance。

下表為時間切分後兩個 run 合併評估；百分比代表 MAE 相對 persistence 基準降低多少，負數代表較差。

| 預測目標 | 1秒 | 2秒 | 5秒 |
|---|---:|---:|---:|
| 等待首 token 數量 | +39.3% | +50.9% | +12.1% |
| 到達 cohort 的 TTFT 達標完成數 | +42.6% | +44.1% | -4.0% |
| 到達 cohort 已接受但 TTFT 超時數 | +48.8% | +56.6% | -9.6% |

在 cap 剛變動附近的5秒預測更弱：達標數的誤差比 persistence 高約30.3%，已接受但超時數高約43.9%。因此不能只看1–2秒表現，就宣稱5秒 MPC horizon 已可用。

跨 run 驗證中，2秒等待首 token 數量的 MAE 降低約59.6%，TTFT達標數約48.7–58.2%，已接受但超時數約67.7–70.6%；但5秒的達標數優勢在兩個方向只有約6.1%與22.5%，對訓練／測試切分敏感。兩個方向仍使用相同到達 trace。

## 控制模型的限制與額外診斷

- 現有控制輸入使用未來實際套用的 cap 平均值／終值，報告明確標為 realized_control_conditioned。這可用於辨識，但線上 MPC 只能使用當下計畫的控制與明確的延遲假設。
- 5秒等待首 token 模型的 cap 係數總和在時間切分與兩個跨 run 切分均為負，與原先簡單方向假設不同。可能涉及狀態、延遲及相關特徵；這不是較高 cap 必然降低 backlog 的因果證據。
- 額外開發 ablation 保留所有當下特徵，只把兩個未來 realized cap 欄位設為0後重訓。在 cap 變動附近，2秒達標數 MAE 由4.474變4.445，超時數由6.169變5.942；5秒達標數由9.377變9.200，超時數由9.521變8.608。移除未來 cap 欄位沒有惡化這些預測，顯示短期預測準確不等於已學好不同 cap 決策的效果。這是探索性開發診斷，不是預先宣告的最終驗收。

## 初步下一步（後續實測更新見下節）

先不租新 GPU，也不使用 C 調模型。在 A／B 上分開描述尚待 prefill 的 input token 工作量與已開始 decode 的工作，以及工作年齡與 cap headroom；這些可從 accepted／first_content／completed 等既有事件重建，但不等於精確的 GPU 工作量。建立以當下 planned cap 和延遲假設為輸入的版本，用時間切分與跨 run 驗證比較，檢查 cap-change 附近的誤差與控制敏感性。可以平行搭建 MPC 求解器介面或 shadow 記錄，但先不接真實 actuation。待模型與規則固定，再宣告並收集一次 C；之後才另做 closed-loop goodput 比較。

## 重現與來源

分析程式使用同一個 runtime commit 的 identification_features.py、forecast_baselines.py 與其依賴。未改原始 capture；無 --allow-unqualified、無測試資料、無 E2E analyst override。此分析環境不是 rental host 的 native CPU acceptance 環境；476項通過證據來自上傳的實際主機紀錄。

```bash
python identification_features.py <A根目錄>/results/shared-client-a-r1 <B根目錄>/results/shared-client-b-r1 --output-dir features
OPENBLAS_NUM_THREADS=1 python forecast_baselines.py features --output-dir forecast
```

A archive SHA256：`10abf26c841f32476a3fd8b25d41752158cdffa3464596d879b495a6cdffa8e1`。
B archive SHA256：`c4d5a3df4bef58d316ee14c781a4fc919ddedd8c168e2d61876fda3d4865f64b`。
B collection declaration SHA256：`6610d3839bb16db0c8b18e90dc11718d9adc2ae7f56ed1be82bbe024e6a4625f`。


## 進一步分析：階段特徵、planned cap 與模型結構

此節為收到「還了，現在來分析」之後的追加開發實驗。C 仍未執行、未讀取，沒有正式驗收結果。所有變體沿用相同 A/B、相同時間切分與跨 run 切分，沒有重新租用或跑 GPU，也沒有改 gateway/runtime 程式。這些是探索性比較，不是逐一獨立的確認性檢定；不能從其中選一個最佳分數就宣稱泛化成立。

### 已測的假設

1. **缺少 prefill/decode 階段資訊？** 用當下已發出的 lifecycle 事件，重建尚未 first content 的估計 input tokens、reserved cost、等待年齡，以及 streaming 的 input/output-cap 總量與 first-content 年齡分布。它們是 gateway 工作階段代理，不是 per-request GPU prefill/decode 執行追蹤。每列的 pending cost + streaming cost 與原始 reserved cost 對齊，streaming 年齡分箱總數與原始 streaming 數對齊。另加入1/2/5秒過去狀態與事件速率；沒有讀未來事件。加進現有 ridge 後，5秒 goodput 預測沒有改善，因此「只補這些特徵即可」不成立。
2. **是否主要依賴偷看未來實際 cap 套用時間？** 改成當下已知的 current cap、預先宣告 schedule 與固定50 ms actuation delay 假設，不讀未來 applied_at。這是已知 scheduled experiment 下的 planned-input 對照，不是已實現 MPC policy。5秒 MAE 與原版差異不到0.001 request；在本次30秒 holds、準時套用的資料上，這個時間差不是主要誤差來源。
3. **版本計數器是否造成時間外推？** 移除 budget_version。它是行政序號，並非直接代表服務工作量，線上控制長期運行會超出1–10訓練範圍。移除後5秒達標數 MAE 由7.49改善至6.98，但優於 persistence 的幅度只有約3.1%，超時數仍近乎沒有優勢。這是應排除的特徵風險，仍不足以解決主要問題。
4. **若知道實際未來流量摘要呢？** 僅診斷性地提供未來5秒 actual offered count 和 input-token 總和。它們不是可部署的當下特徵，不列入線上候選。達標數 MAE 改善到6.75，但已接受而超時數仍為8.29，高於 persistence 7.69。這只能說該 ridge 加入兩個未來摘要仍未解決問題，不能推論未來流量不重要。
5. **線性模型是否漏掉交互作用？** 用固定、未調參的 HistGradientBoosting 基準，以及另一個 ridge 的 state×planned-cap 與 cap² 變體。前者改善 backlog，後者也改善部分 backlog／超時誤差，但5秒 goodput 仍沒有穩定優勢。沒有宣稱新模型合格。

### 5秒時間切分結果

數字均為 pooled MAE，單位 requests，越低越好；評估各382列，鄰近列重疊。除第一列與標示診斷變體外，候選用當下已知的 planned cap 加固定50 ms delay，不使用未來實際套用時間。

| 變體 | 等待首 token 數 | TTFT達標數 | 已接受但TTFT超時數 |
|---|---:|---:|---:|
| Persistence 基準 | 4.14 | 7.20 | 7.69 |
| 原始 ridge／realized cap | 3.64 | 7.49 | 8.42 |
| Ridge／planned cap | 3.64 | 7.49 | 8.42 |
| 加階段特徵 | 3.71 | 7.81 | 8.52 |
| 階段＋過去歷史 | 3.63 | 7.56 | 8.18 |
| 移除 budget_version | 3.72 | 6.98 | 7.72 |
| 非線性基準＋階段＋歷史 | 2.96 | 7.93 | 6.96 |
| 階段、移除版本、state×cap／cap² | 3.24 | 7.50 | 7.58 |
| 真實未來流量摘要（僅診斷） | 3.65 | 6.75 | 8.29 |

非線性對照使用 sklearn 1.8.0 的 HistGradientBoostingRegressor，learning_rate=.05、max_iter=100、max_leaf_nodes=7、min_samples_leaf=50、l2_regularization=1、early_stopping=False、random_state=0；沒有隨機切分驗證或超參數搜尋。交互作用變體固定 ridge alpha=1，增加 planned cap mean × pending input tokens／streaming cost／streaming mean age／headroom，以及 cap mean²，未做參數搜尋。

### 為什麼「預測得準」還不等於能用來選 cap

原始 ridge 是狀態項加上 cap 平均值／終值的線性加項。在其他狀態固定時，cap 的邊際斜率不隨工作狀態改變；它無法直接表達「在某狀態加 cap 提高 goodput、在另一狀態卻降低 goodput」這類交互作用。約束仍可能產生不同控制決策，但不能因此說模型已學會狀態相關的 goodput 曲線。

這個斜率也不穩定：原始5秒 TTFT達標模型，在時間切分中，每增加一單位 held cap 的線性效果約−0.065 request；A訓練→B評估的模型約+0.062，B訓練→A評估約+0.030。這是相關模型的係數，不是因果效果；正負方向依訓練資料而變，不能直接交給最佳化器追逐。

誤差有時是系統性偏差：例如 B 的 t=268秒，原模型預測隨後5秒 arrival cohort 約5.70個 TTFT達標，實際為32個。B 後段有持續低估達標數、過度估計超時數的情形。只用平均輸入長度並不能確定解釋這個偏差；兩次到達 trace 相同，輸入長度組成仍有變化，state、選擇哪些 request 接受、服務階段與 host 也互相影響。

A/B 共有18次非初始 budget 切換；600列/次的特徵表不是600次獨立控制激勵。時間切分的訓練段也沒有涵蓋所有切換方向，例如兩次64→96都在270秒，而時間切分訓練在200秒前結束。跨 run 結果相對好，不能消除這個控制覆蓋與時間外推問題。

### 更新後的實作順序

1. 先修模型輸入契約：行政 budget_version 留作 telemetry，不作預測特徵；planned-control 輸入與 actuation-delay 假設明確分開記錄；actual applications 只作證據。保留原資料、原 baseline 與此開發比較。
2. 下一版先嘗試短步長的工作狀態轉移／服務量模型，再檢驗多步 rollout。使 admission、待 first content、streaming、completion／release 之間有守恆與非負約束，讓 goodput 與已接受而超時的數量互相一致。這是待測的候選設計，不保證比本次直接5秒迴歸好。
3. 現有 raw engine observations 已有 prompt_tokens、generation_tokens、queue/prefill/decode histogram 與 counters；目前 baseline 只用了 waiting/running/KV 等少數欄位。可先用已在 t 前收到的樣本，處理 counter epoch/reset、age 與 unavailable，建立過去 token throughput／服務量特徵。aggregate counter 不等於 per-request GPU attribution，也不能無條件推得每個 request 的精確剩餘 tokens。這部分尚未實作／驗證。
4. 先用 A/B 檢驗1步與5秒 rollout、cap-change 附近誤差、候選 cap 敏感性及跨 run 表現。不要只因1–2秒預測不錯，就把MPC horizon縮到2秒後宣稱問題解決；既有decode工作可延續更久。
5. 可以並行建立 MPC solver／shadow 記錄介面；待模型有可重現的控制反應證據，再進行 actuation 比較。若 A/B 的18個切換仍不足辨識，下一次 GPU 應補多次、時間位置平衡的 development 激勵，再收保留 C；此刻不需要重新租機器。

總結判斷：測量可靠且已排除幾個簡單補法；尚未定位唯一根因，也尚未驗證可用的5秒 goodput 控制模型。C 保持保留。下一步是具守恆關係的短步狀態轉移原型與既有 engine counters 的因果特徵化，而非持續盲目增加回歸特徵。
