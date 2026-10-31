# NV3-007：Native progress 預測訊號與尚缺的 plan-response

**預測有進展，MPC 仍未驗收。** 四模型完成同一核心問題的R6獨立提案，
全部交付凍結後才啟動R7互審。沒有用模型共識代替GPU閉環結果。

## 已執行、可追溯的證據

| Backend／Worker | R6完整local commit |
|---|---|
| claude-opus-5.5／amber | `24746f62a6c56e270e1b6f530fcaaaf22ee4e158` |
| gpt-6-astra／jade | `01846136db77720cfc84a7eece0d421d90436435` |
| gemini-3.8-flash／pearl | `27f30683d726c22bf62fcf175d2e8b013a18b329` |
| gpt-6.1-sol／flint | `df00f30f5552f9d2299615930b2c612f82e070e8` |

Mayor逐檔核對12個source/report/test files，實際執行
**204 tests／139 subtests**，使用必要的SHA-pinned舊fixture。
R6 source archive SHA256：
`a70e13aa7c9276c032024f10c6a5f9224f37c2a937775941fb783b0a2231b851`。

Sol的`7787a7d`是auto-WIP checkpoint，當時report仍dirty，
不當成完整交付；等到`df00f30`三個owned files皆committed才封存。
Lifecycle idle或看到新commit都不能單獨證明工作完成。

Mayor另外執行凍結Opus的
`--replica-progress-diagnostic pr33-native-v3-first-three-complete-train-blocks-frozen`。
Train181001／181002 fit，已使用的development181003評估；Final未看。
Worker raw SHA為
`016456f6eab563bc2f0573ba2b0f90db6789fd3a45a9c7660019de780ae8ed7a`，
Mayor raw SHA為
`b987310fe835476c8d9629cfe51a8118eb01f0b31271b7368c6361a376e480b2`。
兩份JSON並非byte相同：只有8個scalar summation roundoff不同，
最大absolute difference約`5.56e-17`，其餘結構、counts與fields完全相同。
保留兩份原始輸出，不重寫成同一SHA。

## 已確認：描述性預測改善，不能換算為Goodput提升

| Good-count forecast MAE | History mean | Aggregate hybrid | Progress hybrid |
|---|---:|---:|---:|
| 1秒 | 2.7014 | 1.4158 | 1.0666 |
| 3秒 | 5.6858 | 3.7025 | 3.4472 |
| 5秒 | 8.7573 | 6.0043 | 5.9029 |

5秒progress hybrid相對history的MAE降低約32.6%，
但相對更強的aggregate hybrid只改善約1.7%。
不能把全部改善歸因於Native progress：
主要差異還包括不同的direct/reduced-form future-birth處理，
而不是把R5 recursive模型原封不動多加一個state。
不同模型的可用origin／support／missingness仍須在互審中逐一對齊，
不能用選擇性可用樣本宣稱普遍勝出。

原始cohort的5秒MAE約0.1235，**不是整個5秒Good-count MAE**；
未來births占主要部分。Oracle future-arrivals與future-birth診斷另列，
不當成可以上線取得的輸入。

此結果來自一個已使用的Train development block、相關paired arms，
沒有泛化CI、conditional coverage或性能改善保證。
原始R5 recursive五秒9.84／history8.76負面結果仍保留。

## 尚未成立：由預測器直接宣稱MPC

R6 current birth model混合admission與service的reduced-form ridge，
不是可以直接拿來當GPU service law的模型。
它沒有完整candidate-plan conditional transition／coherent scenario kernel，
因此不能合法排序任意多步cap sequence。
原始JSON明列`mpc_result=false`、`calibration_result=false`、
`statistical_guarantee=false`。

Astra／Gemini的progress-aware hybrid與Sol的source-bound counterexamples
提供數學結構或可辨識性限制，尚不是fitted GPU kernel；
toy state反例不等於已在GPU重現的介入效果。
舊join遺漏`computed_tokens`不代表raw Native field不存在，
但raw frontend counter也不等於無噪聲、逐step GPU truth。

## 正在執行：兩個競爭架構的R7反駁

四模型已取得完整R6凍結來源，獨立批判以下未裁定方案：

1. Causal hybrid marked-progress kernel：cap只透過admission/state改變batch，
   再由prefill/decode service、EOS uncertainty決定未來events，
   保持count／original whole-cost／lifecycle一致。
2. Direct multi-horizon full-plan response：
   `F_N(x_history, U_future, common_exogenous_law)`，
   利用既有randomized budget sequence辨識完整五個1秒future inputs，
   不只讀`next_plan_budget`，不餵入已實現的future offers或EOS。

第二個只是Mayor提出供反駁的新假設，不是已實作或四模型已同意的架構。
必須處理planned-assignment ITT／實際ACK與timing compliance、
state/action positivity、constant與多步plans的support，
以及random-policy uncertainty能否移植到optimized policy。
不能把marginal residual inclusion換成所有candidate plans的風險保證。

兩方案均維持horizon1／3／5秒、sampling1秒、現有SLO/output語義，
不搜索weights，不把Calibration／Validation／Final用作新模型選擇。
驗收目標是可用且可反證的cap-plan response，而非再交付一個只有名字的MPC。

Native source729長時辨識與off-Pod resource/evidence monitor仍獨立運行；
planned44,400秒不是完成時數，十小時與閉環performance仍須實際驗收。
新CPU source沒有hot-replace正在GPU執行的source。
