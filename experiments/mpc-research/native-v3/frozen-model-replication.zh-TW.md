# NV3-010：未改動v5在新Train6／7的replication

對應已預註冊的 `experiment_id=NV3-009-frozen-birth-response-replication`；
保留原始spec名稱與commit，不重新命名歷史證據。

**五秒預測改善已在另外兩個complete blocks重現，仍不是MPC效能驗收。**
Preregistration commit `517f9288085b28c5bc28220d00dd0b332ab30647`
先固定source、fit、metrics、masks，再freeze／score181006與181007。
v5 source SHA保持
`b1d83f7af2956d51b5f9d9b014ecff50b2b3a6915c5f10cdb52f034ed8a920b5`，
只使用既有diagnostic的explicit eval-block參數，沒有改model或refit已用blocks。

Frozen input：20episodes／40verified JSON與event streams／2400offered秒，
index SHA：
`6ee2db3e6fd8346b9553d434343ca850e07cb461ace5e22f53911ccd8eca0875`。
1111個scored origins是correlated observations，不是1111個獨立實驗。

| 新development block | v5 G5 MAE | 最強同-mask comparator | v5 paired G5 MAE | 零差異 |
|---|---:|---:|---:|---:|
| 181006 | 4.4234 | progress5.9191 | 4.3661 | 6.0331 |
| 181007 | 4.2260 | additive5.6450 | 4.0481 | 6.8257 |

兩個blocks均通過預先固定的描述性replication rule，
相對各block最強comparator的level誤差降低約25%。
Paired sign agreement約0.768／0.736，
但兩個arms狀態不同，不能當same-state action效應。
181004–181007都已USED development；後續調整不能再把它們稱作盲測Final。

## 同時重現的未解問題

五秒Good bias−1.375／−1.959，rejection bias+2.380／+3.034。
名義80% Good scenario inclusion只有0.577／0.614，
offers約0.468／0.504、rejected約0.513／0.547。
沒有校準成功、selected-policy risk guarantee或Native MPC improvement。
新的forecast訊號足以選作研究candidate，不足以跳過真實閉環比較。

## 發現並隔離的descriptive aggregation bug

凍結v5的`mean_truth`用episode ID作dict key，把同episode之前的origin覆寫，
因此只平均每個episode最後一個origin，不是所有scored origins。
**Primary MAE、bias、band inclusion與paired error使用append lists，沒有這個覆寫，
本次replication結果不受影響。**

Mayor以read-only return-frame capture保留全部1111個origin的truth／predictions，
不修改凍結模型或其eligibility，並另外計算corrected full-origin truth。
本次G5 mean33.2791，其中current cohort6.2858、later IDs26.9933；
R8 raw的mean-truth數字應帶這項限制，不能把它當完整平均capacity。
Original raw未重寫成「修正後模型」；Opus下一輪負責非predictive aggregation
regression與明確boundary-sensitivity，不能用改summary救分數。

## 下一階段工程

已建立[實驗性MPC architecture ADR](experimental-mpc-architecture.zh-TW.md)：
共同5秒window與32scenarios，full32plans對照strong One-move4plans，
最大化expected API-stop SLO-qualified Good count。
這是mean-only research candidate，沒有chance/stability／production permit。

四個原生workers R9已執行：
release lower／upper sensitivity、frozen-kernel solver與deadline量測、
真實capture＋guarded-actuator mock pipeline，以及source-bound online model-input adapter。
不換horizon／weights、不操作Final、不修改live729。
GPU長時辨識仍持續；十小時與真正MPC／Fixed／One-move closed-loop結果尚未完成。
