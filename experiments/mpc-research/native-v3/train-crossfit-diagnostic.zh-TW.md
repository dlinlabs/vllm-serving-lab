# NV3-003：凍結模型的兩個 Train block 交叉預測

**真實 GPU traces 上的 batch-aware marginal release 預測有初步訊號；
尚未證明 joint dynamics、MPC Goodput 或 held-out 泛化。**

## 已執行的固定實驗

Collector source 為 `6c11fe7e683439f379f0dfdbd21b05f5873b2c90`，
Opus R3 module SHA256 為
`98edb2df3b0bc9f5f4c8e9bedc0df5c4e08070f65408d32f63666004dc55ade3`。
只讀取 complete Train seed blocks181001、181002，每個 block 皆包含
五個 cells 的兩個 paired arms。20個原生 episodes 的 JSON 逐檔 SHA
核對，原始資料持續 off-Pod 保存。沒有讀 Calibration、Validation 或 Final。

固定兩個方向：以181001 fit、對181002 score；再反向執行。
沒有改 features、support、horizon、threshold 或 weights 以挑有利結果。
共解析1,210與1,214個 causal grid origins；2338個非零、in-support origins
獲得 score，69個 zero-reserved origins 與17個 OOD abstentions 分開記錄。
這是 **Train block cross-fit**，不是 final held-out 測試，也不是把 origins
當2338個獨立工作負載樣本。

## 區間真值的誤差，不是信賴區間

使用凍結 R3 的 score：每個 origin 的1／3／5秒 released original-cost
absolute error，除以該 origin 的 reserved cost；先平均 horizons，再依
episode、seed block 等權平均。snapshot disappearance 的 interval truth
提供最佳／最差可能 error。下表上下界是 **label-censoring error bounds**，
不是統計 confidence intervals 或 uncertainty-coverage 保證。

| Fit → score Train block | train-fraction baseline | fluid-history baseline | batch-aware marginal |
|---|---:|---:|---:|
| 181001 → 181002 | 0.0909–0.1107 | 0.1600–0.1812 | **0.0332–0.0519** |
| 181002 → 181001 | 0.0897–0.1076 | 0.1502–0.1695 | **0.0351–0.0523** |

兩折1秒 fit 均在8次迭代收斂；3秒為12／11次；5秒則是 degenerate
constant fit，沒有 exact unreleased labels。scoring平均包含較平凡的長
horizon target，不能把表中約3%–5%的數值描述為1秒獨立 MAE、
tail-risk calibration、recursive multi-step state error 或完整 serving 預測誤差。
CPU parsing／fit／score 合計14.59秒；**不是線上 solver latency**。

## 裁定與下一步

在這兩個相同部署、相同 workload families 的 Train blocks，batch/history
條件訊號對 original marked release 有預測價值；它提供繼續辨識的理由，
不是 stable MPC win。只有兩個 block，沒有可信的泛化 CI 或 OOD workload
結論。OOD abstentions保留，不刪除不利資料。

目前模型仍缺 future births/admissions、joint terminal good/tail、共同 batch
variation 與真正 free-running transition；不能將 release MAE 改善換算成
MPC throughput 改善。下一輪重點是 joint stock-flow 而非盲調 cost weights。
當真正 closed-loop protocol 凍結時，另固定主要 Goodput 邊界、practical
margin、獨立樣本與 multiple-baseline 推論；不以這份 Train 結果調 Final。

完整數值、逐檔資料 SHA與原始 runner 保存於
`results/mpc-native-v3/train-crossfit/`：

- `pr33-native-v3-r3-two-train-block-crossfit.json`：
  `16cfba893a428979431c30195c580e98cf99f9aabd8ca0f6c79b0059921dd693`
- `pr33_native_r3_train_crossfit.py`：
  `f14f83088486e6ea1dfef8566471478db5b6ff7c26bd8c3b4fa64866ba9b5bc0`

Runner 是保存的固定實驗 artifact，不是 production API。重現需將
byte-verified `pr33-native-v3-long01-replica` 和解封的
`pr33-native-v3-round3-frozen` 放在 runner 同目錄；輸出必須是 fresh path，
不可覆寫先前結果。公開 JSON包含原始20個資料 artifact的 SHA，
不把尚未完成的整個長時實驗當成已封存完成或十小時驗收。
