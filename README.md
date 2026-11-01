# vLLM Serving Benchmark Lab

**研究如何在大量請求湧入時，讓單張 GPU 上的語言模型服務保持可用。**

這個專案使用 vLLM、Qwen3-4B 和相容 OpenAI API 的 Gateway，
比較不同的請求接納策略。重點不是把 GPU 使用率拉滿，而是：
**有多少請求能及時完成？為了維持速度，又拒絕了多少請求？**

## 它怎麼運作？

```text
測試請求 → Gateway 決定接不接 → vLLM 產生回答 → 記錄速度與結果
```

Gateway 就像模型前面的交通管制員。忙不過來時，先拒絕部分新請求，
避免所有人一起排太久；這是延遲與拒絕率的取捨，不是增加 GPU 算力。

| 策略 | 白話說明 |
|---|---|
| 不限制（`baseline`） | 所有請求都交給模型。 |
| 按數量限制（`protected` / count） | 按請求個數給額度，長短請求都算一筆。 |
| 按估計工作量限制（`cost_aware` / cost） | 用輸入長度與要求的輸出上限估算成本，總額度滿了就拒絕新請求。 |

**count16 和 cost16 不是相同容量。**前者按個數計算，後者按估計成本計算；
cost 也不是 GPU 記憶體 GB 或實測使用秒數。降低 cost 額度不會取消已接納的請求。

## 現在做到哪裡？

**固定策略比較已完成；MPC 的效果尚未驗證。**

最新一批在同一個 RTX 4090 後端完成 **80 個正式實驗**，
比較 count16、count32、cost16、cost32，涵蓋四組請求種子與五種負載。
112,652 個請求中，77,064 個完成、35,588 個被拒絕，沒有其他終端失敗或輸出截斷。
四個不同模型已完成獨立分析與交叉檢查。

結果沒有一個普遍贏家：長短請求會改變策略排名。
cost32 的平均視窗內合格完成率只比 count32 多約 **0.05 請求／秒**，
不能據此宣稱統計顯著、所有情境都更好，或 MPC 已經勝出。

「合格完成」指符合本批預定的延遲要求；視窗內計數與允許排空期完成的計數分開報告。
拒絕、超時與失敗不會從評估中刪掉。

[閱讀完整結果、指標定義與限制](https://github.com/dlinlabs/vllm-serving-lab/blob/08b933e65ad2b5054beca1c4651296c9a6684be8/experiments/mpc-research/native-v3/baseline04-four-model-technical-consensus.zh-TW.md)

PR33 已合併到 `analysis/cap-miss-causes-v1`，不是 `main`。
首頁所在的 `main` 提供基礎服務與測試工具；
[PR33 的完整研究程式與資料](https://github.com/dlinlabs/vllm-serving-lab/tree/08b933e65ad2b5054beca1c4651296c9a6684be8)
另有固定版本可供查閱與重現。

## MPC 是什麼？

MPC（模型預測控制）會先預測未來幾秒的狀態，比較多步額度安排，
**只執行第一步，再用實際回饋重新計算**。
它不是固定額度，也不是單純看到佇列變長就調整。

接下來要先驗證真實狀態、額度修改與回饋鏈路，封存可驗證的新模型，
再和固定策略、反應式控制及單步規劃公平比較。
目前不能把離線預測、CPU 測試或一次控制成功當作 MPC 效能證明。

## 如何開始？

先閱讀報告不需要 GPU。在 Linux、Python 3.12 環境中，可執行 CPU 測試：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu-tests.txt
python scripts/run_linux_cpu_tests.py
```

這組步驟不啟動 vLLM、不下載模型；CPU 測試通過也不代表 GPU 實驗已通過。
若要實際啟動推論服務，請另看 [V1 環境設定](experiments/v1-setup.md)，
並先確認 GPU、模型與費用。
V1 記錄的核心版本是 vLLM `0.30.0` 與 PyTorch `2.13.0+cu130`，不是完整歷史 lockfile。

## 程式與資料在哪裡？

| 位置 | 用途 |
|---|---|
| [gateway.py](gateway.py) | 接納控制與串流代理。 |
| [run_experiments.py](run_experiments.py) | 產生負載、比較策略與保存結果。 |
| [experiments/](experiments/) | 實驗方法、設定與操作文件。 |
| [results/](results/) | 原始資料、分析與重現工具。 |
| [最新研究資料](https://github.com/dlinlabs/vllm-serving-lab/tree/08b933e65ad2b5054beca1c4651296c9a6684be8/results/mpc-native-v3) | PR33 的模型研究與原生驗證證據。 |

逐階段的數據、架構與重現命令保留在 [舊版完整 README](https://github.com/dlinlabs/vllm-serving-lab/blob/aea7694a0af2d1dadc883b4b9e33d1a9bdb61098/README.md)。

這是一個研究實驗室，不是已驗證的生產環境自動控制器。
