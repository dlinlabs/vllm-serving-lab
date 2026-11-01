# Baseline04：80 個原生實驗與四模型技術共識

## 裁定與完成範圍

**80／80 個正式實驗已完成，四模型獨立分析與各自三方交叉批判已封存。**
400 份原始 HTTP／events／closed Gateway artifacts 已逐檔核對 SHA，
四個每 seed archive 解壓後再次核對原始 bytes。沒有在十小時通知時
截斷矩陣，沒有用部分結果挑選有利策略或替換已曝露 seed。

本批只執行 fixed-count16／32 與 fixed-cost16／32，**沒有執行 MPC、
OneMove 或 reactive 控制器**。完成原生實驗與四模型共識，不等於 MPC
勝出，也不是所有 batch-aware MPC 不可行的負面證據。

full04 從 `2026-10-31T19:10:30.794309Z` 至
`2026-10-31T22:01:12.795786Z`。80 個不重疊 offered windows 各 120 秒，
新有效時間 9,600 秒；加原已封存的 26,880 秒，累計
**36,480 秒＝10 小時 08 分鐘**。所有 smoke 排除；本輪 baseline audit
沒有重新讀取舊 Source03／Cal／Val outcomes 來重算先前時數。

既有授權 RTX4090 Pod 上的 owned vLLM 61440 與 children 61596／61597
已退出，serving port 關閉，GPU compute processes 為空。Pod 保留；
**provider billing 停止未驗證**，不能把停止推論稱為停止供應商計費。
只操作 pr32，沒有新增 GPU、worker push／gt done bypass 或 main merge。

## 實驗與指標定義

四個 Train seeds 181045–181048 × 五 cells（p128-m128、p128-m256、
p4096-m128、p4096-m256、mixed）× 四固定策略＝80 episodes。
來源為未改產品的 Source144、既有 Qwen3-4B、同一 engine／clock domain，
collector16e4／classifier18d 與 protocolf5e6 封存後才曝露完整矩陣。
offered-window observer 對所有策略使用相同 metrics＋native；
cost state 只在初始／最終 drain 讀取。

Natural SLO-good 的名稱沿用 collector，但其終端定義是 HTTP complete
且 API `finish_reason=stop`，output tokens >1，scheduled-to-first-content
TTFT 在 [0,0.5] 秒、(terminal-first-content)/(output-1) 在 [0,0.05] 秒。
這是 aggregate terminal TPOT proxy，不是逐 token ITL，也不是特定 EOS
token 或 GPU 實體釋放的證明。HTTP attempt deadline 明確為 30 秒。
拒絕、length、timeout 與其他失敗不移出 assigned cohort。

兩個 estimands 必須分開：**cohort-good** 允許 assigned 請求在排空期
完成；**in-window-good** 再要求 terminal 不晚於 offered window 結束。
兩者除以 offered time，不包含啟動／drain 的 wall time。觀察到 917 個
terminals 在窗口後，最大 overhang 6.782542 秒，其中 555 個仍滿足 SLO；
不能悄悄換指標或把排空期 successes 當成窗口內吞吐。

## 全矩陣結果

全部 112,652 個 assigned HTTP UUID 的終端命運為 77,064 complete、
35,588 rejected、0 other、0 capped；complete 均為 API stop。cohort-good
64,188，in-window-good 63,633。每策略 offered 數均為 28,163、offered
時間均為 2,400 秒。

| 策略 | 完成 | 拒絕 | Cohort-good | In-window-good | Cohort good/s | In-window good/s | In-window／全部 assigned |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-count16 | 18,945 | 9,218 | 15,733 | 15,596 | 6.555417 | 6.498333 | 55.38% |
| fixed-count32 | 22,724 | 5,439 | 17,040 | 16,900 | 7.100000 | 7.041667 | 60.01% |
| fixed-cost16 | 14,761 | 13,402 | 14,233 | 14,117 | 5.930417 | 5.882083 | 50.13% |
| fixed-cost32 | 20,634 | 7,529 | 17,182 | 17,020 | 7.159167 | 7.091667 | 60.43% |

cost32−count32 的 cohort count 差為 142、平均 +0.059167 good/s；
in-window count 差為 120、平均 +0.05 good/s。四 seed 差分別是
`[-19,+25,+62,+74]` 與 `[-27,+17,+60,+70]`：兩者都是三正一負，
**不是普遍、穩定或統計顯著的 superiority 宣告**。

排名會隨 cell 反轉：短提示詞 count32 有較高窗口 good 數；長提示詞
count 策略的 SLO-good 極低，而 cost 策略仍有非零 good。p4096-m128
每策略 offered 3,648，in-window-good 分別為 count16=3、count32=0、
cost16=657、cost32=405；全部 assigned 比例是 0.0822%、0%、18.01%、
11.10%。657/941=69.82% 與 405/1299=31.18% 是**完成者條件比例**，
不能叫作 all-assigned goodput 或延遲保證。cost16 在此 cell 勝過 cost32，
本報告不事後選出一個「最強固定策略」。

## 四模型方法與交叉批判

| 模型／原生 worker | 可核對的貢獻 | 資格邊界 |
|---|---|---|
| Claude Opus 5.5／amber | 全矩陣 count／cell tables、in-window 對比、三方 cross-check | episode percentile 的跨 seed 摘要，不等於 pooled percentile；原 2/2 敘述已改正 |
| GPT-6 Astra／jade | 全 80 計數 replay、全部 paired contrasts、cohort／window 分離、executor receipt | SD／range／leave-one-out 都是描述量，不是 CI；wire payload 完整 bytes 未驗證 |
| GPT-6.1 Sol／flint | 全 400 hashes／archives、80 inventory／summary replay、Native visibility 與 closure audit | HTTP closure 不等於完整 engine visibility、owner authentication 或 physical release |
| Gemini 3.8 Flash／pearl | stored-summary 重聚合、HTTP timing、三方交叉批判 | 不是額外的獨立 400 檔 rehash／SLO count replay；120 既有 CPU tests 不提供原生統計資格 |

Mayor 已重現四份完整 canonical raw，逐位元相同。Opus／Astra／Sol
對全部 80 episodes 的七個共同 count 欄位完全相符；Gemini 的四策略
與 16 seed-policy aggregate tables 也相符。四份各自三方 critique 均
核對封存 input hashes 後執行，不是先互相閱讀再假稱獨立，也不是用
多數票替代來源或方法驗證。

四個 arrival/payload seed blocks 才是名義實驗單位，不能把 80 episodes、
20 cell groups 或 112,652 requests 當作獨立樣本。政策順序在 45／47
與 46／48 重用兩個鏡像版本，不是每 seed 獨立隨機化；完整 outgoing
message／token-ID bytes 沒有被 capture 封存。匹配 tape、recipe 與觀察到的
geometry 不足以把對比提升為同狀態因果反應。

## 保留的錯誤與觀測限制

Opus 原本把 sign split 說成 2/2，已由原始 seed 向量改正。Astra 原
embedded source 少了 stdin executor 與 final-LF convention；後補 exact
source／write-intercepted executor，原 raw 不變。Sol 初始 64 個 CPU
assertions 失敗，包含 40 packaging assumptions、24 literal-zero assumptions；
**不是 24 次實際 native drain 失敗**。canonical empty-mark contract
原樣接受 4.44e-16 至約 4e-15 scalar 殘值，沒有 rounding／TTL 放寬。
Opus 初次 critique 的過嚴 display tolerance 失敗亦保留敘述；其已覆寫的
初次 untracked raw 不冒稱已完整保存或重現。不存在單一零失敗歷史。

Gemini 原有 cost=memory、physical lease、延遲保證、穩定 superiority、
完整 engine UUID、特定 EOS、steady state 與 body geometry 敘述錯誤，
已列入原 seal／errata／crosscritique。交叉批判另外指出 conditional
denominator、p50 誤標 p95、manual SD 數值錯誤。本共識只採 canonical
計數及正確定義：p4096-m128 count32 的 p50 TTFT/TPOT 為
5.9106s／85.1ms，p95 為 6.3969s／97.5ms。四 seed in-window count
的 sample SD 為 cost16=39.894653、cost32=84.896015、
count16=11.916375、count32=90.303193；cost32−count32 SD=44.414712。
這些是描述性 count SD，不是延遲 SD、CI 或安全邊界。
Gemini 最終 report-only accuracy seal
`30886c55ffe2565d0e35fb6f401495cf5c9b2abd` 已修正 quantile 標籤與 sample
SD，原始計數與 canonical raw 不變；先前錯誤與交叉批判仍保留。

HTTP tail `terminal_at−last_content_at` 的 median 27.879676ms、
p99 323.195314ms 只描述客戶端串流關閉尾部；不證明 GPU 或 Gateway
內部實際釋放時刻。Native samples 中有 27 個 admitted complete 從未
visible、229 個不在 offered window visible；71,054 frames 中 69,492
async pending，26 個 inter-sample gaps 超過 500ms，最大 gap 704.396ms，
雖然最大 recorded receive age 約 182.502ms。**sample age 不限制 gap，
HTTP UUID closure 不提供完整 engine suffix／physical release 證明**。

count 與 cost 的 16／32 是不同單位；cost 是原 public cost stock，不是
GPU 記憶體單位。count observed outstanding peaks 18／34 包含設定的
兩個額外 outstanding margin；這不是 observed waiting-only cap=2，
實際 waiting peaks 可為 4／9。cost branch 不能冒稱相同 count32 限制。
延遲差的具體 KV／bandwidth／排程機理仍未辨識。

## MPC 與後續資格

本批建立完整可重現的固定策略對照，沒有授予 calibrated／physical-pilot
或 closed-loop MPC permission。H14 原始完整 fitted-state checkpoint
未恢復；原 Train／fit-run receipts 不是 learned-weight identity，不能
refit 或以 lossy counts／quantiles 替代。H16 bundle735 的 CPU layout
缺口已閉合，但 actual source/runtime Q1–Q5、雙 bundle identity 與
written owner closure 仍未資格。41／42 與 43／44 未因本共識而授權曝露。

完整 benchmark 與這份 baseline 共識已完成；**更大的 MPC 研究尚未證明
勝出**，也不能把上述資格缺口偷換成所有 MPC 不可能勝出的結論。

## 證據索引

完整 80 episodes／400 artifacts：
`results/mpc-native-v3/review-evidence/nv3-026-complete-native-baseline04-receipt.json`。
四份獨立分析、重現結果與歷史 errata：
`results/mpc-native-v3/review-evidence/nv3-027-four-independent-baseline-reviews-receipt.json`。
四份三方 crosscritique 與最終共識：
`results/mpc-native-v3/review-evidence/nv3-028-four-model-baseline-consensus-receipt.json`。
各 receipt 列出配套 archive、完整 SHA、bytes 與檔案 manifest。
