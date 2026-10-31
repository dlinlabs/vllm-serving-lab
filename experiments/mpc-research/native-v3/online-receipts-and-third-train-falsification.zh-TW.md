# NV3-006：線上 receipt 介面與第三個 Train block 的獨立反證

**MPC 尚未勝出。** 本輪完成可重現的模型反證及 CPU 線上量測介面，
沒有修改正在執行的 GPU source `729b144`，也沒有把新介面稱為 Native 驗收。

## 已確認：修正外生 arrival 仍不足以修復五秒 Goodput

Opus R5 以 Train181001／181002 fit，在已使用的 development181003 評估，
保持 horizon、ridge penalty、scenario seeds、SLO 與 metrics 不變。
Mayor從其 local commit `52a2b69f8c20257c9592cd2152d18e37bf8d9250`
逐檔凍結，再實際執行：

```text
native_release_model.py --replica-exogenous-arrival \
  pr33-native-v3-first-three-complete-train-blocks-frozen
```

Mayor完整輸出與Worker輸出 **byte-for-byte相同**，SHA256：
`407a1b46c3b9acb61a309e38daa99c19760366c64d192f4d2253292326ddc123`。
30 episodes／30 streams已核對，failed181004不在資料內。
181003不是之後可再次稱為「新 held-out」的資料。

| 181003 free-running good-count MAE | 共同外生 arrival 候選 | 舊 joint | history mean |
|---|---:|---:|---:|
| 1秒 | 1.78 | 1.78 | 2.70 |
| 3秒 | 5.56 | 5.58 | 5.69 |
| 5秒 | **9.84** | **10.18** | **8.76** |

即使診斷時提供已實現的未來 offers，5秒 good MAE仍為8.78，
沒有優於history mean8.76。這個oracle不是可部署forecast，也不是controller。
因此「只修正arrival即可消除五秒誤差」在此development診斷未成立。

Reserved stock與TTFT-miss預測有訊號，但nominal scenario bands仍under-cover：
5秒外生offers inclusion約0.44、rejection約0.55，
TTFT-miss約0.49。TPOT-miss幾乎全零，風險仍不可辨識。
這不是獨立seed泛化CI、conditional-risk保證或MPC效能改善。

## 已確認：可用的 Native 狀態，不以猜測補欄位

真實frame的每個Native request包括`computed_tokens`、`output_tokens`、
`prompt_tokens`、`maximum_output_tokens`及request ID；
frame另有running、waiting、KV cache usage。
Gateway marks則有原始whole cost、admission／first／receive時間。

這提供研究token progress、batch composition及EOS hazard的可能性，
但不是已辨識的服務模型。Native與delivered token counters、async forward
overhang、physical EOS與API stop不可直接互換；
accepted-but-not-native-visible及兩次snapshot間born-and-released的工作
不能被當成零。原始reservation cost只在真正release時移除，
不能隨remaining tokens比例減少。

## 已確認：CPU publication hook，尚未 Native 部署

Source commit：`734d4593017744e745c041a5825468d2709d2e64`。
`mpc_lab/http_capture.py` SHA256：
`a8bbafd6a84f8a1581a2f0e0d0fe11a086c542e392a45c1466fa9071df93fb6c`。
Controller仍未啟動，GPU workload／source／flags維持原樣。

新增optional synchronous `observe(kind, receipt)`：

| Receipt | 實際邊界 | 不能宣稱 |
|---|---|---|
| `offered` | client attempt開始，既有`sent_at`邊界、callback publication time | 已送出實體wire bytes或Gateway已入場 |
| `first_content` | 實際收到第一個SSE content後立即publication | 逐token GPU／HTTP timestamps |
| `terminal` | 完成、rejected、timeout、failed或cancelled，當下已知row | API stop即獨立physical EOS proof |

等待排程期間取消仍有terminal receipt，但`sent_at=None`，
不能因此增加實際offered count。發生callback例外或誤用async callback，
只記`observation_error`類別、不回顯例外message，不把HTTP完成改成成功量測。
Receipt使用獨立copy，不容callback修改nested usage而污染原始outcome。
Consumer仍須實際stamp receive time、validate clock／source、
seal no-future cut及在receipt不足時撤銷MPC permit。

Mayor已執行102個coupled tests，涵蓋capture、actuator、contracts、
runtime、integration及既有Native hold／capture qualification。
第一個新測試曾錯誤假設fake clock呼叫次數；已改為驗證真實publication
先後與POST前可見性，而不是放寬任何SLO或時間門檻。

**沒有把舊final outcome row倒填為已存在的線上callback。**
新的source會改變implementation binding，不能沿用舊hash假裝相同runtime；
正在GPU執行的729仍凍結，後續新source須另行資格、source binding與calibration。
Sol已交付獨立receipt consumer介面，local commit
`ebf487af25ea82e8da740c1e3fb91348cd3a385c`；接入與Native權限尚未驗收。
四模型R5全部12個owned files已逐檔凍結，Mayor執行188 tests與134 subtests。
第一次未供應舊round fixture時27項明確skip，不能算通過；提供既有
`NV3_R1_FREEZE`、`NV3_R2_FREEZE`、`NV3_R4_FREEZE` SHA-pinned fixtures後，
全部實際執行通過，沒有刪除或更改檢查。
R5 archive SHA256：
`2af0d40ddd98672ab4a823a254aab49c66e1dcbe6a2a6df322e3c766b1d1c5d5`。

## 推論與下一個可反證假設

外生arrival修正不足，下一輪優先辨識服務／state transition誤差，
而非擴大horizon或搜索cost weights。三個模型已收到相同核心問題的
R6獨立派工；第四個完成receipt任務後亦已加入，所有獨立交付凍結前
不交換R6主要結論，不宣稱已完成四模型R6共識。

要比較的假設是：token progress、prefill/decode competition及合理EOS
uncertainty能否提供比marginal stock-flow更有用且一致的多步預測。
候選可以是batch-aware、fluid、hybrid或marked-event模型，也可以得到
「目前觀測不足」的負面結果。必須守住共同外生future law、
exact lifecycle／count／whole-cost一致性，以及cap只透過admission/state
影響GPU的因果路徑。

仍待驗證：Native線上receipt完整性、source-bound original cost mapping、
coherent fitted joint kernel、conditional uncertainty、solver deadline、
真正Fixed16／One-move／Reactive／MPC閉環比較與未見Final資料。
目前「16」是reservation budget語義，不能稱為16個in-flight requests。
長時辨識持續off-Pod封存；planned44,400秒仍不等於實際完成十小時。
