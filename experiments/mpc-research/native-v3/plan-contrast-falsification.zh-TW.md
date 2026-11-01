# NV3-008：五秒 plan contrast 反證與下一輪實作

**描述性預測改善尚未轉成可用的控制效果預測。** 四模型完成 R7 互審，
Mayor 封存各自已提交的三個檔案，並獨立執行 217 tests／146 subtests。
第一次使用 unittest discovery 沒有收集到 Opus 的 pytest tests，未算通過；
改用專案既有 pytest runner 後完整執行，沒有 skip。

| 實際 backend／worker | R7 完整本機 commit |
|---|---|
| claude-opus-5.5／amber | `36dfd7841f97485bba64e0832f9e9a319101c1d0` |
| gpt-6-astra／jade | `f863d86a642e70e5bdd7f055f3514b02fa7ba341` |
| gemini-3.8-flash／pearl | `23e93cf7cfe9b70002c9a5e4cd787b8bbd10b389` |
| gpt-6.1-sol／flint | `e86b31a8e13134bfdaebde651786e796810eee1c` |

Source archive SHA256：
`7b192249c55991956435813c06669b6ecb96c58848889fec2230da236e572f24`。
Mayor 執行凍結 Opus 的 `--replica-plan-response`，raw JSON 與 Worker
byte-for-byte 相同，SHA256：
`f65efd69ef1fb8b2a5af26a3499f69b1ee7179cf62019e5520f3a396a3295f64`。
Train181001／181002 fit，已使用的 Train181003 development 評估；
這不是新的 held-out 或 MPC GPU benchmark。

## 已確認的診斷

在相同 origin／support masks 下，264 個 paired factual contrasts：

| 五秒指標 | 預測零差異 | 不讀未來 plan | 完整 additive plan |
|---|---:|---:|---:|
| Good-count paired-difference MAE | 6.90 | 8.15 | 8.15 |
| Rejected-count paired-difference MAE | 7.19 | 7.94 | 7.14 |
| Good-count level MAE | — | 5.92 | 5.84 |

完整 plan 的 Good 差異 sign agreement 約 0.62，不是完全沒有訊號；
但差異量值誤差輸給零差異。較低的 level MAE 不能直接當成較好的控制排序。
一秒 Good paired MAE 約 1.40／零差異 3.58，短時間已有訊號，
不能將五秒失敗擴張為所有 horizon 或所有 MPC 都無效。

配對 arm 共享 offered tape，但過去動作已使狀態分岔。
**上述差異是兩個 factual forecasts 的診斷，不是同一狀態下的 action
counterfactual，也不足以單獨證明某模型能或不能排序任意 candidate plan。**
下一輪須同時報告 full Good MAE、paired diagnostics、support 與缺失樣本。

實際 Train181001／181002 包含全部 32 個五槽 pattern，最少 43 個
overlapping windows；constant16／32 各 43 個。
這排除了「本資料必然沒有 constant plan」的說法，但不是 43 個獨立實驗，
更沒有證明每個 history／batch state 的 conditional positivity。
隨機指定的 ITT 能否轉成部署時的效果，仍依賴已知 randomizer、
pre-assignment history、未選擇性排除未來失敗與 application protocol。

## 保留的限制與裁定

1. Direct full-plan IO 可以表達 admission-mediated total response；
   不能將所有 budget-to-outcome 模型等同於非法的 budget-to-GPU-service edge。
   目前 additive 版本沒有展示足夠的五秒 contrast 精度。
2. ACK receipt lag 包含 slot scheduling／retry／receipt 時間，
   不等於 command-created-to-applied lag。舊 Train p99 約 0.42 秒、
   max 約 2.41 秒，不能說 Gateway 的 100ms TTL 被放寬。
   新 confirmed-hold protocol 的 application law 另行綁定。
3. Astra／Sol 的缺失 pattern、selected-policy coverage 反例是數學 witness，
   不是對本資料 randomizer 的測量。Random-policy coverage 不自動提供
   optimized-policy 或 simultaneous candidate guarantees。
4. Gemini 已修正：120ms／12% dilution 是 hypothetical scheduled-slot lag，
   不是 Native 實測或 command TTL 違規。其 additive uncongested 反例不否定
   有 state/headroom interaction 的 direct IO。
5. Offers placebo 的 factual paired 差異很小，模型卻預測較大差異，是錯配警訊；
   但不同 pre-cut histories 的 conditional forecasts 不必 bitwise 相同。
   正確要求是**同一 origin 的共同外生 arrival/payload draw 對所有 plans 相同**，
   並區分 scheduled arrivals 與可能受 client scheduling 影響的實際 attempts。

## 已派工 R8：實作而不是再增加提案

Opus 實作 fitted admission／marked-birth hybrid；Astra 實作小型
piecewise full-plan IO competitor。兩者使用 Train181001–181003 fit，
先提交 feature／fit／mask／loss 定義，再評估已封存的 recovery
Train181004／181005。這兩個新 blocks 的粗略 summary 已被 Mayor 看過，
明確列為 development，不冒稱盲測 Final。

兩者固定 1／3／5 秒、現有 SLO 與 API-stop outcome，保留 whole original cost、
未知 state、source/core/cost／ACK-bound holds；不搜索 weights 或 horizon。
比較同 origin 的 history、較強 aggregate hybrid 與 additive plan，
保留所有 abstention、future-label exclusions 和負面結果。

Gemini 實作 future-hidden controller callback 的公平 benchmark harness；
Sol 實作真正 `capture_chat.observe` 到 causal receipt history 的 bridge，
以 fake transport 執行真實 callback integration。
這些尚未完成，尚未部署 GPU，也不授予 active MPC／risk／physical-EOS permit。
未來 Final 的最小有意義改善 5% 是待凍結的研究提案，不是實測改善。

## 長時實驗與資源

### CPU Native state adapter：已實作，未部署

`mpc_lab/native_state.py` 新增 source-bound adapter，67 項 native-state／
runtime／actuator coupled tests 通過。先檢查兩邊 clock domain／run／observer／
vLLM version，再檢查 timestamps、完整性與有限數值；跨 boot 返回可觀測
refusal，不先解析或比較不相容時間。

Gateway 原始 whole-cost owners 與 raw engine request IDs 分開保留，
不猜 UUID aliases，不把 token progress 當成 reservation release。
`visible_prefill_remaining` 只涵蓋 running 列表；
waiting requests 沒有逐筆 token inventory，不能補成零工作量。
`visible_output_headroom` 是 max-output cap headroom，不是已辨識的剩餘 EOS 工作。

以已封存 Train181004／181005 的 recorded receive time 回放：
20 complete episodes／17,909 frames 均通過這個 adapter 的資料格式與時效檢查；
其中2,364 frames 有 waiting。這是 retrospective replay，
不是 online publication／latency／source-auth／MPC qualification。
computed counter 超過 prompt 的108,971個 request-observations 包括正常 decode，
不是108,971個獨立請求或 physical overhang 錯誤。
Raw evidence 保留20個 input SHA、measured source729與 adapter SHA
`835df8ebaa58c773074e65cedba4627b468d14b560233a7655bf9cd2240b83d6`。

新增這個 package module 會改變本機 implementation hash；
不能拿本機新 hash 冒充正在量測的729／b4 core，沒有 hot-patch GPU。

截至 `2026-10-31T10:55:50Z`，continuation02 已 off-Pod 核對
30 complete episodes／3,600 offered-window 秒；原 long01 的 3,600 秒另列，
合計約兩小時，**十小時尚未完成**。本輪 planned44,400 秒不是完成數。
GPU RTX4090 當時 utilization100%、memory19,804MiB、60C；
GPU 費率與 Pod billing-stop 權限仍未驗證，不捏造費用。

四個 current Native worker sessions 的當時 backend usage：
Opus1608.11542、Astra3182.83575、Gemini591.0859425、Sol636.1454 AI credits。
這不是本輪增量、不含 Mayor／舊 sessions／GPU 費用，也不是最終帳單。

Monitor v2 保留 finalized artifacts／checksum／resource ledger，
失敗或完成時核對 owned PID／start ticks／command fingerprint。
停止服務後還須確認 owned descendants 不再運作、8000 關閉、
GPU compute process list 為空，而不是只看 port。
這個 shutdown 尚未發生；**停止 GPU workload 仍不等於停止 Pod 計費**。
只重啟 monitor，沒有重啟辨識 driver 或替換正在運行的 GPU source729。
