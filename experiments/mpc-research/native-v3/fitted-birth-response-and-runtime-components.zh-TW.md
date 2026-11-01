# NV3-009：Fitted birth-response 正向訊號與尚未成立的閉環價值

**首次取得可重現的五秒 birth-response 改善，但尚未證明 MPC 的 GPU 效能。**
四模型 R8 完整來源已封存；Mayor 實際執行241 tests／152 subtests，
包含真正 `capture_chat` callback 與 fake HTTP transport integration。

| Backend／worker | R8完整本機commit |
|---|---|
| claude-opus-5.5／amber | `fa9079e6c436b94f776c1ddc39a90e2c86fe8a46` |
| gpt-6-astra／jade | `6110648c364bb1438bf5c69cce83a692e680ffc4` |
| gemini-3.8-flash／pearl | `85bc1cf7b8448f7943973bbc241f0cbd8c49a62f` |
| gpt-6.1-sol／flint | `3e32812f377f029e620c6272ced3eeb9d4cab36e` |

R8 source archive SHA：
`c978d4d52cca9b19e1115659eccaa6946d6b46dde4a75608af42e83c9035aba1`。
Train181001–181003 fit；先提交規格，再評估 development181004／181005。
兩個blocks不是Final，也不是540個獨立實驗。

## 已獨立重現的正向結果

Mayor 執行凍結 Opus 的 `--replica-admission-birth`，完整 raw JSON
byte-for-byte 相同，SHA：
`939e835a9162d31be6a3be8cc20570e46f6156290578277eaa9162f2877b8a68`。

| 同1108 eligible origins的 Good-count MAE | 1秒 | 3秒 | 5秒 |
|---|---:|---:|---:|
| History | 2.86 | 6.04 | 9.16 |
| Aggregate hybrid | 1.39 | 3.84 | 6.12 |
| Progress hybrid | 1.11 | 3.58 | 5.96 |
| Additive full-plan | 1.11 | 3.51 | 5.93 |
| Admission／birth hybrid | 1.22 | 2.63 | 4.08 |

五秒相對較強的同-mask comparator 改善約31%，在兩個blocks皆成立：
4.06／4.09，對照 additive5.67／6.20。
一秒反而較差，不能說所有時間尺度都改善。
540個 paired factual contrasts 的五秒 MAE4.17／零差異5.91，
sign agreement約0.70；是不同arm狀態的兩個factual forecasts，
不是同一狀態的 action counterfactual 或已驗證的 MPC plan ranking。

模型以共同、plan-free offered draw 模擬 admission，
透過 Gateway-owned-load bin 改變 marked birth／completion，
整筆原始成本只在模擬 release 移除。它不是已識別的 GPU service law；
Native running／KV／computed-token dynamics 尚未完整接入。

## 邊界修正與保留的負面結果

初稿用client terminal代替Gateway release；在新blocks曝光前改成
Gateway owner disappearance 的 interval-censored upper endpoint。
舊development181003 smoke分數因此從3.75變差至4.04，
不是看新分數選擇較漂亮的boundary。

Train中release upper minus client terminal的median約+72ms，
lower minus terminal的median約−69ms。真實release只能定位在snapshot interval；
其中41個沒有後續snapshot的Train admitted IDs明列unidentified，
沒有偷偷改成client terminal或抹除。

五秒Good bias約−1.26，rejection bias約+3.0；
rejected level MAE9.25仍不比additive9.14好。
10–90 scenario bands的Good／rejected／offers inclusion約
0.63／0.52／0.49，低於名義0.80，**不是校準好的風險區間**。
Constant32-minus16的預測增益是未驗證counterfactual，不是GPU實測。
Planned-to-applied ACK law及optimized-policy support仍待驗證。

## 競爭 direct IO 的負面結果也已重現

Mayor重跑Astra固定headroom／interaction fit，兩個blocks的G5 MAE
5.7273／5.9520，輸其同-mask additive5.4652／5.8332；
paired errors7.5339／7.5243仍輸零差異5.8077／5.8289。
這組1125 origins與Opus的1108 origins不完全相同，
不能直接將兩組aggregate MAE當成無條件head-to-head比較。

Worker compact raw SHA：
`a260cfb391b50a590869d33ee37215cbde5c8b87663f0208b72eb444a14351c2`；
Mayor輸出SHA：
`d0d43332df585b601d02a0cf0bbf3e00d5fd77a0ac9abbec1843dbc19616fc95`。
遞迴比較全部JSON結構／1125 predictions／masks／coefficients，
**零差異**；byte hash不同只因serialization。
第一次從非Git cwd執行source-seal檢查失敗，未算通過；
改從包含seal的pr32 Worker object database執行，未修改任何remote或frozen source。

## 真正完成的CPU工程與仍缺的整合

Sol bridge已使用真實capture callbacks測試：
attempt在POST前發布、first-content在SSE中可見，terminal後完成；
包含unsent cancellation、reject、timeout、fail與dropped/mismatched receipts。
不倒填舊cut，不把planned registrations當actual attempts，
不把offered estimate／unknown charge假扮Gateway原始成本。
Source-auth、global completeness、physical EOS及active permit仍未授予。

Gemini交付future-hidden控制callback、同tape多policy、late／unfinished
accounting及fallback／active分開記錄的CPU harness。
這不是實際HTTP／GPU closed-loop benchmark。
Primary自然API-stop outcome不冒稱independently verified physical EOS；
Final5% practical margin／八類workload仍標為draft unsealed proposal。

Sol再次嘗試`gt done`，實際被mechanical guard以86阻擋；
global pre-push hook無條件exit86。未Push，未建立正常完成許可。
Witness immediate nudge因沒有pane失敗，不當成成功通知，
也不為了完成流程啟動Witness／Refinery。Mayor保留caller-managed本機證據。

## 下一個已預先固定的診斷

`round9-replication-preregistration.json` 固定未改動的v5，
同Train181001–181003 fit、同32scenarios／bins／1/3/5秒／release-upper，
評估新的complete Train181006／181007。
不refit已用blocks，不換metric、weights、masks或seed救結果。
先保存preregistration commit，再做資料freeze與prediction replay。

這是下一輪development replication，不是Calibration／Validation／Final。
後續release lower／upper sensitivity需另行預註冊，
不能將兩個endpoints當成非線性Good／risk的已證明全域bounds。

## GPU與資源狀態

截至2026-10-31T11:20:06Z，新continuation42episodes／5040 offered秒，
原long01的3600秒另列；合計8640秒，約2小時24分鐘，十小時未完成。
GPU RTX4090約79% utilization／19,804MiB／53C，持續有辨識工作。
Rate與provider billing-stop仍unknown；沒有將停服務等同停計費。
沒有將新CPU source／fitted model替換live729，也沒有啟動第二個GPU。

2026-10-31T11:24:46Z的backend persisted usage：Opus1984.76366、
Astra4105.1571、Gemini705.0703875、Sol740.03669 AI credits，
四個current Native sessions合計7535.0278375。
相對R7報告snapshot增加1516.845325；只是兩個snapshot間的增量，
不包含Mayor／舊sessions／GPU費用，亦非完整R8或最終帳單。
