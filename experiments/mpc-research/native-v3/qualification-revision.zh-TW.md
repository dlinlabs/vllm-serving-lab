# Native v3：實測反證與驗收修訂

本輪尚未完成十小時實驗，也沒有 MPC 勝出證據。只使用既有 RTX 4090 Pod；
沒有建立新付費資源，沒有 Push、Merge 或呼叫 Mayor 的 `gt done`。

## 可追溯的已確認結果

- `311c82e`：修正 serving launcher 沒有將已選 venv/bin 加入 PATH，
  導致 FlashInfer 找不到「已安裝」的 ninja。沒有關閉 graph、async 或 sampler。
  真實模型載入及服務就緒隨後成功。
- `2f9e145`：第一筆 HTTP 200、自然 EOS、4096 input／32 output、
  `[DONE]` 完整，但回傳 UUID 與指定 UUID 不同。啟用既有
  `BENCHMARK_REQUEST_IDS` 開關；沒有放寬 HTTP identity 檢查。
  修訂後 HTTP identity／stream／usage 檢查成功。
- TRAIN `nv2-qual-170001-p4096-n1-light` 的舊 v2 audit 仍是 **false**：
  實際 34 個 scheduler／worker 對應步驟，兩個 2048-token prefill，
  32 個單 token decode forward。已交付 output=32，其自回歸最小 forward
  位置終點為 4127，實際连续位置終點為 4128。
- 額外末端步驟的 BEFORE 狀態是 computed=num_tokens=4127、placeholder=1；
  排程早於自然 EOS 的 host commit，該步 output commit 晚於 EOS。
  Worker 的 executed position／count 同步對應，current-stream forward
  span 約 9.3655 ms。此 scalar **不是絕對 GPU 時鐘、kernel busy，
  也不能證明 GPU forward 在 EOS 後才結束**。

環境：Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`；
vLLM 0.30.0、Torch 2.13.0、Transformers 5.22.1；
RTX 4090 24564 MiB、driver 595.91.07；BF16、seq32、B2048、
maxlen8192、prefix cache off。控制器仍是固定 reservation cost budget32，
不是已校準 MPC，也不是 concurrency=32。

## 推論、修訂與限制

資料反證了「自然 EOS 時，所有實際 forward 必須剛好等於
prompt+delivered_output−1」這項驗收假設。它符合 async placeholder
在 host 收到 EOS 前已推進下一個 forward 的機制；不應刪除額外工作、
把它計入已交付 output，或停用 async 冒充原配置通過。

新增版本化 v3 accounting：完整保留原 v2 audit；只有唯一、連續、
實際 worker 對應、具正確 BEFORE placeholder 與 host chronology 的
一 token 末端 suffix 才能另列為 terminal overhang。沒有 placeholder、
worker mismatch、兩 token 差異、零 output、非有限 clock 都仍拒絕。
這是可反證的物理工作分類修訂，不是修改 Goodput／SLO 或放寬時間門檻。
`2587d2b` 的 fresh `qualification-retry03` 已完成完整 16-arm Native
qualification：40 HTTP 請求皆完整、40 自然 EOS，16 個 v3 audit 全部有效；
8 個舊 v2 audit 仍為 false，20 個末端額外 physical token 獨立列帳。
舊版本不能宣稱通過，這也不是 MPC、預測模型或 controller overhead 驗收。

第三份原始證據已封存並下載到 Mayor，遠端／本機 SHA256 相同：
`cbbb88639bae23a9b4fe9a5aa3f8b39d7716506d09ee40710561e649a28ca800`。

原始失敗紀錄保留於 Pod 的
`/root/pr33-native-v3-src/results/mpc-native-v3/qualification-retry01`、
`qualification-retry02`。第一份封存本機／遠端 SHA256 相同：
`61e5f49eadee73865af3807fa5d7d06243c563f511c115d73033ab4e1d3bfe80`。

四個新原生 Worker 的 model args 已確認；workspace trust 及 sandbox
Dolt denial 曾阻止實際研究。Mayor 只確認四個 pr32 workspace 的當次
trust，不取消 sandbox；以 native immediate nudge 提供完整離線任務。
必須另以 backend usage／成果證明模型確實研究，不能只引用 tmux 存活。
Astra、Sol 的錯誤 `gt done --status=ESCALATED` 嘗試已被 exit86 防線攔截。
四個 active research session 的 backend `session.usage_record` 已分別
確認 `claude-opus-5.5`、`gpt-6-astra`、`gemini-3.8-flash`、`gpt-6.1-sol`。
四份獨立 R1 原始報告、程式及測試已從各自提交 byte-verify 凍結；
Mayor 單一 runner 獨立重現 65 passed、68 subtests，第二輪交叉審查已實際派出。
Gemini R1 的兩個模組誤放 repo root，原始提交已保留，另要求 owned-path 修訂；
測試通過不代表其 duration、OOD、GPU clock 或統計論點已獲認可。

GPU lifecycle credential 仍不存在；只能停止自身服務，不能宣稱 SSH
stop 等同停止 Pod 計費。使用者已明確允許在此限制下繼續，但使用時間、
計費未知狀態及任何實驗停止都必須如實記錄。

## NV3-001：受控 cap／HTTP 介面資格，已執行

`87f7e56` 在原 RTX4090 上實際執行四個 12 秒 measurement windows，
1 秒 reservation cap16／32 干預：922 offered、648 complete／自然 EOS、
274 admission rejection，沒有其他或截斷結果。48 個 assignment 最後
均得到 guarded applied ACK；另兩個 pending attempt 完整保留，不能說
每一次嘗試都成功。386 個原生 state samples 的 source／clock／freshness
availability 均有效，所有 trial 最後都有兩次完整 drain 證据。

封存原始 HTTP／state／action／GPU monitor，本機／Pod SHA256 相同：
`2f00e9217800d36c9a62a659a0567ed7b6c00ac07a50d5571ef21bae2e177467`。
Native vLLM epoch 仍為 `2587d2b`，gateway／collector 為 `87f7e56`；
產品 core implementation hash 沒改，native-launch metadata 分別記錄，
不把 collector revision 冒充已載入的 vLLM revision。

這是介面 smoke，不是 MPC benchmark、獨立重複或原生模型驗收。
兩個短 workload arm 的 cohort-good 分別 265／280，mixed 為 41／36；
只支持「值得研究 workload 與作用序列交互作用」，不能相減後宣稱
同狀態 cap causal effect 或 MPC 改善。正式辨識另加入獨立、可重播的
arm 執行順序隨機化；smoke 的原順序資料保留，不拿來做顯著性結論。
