# Native v3：至少十小時的實證研究

使用者於 2026-10-31T06:31:01.884Z 明確要求繼續至少十小時的實驗。
僅使用已授權 Pod `24nm6zrh2sej8c`，不得另租、terminate 或刪除 Pod。
API 控制憑證仍缺失；使用者已允許在此限制下繼續。
SSH 停止 serving processes 不等於停止 GPU 計費，不能宣稱後者已達成。

## 已確認與尚未執行

SSH host key 已固定。GPU 是 RTX4090 24564MiB、driver595.91.07。
獨立 Python3.12 venv 已安裝 vLLM0.30.0、Torch2.13.0、
Transformers5.22.1。這是環境準備，不算 HTTP／GPU／MPC 驗收。
模型 revision 沿用 `cdbee75f17c01a7cc42f958dc650907174af0554`；
下載、實際 loaded revision 與 source38 guard 必須另外核實。

原生實驗時間從第一個有效 GPU qualification arm 記錄；
不能把排程十小時、安裝時間或無需求的 idle time 說成已完成十小時。
每輪保存 raw、實際動作／clock／identity、outcomes、失敗及資源時間。
API quote 未取得，不沿用舊 Pod 的美元價格冒充新報價。

## 執行關卡

1. 新 source-bound Gate A：標準 Chat、OFF／LIGHT、exact ID joins、
   source／clock、END／ACK、natural output audit 與 drain。
2. 隨機 open-loop interventions、observer ablation、same-action repeats，
   辨識 batch composition／whole-cost release／EOS／queue dynamics。
3. 用互不混用的 Train、Calibration、Development 評估 one-step、
   multi-step、risk calibration、action ranking 與 OOD。
4. 模型有可驗證預測價值後，建立可撤回的 MPC candidate；
   同 offered trace／SLO 比較 Fixed16、best-supported-fixed、
   reactive、strong One-move 與 MPC。
5. Final metrics、最小實用 effect、獨立 seed-block 與 CI／多重比較
   必須在 final outcomes 產生／查看前凍結。不重用 final 調參。

尚未宣稱十小時完成或 MPC 勝出。Correctness／credentials／raw-data
安全失敗應停下該實驗、保留失敗證據與修復，不能硬跑湊時間。
新假設必須有 measurable information value；不得盲調 weights／horizon。

## 四模型工作

四個原生 profile 的 push wrappers 與兩個 pre-push fallback hooks
維持 exit86。Workers 只能本機提交與交付，禁止 `gt done`。
各自先獨立提出可實作設計；互審只有在全部初稿 ready 後釋出。
重用舊 serial HTTP proxy 不被接受，也不能把 passing CPU fixtures
當作 native 性能或統計保證。
