# PR33：原生 GPU／HTTP 資格與 Train-only 偵察

## 結論與範圍

已使用使用者提供並完成 SSH 授權的 RTX 4090 Pod，完成真實 CUDA、
標準 Chat HTTP、read-only EngineCore RPC、guarded budget actuation 與 drain
資格檢查，再執行四個獨立 Train scout seed block、每個 block 兩個固定 budget arm。
**尚未執行 Native MPC，也沒有證明 MPC 穩定勝出、持平或不可能勝出。**

最重要的新資訊：短 prompt 下 budget32 的 TTFT 達標完成數高於 budget16；
長 prompt 的飽和場景則相反。這是 workload-dependent action ranking 的原生
探索證據，不能將舊合成環境的「Fixed16 優於 Fixed32」普遍化到這個部署。
但每格只有一個 seed，不能宣稱穩定差異，更不能將 fixed-budget 比較當 MPC 勝出。

產品 source commit：`0588707949f651039d44a7a034c12bd35b8c8047`。
產品 implementation hash：
`b4f234020105f69f608da45dba7317a8bb632ecd84aeb0e310362740e5fa083a`。
本輪沒有修改 MPC model、solver、cost weights、horizon、deadline 或舊 benchmark。
新增元件位於 `experiments/mpc-research/native_observer/`，獨立 opt-in，
不替換 scheduler 或 GPU kernel，另綁定 observer source SHA。

## 環境與已確認的資格

| 項目 | 實測／固定設定 |
|---|---|
| GPU | RTX 4090，24564 MiB，driver 595.91.07 |
| 模型 | Qwen/Qwen3-4B-Instruct-2507，BF16 |
| 模型 revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| 軟體 | Python 3.12.3；vLLM 0.30.0；Torch 2.13.0+cu130；Transformers 5.22.1 |
| Runtime | max model len8192；max seqs32；batch tokens2048；GPU memory0.8；TP1；prefix cache off |
| Execution | 保留正常 graphs／async，沒有改成 eager 來迴避問題 |
| Installed wire | 真實安裝環境 24 passed；不是先前的 installed-vLLM skip |
| Source guard | 38 個 reviewed vLLM source file 完整匹配 |
| Observer | 5 個 focused tests；真 RPC 可見 running、waiting、pending batches、KV usage、request progress |
| HTTP 資格 | 9 planned：8 complete、1 預定 capacity rejection；0 failed／unresolved／integrity errors |
| Prompt 資格 | prepared token IDs 與 native usage 同為128／4096 |
| Drain | Gateway 無 reservation／requests，加上兩次 fresh native idle snapshot，且無未解 pending operation |

Observer SHA：
`1bbf7ebed9b3427fd2131dffb8910751544d725f42a3f03f4ce74a4721ee189e`。
僅新增 authenticated loopback endpoint 與 EngineCore utility method。核心與 frontend
各記錄真實 monotonic sample time；先驗 clock／source／epoch，再比較時間。
資格限制為 age≤0.5s、core-to-frontend sample width≤0.05s，沒有用
Prometheus HTTP response time 假裝引擎 sample time。

上述為實際 GPU run 的 observer v1；原檔案也封存在 archive。
收尾時另修正 package 被 vLLM自動發現但 feature未啟用時的 endpoint行為：
現在 default-off不註冊route、不存取engine、不使正常startup失敗。
修後 source SHA為
`29fc1ebd875ccc1a53b9584fd0f3b002c518e3653c9b4238877b67b7ec13802e`，
CPU及原生環境的6個focused tests通過；**修後版未重新啟動 GPU服務，
不能將 v1的GPU資格冒充修後版重新驗收。**下次啟用須重新綁source並做smoke。

**觀測限制：**`num_computed_tokens` 是 scheduler bookkeeping，async pending batch
可能尚未完成 GPU 工作；不能直接將它當物理完成量。非空狀態的 `orphaned`
回傳未知，而非假填0。此次 RPC 資格不等於 GPU phase duration 或 state estimator
已完成因果驗證。

## 第一版錯誤與控制路徑失敗完整保留

第一版 preparation 對 Transformers `BatchEncoding` 呼叫 `len()`，得到欄位數2，
不是 token 數；名義128／4096實際送成146／4114。修正為
`return_dict=False`，加入 prepared/native/nominal 三者相等檢查，再重跑。
第一版 script、JSON、log 保留；它僅能支持 framing 功能，不支持 token-shape 資格。
這是實驗腳本問題，**不是已證明的 MPC 性能落差成因**。

兩次初始啟動失敗也保留：venv 的 ninja 不在 PATH，以及 exec argv0 使 Python
錯認系統 interpreter。只修啟動環境，沒有放寬 runtime 或實驗門檻。

Scouting 第七次量測前 assignment 收到 HTTP409，client 正確保留
`pending`，立即停止，未在未知 cap 下發送該格的 offers。
原始失敗 run 保留。後續使用既有 `reconcile()` 得到
`expired, observed_version=8`，等於原 expected version，才清除持久 pending，
重新取得 state 並收到新 assignment 的 applied acknowledgement，完成尚未量測的兩個 arm。
沒有刪 pending 檔案、增加100ms TTL，或重跑已完成的六個 episode。

已確認的是拒絕／unknown-reply 路徑及 expiry fence；原409的完整 reason／persist
時間未被 receipt 記錄，**不能斷言 fsync 是根因**。這也不是 Native MPC
solver deadline fallback；Native MPC 尚未執行。

## Train-only paired scout

每格共用12秒 Poisson offered tape，從兩次確認 drained 狀態開始，溫度0，
固定相同模型／runtime／instrumentation；budget順序在格間交替。
短文與較長文自然 EOS，實際 completion tokens約36–116，不再使用只回兩個 token
的功能測試充當代表性 decode workload。所有1189個完成回應均為 natural EOS。

**16／32是 reservation-cost budget，不是同時請求數。**
原生成本規格未改；短 prompt兩格的最大 observed running 分別為15／31、9／18。
長 prompt兩格分別為3／7、3／6。

| Prompt / Max output | Offered / arm | Budget16：complete / reject / TTFT達標 | Budget32：complete / reject / TTFT達標 | TTFT P95：16 / 32 |
|---|---:|---:|---:|---:|
| 128 / 128 | 738 | 307 / 431 / 307 | 517 / 221 / 517 | 100 / 118 ms |
| 128 / 256 | 388 | 94 / 294 / 94 | 164 / 224 / 164 | 95 / 98 ms |
| 4096 / 128 | 167 | 27 / 140 / 10 | 35 / 132 / 7 | 706 / 1510 ms |
| 4096 / 256 | 193 | 18 / 175 / 6 | 27 / 166 / 5 | 718 / 1260 ms |

TTFT達標定義為從**scheduled arrival**到first content≤0.5s，包含發送 lag。
這是有 drain 的 offered-cohort 描述量；不是 measurement-window throughput，
也不是已驗收的 TTFT+TPOT+E2E 綜合 Goodput。
每格只有一個獨立 seed；不能拿2972個 requests或1412個 native samples當獨立重複。
最末格的6對5尤其不能當穩定 ranking。

全部2972個 physical offers＝1189 complete＋1783 rejection，沒有丟棄不利回應。
八個完成 episode 的3567個 ledger frame皆連續、identity／clock／watermark一致，
telemetry dropped/write errors皆0。`reserved_cost_at()` 已從實際 frames驗證每格
完整 observation window，可重建 R(t+1)，不使用 polling interpolation。
離線亦完成1189個 native／Gateway request ID無歧義join，檢查prompt／max output
一致與output progress單調且不超過最終usage；尚未涵蓋cancel／multi-output／其他版本。

## 對 MPC 的意義：確認、推論、未驗證分開

**已確認：**相同 GPU／模型的 action 效果隨 workload 改變。短 prompt放大
budget提高接受／完成數，TTFT仍低；長 prompt飽和時接受更多卻增加尾延遲，
達標完成數未隨之增加。舊 CPU negative result保留，但不是跨 GPU／模型不可能勝出的證明。

**推論：**prefill backlog／interference、decode remaining-work與 EOS 的估計，
比盲調 solver weights更接近下一個問題。MPC可能利用 workload shift選 cap，
但簡單 workload-aware feedback也可能取得同樣收益；必須列為強 baseline。
只比較新 controller與一個不適合該 workload的 Fixed16不足以證明 MPC獨特優勢。

**新的警訊：**離線 persistence `R(t+1)=R(t)`診斷，在長 prompt／max256／budget16
的10個事後固定1秒格點可達0 MAE，但該 episode的TTFT P95仍718ms。
reservation occupancy 可在飽和時近乎固定，不能只用其預測準確度宣布 SLO risk模型有效。
這是 retrospective Train診斷，不是 causal online-issued forecast，也未更換凍結的 primary metric。

**仍未驗證：**multi-step是否優於 one-move、Native MPC solver deadline／完整控制路徑
overhead、chance constraint coverage、OOD／跨硬體移植、穩定優勢／實用等效。
request-mean TPOT可作診斷，但逐 token時間與 TPOT-tail SLO仍未資格化。
硬體資料僅最後 point sample，沒有 clock／power／temperature workload time series。

## 下一階段最高資訊價值工作

1. 擴充本輪 per-request Gateway／engine ID join資格，以及 async bookkeeping
   與實際輸出完成量的區分；記錄 command persist前後、POST response與fence時間。
   避免將未知409或posting延遲統稱為 solver timeout。
2. 使用本輪僅 Train scout估计區間，凍結非飽和／近飽和／過載及 recovery rates；
   長 prompt的下一輪不應只重複16req/s高度拒絕场景。原48-episode proposal
   仍不是已執行或已批准的完整驗收。
3. 在明確 offered tape下執行互補16／32動作序列，記 actual applied lag、
   Native pending、KV、prefill/decode進度與完整HTTP outcome。比較整段 assignment效果，
   不把後續已分岔狀態點相減冒充 same-state causal contrast。
4. 同時檢查 R1、多步 residence與all-offered SLO risk；以 persistence／queue feedback
   為強 baseline。正式split仍維持 Train／Calibration／Validation／Test隔離；
   本輪只用159901–159904 scout seeds，未生成160021–160024 Test tape。
5. 原生模型經資格後才進行 Fixed16／Fixed32／Feedback／One-move／MPC公平比較；
   評估是否需要更低 fixed budget作合理 baseline，擴充 action support須另版本化與回歸，
   不直接繞過目前16／32 permit。不能把全 fallback16或Never-binding MPC當成等效成功。

已停止本輪自己的 vLLM／Gateway，確認8000／8080關閉且GPU compute process為空，
保留模型 cache、venv、所有本機證據與提交。**沒有關閉使用者的 Pod；
租金可能仍持續，實際 provider價格／帳單未知。**沒有新租資源、Push、Merge、
PR留言、變更 Git remote或呼叫 `gt done`。

## 重現與證據

原始證據：`results/mpc-native-4090-scout/native-evidence.tar.gz`。
SHA256：`400e96a79c17ad7a127d4b4a5afb8db24cb0a917cde717c5ba2b23d551aa50b5`。
包括 failed startups、token-shape v1／v2、installed-wire log／JUnit、observer tests、
兩版 scout、HTTP rows、native frames、telemetry、環境與完整 package versions。
控制 credential檔案不在 archive，且已檢查實際 key不出現在任何 evidence檔案。
另外的 `results/mpc-native-4090-scout/resource-stop.json`保存實際關閉port與GPU清空檢查。

```bash
python experiments/mpc-research/native_observer/analyze_evidence.py \
  --evidence results/mpc-native-4090-scout/native-evidence.tar.gz \
  --output /tmp/pr33-native-verification.json
python -m pytest -q experiments/mpc-research/native_observer/test_observer.py
```

執行 observer測試時需將該目錄與repo置於PYTHONPATH，或安裝該獨立experiment package。
腳本保留此 exact deployment路徑以忠實重現原執行；它們不是一般部署入口，
也不會自行啟動或租用遠端 GPU。
