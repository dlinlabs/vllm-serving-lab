# ADR：以 admission-mediated marked response 建立實驗性 MPC candidate

**狀態：選定研究 candidate，不是批准部署或宣稱 MPC 勝出。**
依據是已封存 v5 在 development181004–181007 的可重現預測結果，
不是四模型投票。Direct headroom IO 的負面結果保留。

## 控制範圍與可觀測 state

目前 actuator 控制 Gateway 的 reservation-cost budget。
16／32 是成本單位，不等於所有 workload 的16／32個running requests，
也不是KV bytes或GPU seconds。vLLM max-seqs32、B2048、GPU配置保持固定。
本 candidate 不直接控制 scheduler prefill/decode budget、GPU clocks或priority。

令 observation history 為 \(H_t\)，由 source-bound、已在cut前收到的資料構成。
可用的 marked state 寫作：

\[
x_t=(R_t,\mathcal L_t,\mathcal P_t,\widehat\lambda_t,\widehat q_t,m_t).
\]

* \(R_t=\sum_{i\in\mathcal L_t}c_i\)：Gateway原始整筆reservation cost。
* \(\mathcal L_t\)：已知owners的immutable cost、age、first-content knowledge、
  有可信join時的reported output-progress key。
* \(\mathcal P_t\)：已attempt、但尚無足夠ownership／terminal資訊的latent records；
  不能全部算成admitted或全部當成零工作量。
* \(\widehat\lambda_t,\widehat q_t\)：只從pre-cut實際attempts估計的arrival與payload mix。
* \(m_t\)：缺失、age、source／clock／ACK／support masks。

Native running／waiting／KV／computed/output counters 是延遲的observations，
不是無噪聲GPU state。Parent adapter保留running-only progress及unknown waiting work；
v5尚未完整使用這些新fields。不同engine／Gateway IDs不能猜測join。
因此這是partially observed、history-conditioned response，不是已辨識的完整GPU plant。

## 事件式動態與共同disturbance

每個candidate使用相同scenario的arrival／payload draw \(w^{(s)}\)，
不得讀已實現future tape、EOS、ACK或future snapshots。
v5用past2s估计的固定Poisson rate及past payload mix；
latent service／completion以Train marked empirical kernels近似，
由payload與Gateway-owned-load bin條件化。

對future offer \(j\)，在nominal應用budget \(u(\tau_j)\) 下：

\[
a_j=\mathbf1\{R_{\tau_j^-}+c_j\le u(\tau_j)\},\qquad
R_{\tau_j^+}=R_{\tau_j^-}+a_jc_j.
\]

只有接受後的population／load可以改變birth kernel；
不能在保持admission/state不變時，讓GPU service rate直接依budget變化。
Release事件移除完整 \(c_j\)，不依token progress比例衰減，不evict舊owners。
Current owners與future births的marked terminal events產生Good／rejected counts。

v5的release採snapshot interval上端，不是client terminal；
真實release、pending狀態、ACK application law與causal kernel invariance仍不完全識別。
該模擬維持帳本conservation，不代表已精確重建真實GPU timing。

## MPC objective與真正的One-move對照

以1s control slot、5s共同evaluation horizon開始，不增加horizon救分數。
Plan family：

\[
\mathcal U_{\rm MPC}=\{16,32\}^{5},\qquad
\mathcal U_{\rm one}=\{(a,b,b,b,b):a,b\in\{16,32\}\}.
\]

後者有4個plans，不是刻意削弱的one-step score；
兩者用同一kernel、horizon、common random draws、support及outcome semantics。
Full family包含strong One-move與兩個constant budgets。

在scope-qualified experimental mean-only mode：

\[
u^*=\arg\max_{U\in\mathcal U}
\frac1{32}\sum_{s=1}^{32}G^{(s)}_5(x_t,U,w^{(s)}).
\]

\(G_5\) 是模型對API-stop、output至少2、
scheduled-to-first TTFT≤0.5s、request-terminal meanTPOT≤0.05s
且在window內terminal的successful count近似；不是per-tokenTPOT或physicalEOS。
除以5s不改變argmax。Tie-break明列incumbent／action variation，
不以搜索cost weights製造勝利。
最大化Good count不能靠拒絕所有requests取得「完美latency」；
真實benchmark仍須同時報offered／accepted／rejected與SLO panels。

只實際套用first move，每1s重新量測／規劃。
模型內nested-family utility不低，不代表closed-loop MPC勝過One-move：
state分岔、模型錯配、ACK延遲、deadline與OOD都可能抵消它。

## Hard gates、fallback與未證明的性質

Hard allocator／actuation gates是budget domain、immutablecost、
no eviction、source／clock／version／identity／cost binding、
100ms command TTL與單一pending-operation fence。
降budget後舊reservation可以大於新budget；不能為了 \(R\le u\) 刪owners。
Source／clock失配不產生新action；其他fallback要記reason、
active model次數與实际ACK，不能把Fixed16 fallback算成MPC成功。

目前沒有calibrated chance constraint、terminal stability proof、
recursive feasibility／queue stability／deployment safety guarantee。
Terminal cost先不加入任意權重；finite-window myopia與未來未完成work是明列風險。
名義80% bands實測under-cover，不能換成95%SLO risk guarantee。
Mean-onlyexperimental research mode仍需先完成真實feed／source／actuation資格；
不等同production permit。

## 實際下一個驗收

R9工作是：release-endpoint sensitivity、真實frozen-kernel solver耗時、
source-bound在線input adapter及actual capture＋guarded-actuator mock integration。
沒有替換live729，也沒有新增GPU。

完成上述CPU資格後，再用同一已授權GPU、fresh staging/source qualification
執行非Final development closed-loop比較：
Fixed16、Train-selected supported fixed16/32、reactive、strong One-move、full MPC。
保持同tape／GPU／model／SLO／measurement／warmup，
保留所有fallback、errors、rejections、late／censored outcomes與controller CPU cost。
Final workload／主要metric／實際有意義margin／統計規則須在Final曝光前由Mayor正式凍結；
現在的draft5%提案不自動成為已批准的Final結果或統計保證。
