# 同機 fixed-cap 實驗：本輪結果與限制

本輪於美西時間2026-10-19／UTC2026-10-20在同一台RTX4090執行，runtime為
`31f1079`。原定八場正式測量只執行七場：六場合格，cap16-r2完成但未通過
load fidelity與外部診斷門檻，最後cap64-r2沒有開始。Smoke通過。
**原始suite仍為failed/incomplete；沒有刪除失敗場次或放寬門檻。**

## 發現

| Cap | R1合格goodput | R2合格goodput |
|---|---:|---:|
| 16 | 4.130/s | 未通過，僅供診斷 |
| 32 | **5.080/s** | **4.590/s** |
| 64 | 2.267/s，包含一筆502失敗 | 未執行 |
| 96 | 3.330/s | 2.667/s |

Goodput定義：完整完成串流，且scheduled arrival至first content不超過0.5秒，
再除以完整300秒。沒有刪除開頭30秒，所有offered requests均保留。

Cap32兩次平均4.835/s，cap96平均2.998/s，觀察差異為61.26%。對應到達時段的
20個30秒cohorts也都是cap32達標數較多；這些區段彼此相關，不是20次獨立試驗。
低cap可能拒絕過多，高cap則可能接受更多但大量超時。目前資料支持把cap32作為
候選fixed baseline，不代表全域最佳值、統計顯著改善或MPC成果。

**Cap32的P99 TTFT仍為0.837／0.978秒，沒有達到P99≤0.5秒的保證。**
本輪優化的是達標請求數。Cap16-r1中，成功完成的請求有97.71%達標，但拒絕率
83.41%；cap32則拒絕約77%，有效服務數較多。公平性仍須分input長度閱讀，
不能只憑總goodput推論：cap16-r1的短input達標61筆，比cap32的58筆略多。

## cap16-r2為何不合格

- 測量約96.338–102.193秒，gateway loop monitor延遲5.855秒。
- 96.747–98.750秒，vLLM `/metrics` 發生約2秒ReadTimeout，沒有收到headers。
- 102.246–104.050秒的37筆到達，client launch延遲超過50ms；這37筆最終均被拒絕。
- Gateway停頓區間沒有重疊GC span；附近6.602秒取樣區間只增加約0.123秒process CPU。
- Collector正常結束：301samples、1sample error、0write errors。

目前不支持把長停頓歸因於gateway GC或大量Python CPU計算。阻塞I/O、OS排程等
只是候選原因，沒有syscall／off-CPU trace可定案。恢復後積壓處理與runner延遲
在時間上相接，不等於已證明完整因果鏈。延長metrics timeout也不能修復另一項
load fidelity失敗。該場描述性goodput為3.867/s，不列入合格比較。

Cap64-r1另有一筆upstream `ReadError`，gateway回502；reservation已釋放、
cleanup完成，該筆保留為failed且不計入goodput。資料完整性合格不代表服務零失敗。

## 驗證與下一步

完整上傳封存含161個原始檔；56個分片可重建byte-identical tar.gz。119個
supervisor登記的artifact hash吻合，八份已產生報表用原auditor全部重現。
Exact-revision native CPU紀錄為504項全通過，沒有failure/error/skip。

本輪資料足供離線分析，不必為讓suite變綠而立刻重跑40分鐘。若要完成每個cap
兩次合格結果，仍需另行宣告補測，不能直接把資料拼回原suite宣稱完整。
低cap切換動態、模型可用性與MPC增益尚未建立；C保持未收集／未評分。

完整方法、圖表、paired結果與重現指令見[英文報告](report.md)。
