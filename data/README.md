# 資料來源與本機產物

原始資料為 10x Genomics 的 **PBMC3k**，透過
[`scanpy.datasets.pbmc3k()`](https://scanpy.scverse.org/en/stable/generated/scanpy.datasets.pbmc3k.html)
下載為 `raw/pbmc3k_raw.h5ad`。這是單一健康 donor 的外周血單核細胞資料，
原始矩陣為 2,700 個細胞 × 32,738 個基因；不假設有 cell-type 標籤。

- `raw/`：下載的原始 counts，不修改原始檔案。
- `processed/exploration.h5ad`：全資料探索用，不能直接用於模型特徵擬合。
- `processed/baseline.h5ad`：固定 QC、切分細胞、僅用 train 選基因後的模型資料。
  `.X` 是 normalize-total + log1p，`.layers['counts']` 是相同細胞／基因的 counts。
- 來源、原始檔 SHA-256、QC 門檻與資料形狀記在 `outputs/metrics/`。

資料檔不納入 Git；以程式重建。資料的使用條款依原始提供者，本專案的 MIT 授權僅涵蓋程式與說明。
