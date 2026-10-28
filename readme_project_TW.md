# 專案說明：Single-cell Transformer

[English](readme_project.md) | 繁體中文

## 這個專案在做什麼？

這是一個用真實單細胞 RNA 資料學習 **Scanpy、PyTorch 與 Transformer** 的小型專案。

核心任務是：**把一個細胞的部分基因表達值人為遮住，讓模型根據其他可見基因，預測被遮住位置原本的值。**

第一版的目標是把「讀資料 → 前處理 → 訓練 → 評估 → 儲存及載入模型」完整跑通，並用簡單基準檢查 Transformer 是否帶來改善。它是學習與實驗用的 baseline，目前尚未加入預訓練模型或 multi-omics。

## 資料長什麼樣子？

使用 PBMC3k 公開單細胞 RNA-seq 資料；資料來源與下載說明見 [data/README.md](data/README.md)。

| 資料元素      | 在這個專案中的意義                  |
| --------- | -------------------------- |
| 一列        | 一個細胞，也就是一筆樣本               |
| 一欄        | 一個基因，也就是一個 feature         |
| 原始矩陣的一個數值 | 該細胞中測到的該基因 RNA counts      |
| 模型使用的一個數值 | counts 經正規化與 `log1p` 後的表達值 |

原始資料有 **2,700 個細胞、32,738 個基因**，使用稀疏矩陣儲存大量零值。零值表示這次量測未偵測到 counts，不能直接推論該基因在生物學上完全沒有表達。

經本版固定的品質控制（QC）後保留 **2,638 個細胞**，再選出 **128 個基因**作為模型輸入。原始 counts 保留在資料層中，不會被處理後的值覆蓋。

## 模型到底要預測什麼？

以下只是示意，數值不是實驗結果；假設同一細胞有四個已處理的基因表達值：

| 基因 | 保留的正確答案（target） | 模型看到的值 | 遮蔽標記（mask） |
| -- | --------------: | -----: | ---------- |
| A  |             0.0 |    0.0 | False      |
| B  |             1.8 |    0.0 | True       |
| C  |             0.7 |    0.7 | False      |
| D  |             2.1 |    0.0 | True       |

模型根據基因身分、A／C 的可見值，以及遮蔽標記，預測 B／D 原本的值。A 的零值是已觀測值；B／D 的零只是遮蔽後的占位，兩者靠 mask 區分。

原本的 target 另外保留，模型的輸入端會把遮蔽位置清為零。訓練只計算 B／D 的誤差，不會因為複製其他可見值就拿到較低的 loss。

實際設定每個細胞遮住 **19/128** 個位置（約 15%），原本為零或非零的位置都有機會被選中。

## 完整流程

1. **載入與探索：**&#x67E5;看 AnnData 的矩陣、細胞 metadata 與基因 metadata，了解資料軸。
2. **品質控制：**&#x4FDD;留偵測到至少 200、少於 2,500 個基因，且粒線體 counts 比例低於 5% 的細胞。這是本版的教學門檻，不是所有資料通用的標準。
3. **切分細胞：**&#x74;rain 1,846、validation 396、test 396；random seed 為 42。
4. **前處理與選基因：**&#x6BCF;個細胞用自己的完整 library 總量縮放到 10,000，再取 `log1p`；僅用 train 決定基因篩選、HVG 排名與各基因平均值，取 128 個基因。
5. **訓練：**&#x6BCF;個 batch 重新抽遮蔽位置，執行 forward、計算 loss、backward、更新權重。
6. **選模型：**&#x76;alidation 使用固定 mask，依最低 validation MSE 選 checkpoint，包含作為起點的 epoch 0。
7. **最終評估：**&#x9078;完權重才用固定 test mask 比較 Transformer 與平均值基準，並確認儲存前後的預測一致。

資料探索 Notebook 的 HVG、PCA、UMAP 是全資料探索結果。模型訓練會重新從原始 counts 開始，避免把探索時擬合的全資料統計量帶入評估。

## Transformer 怎麼組成？

```text
基因 ID ───────────────→ Gene embedding ──────┐
                                            ├→ 相加 → Transformer Encoder → Prediction head
可見表達值 + 遮蔽標記 ─→ Value/mask encoding ─┘                                │
                                                                            ↓
                                                           加上 train gene mean → 預測表達值
```

- **Gene embedding：**&#x8B93;模型知道目前位置是哪個基因。
- **Value/mask encoding：**&#x628A;可見表達值和「是否被遮住」轉成向量。
- **Transformer Encoder：**&#x5229;用雙向 self-attention，計算同一細胞內不同基因輸入之間的關係。
- **Prediction head：**&#x6BCF;個基因位置輸出一個修正量，再加上 train gene mean 得到預測值。

本版使用 1 層 Encoder、embedding dimension 32、4 個 attention heads，共 **12,769 個可訓練參數**。基因身分由 embedding 表示，沒有替任意基因欄位順序加入 positional encoding，也不使用 causal mask。

一個 batch 的表達值與 mask 形狀是 `(16, 128)`，中間的 token 向量是 `(16, 128, 32)`，最後預測值回到 `(16, 128)`。

## 怎樣判斷模型有沒有用？

比較對象是 **gene-mean baseline**：不看同一細胞的其他基因，直接用該基因在 train 中的平均值作為預測。

兩個方法使用完全相同的資料切分、targets 與遮蔽位置。Transformer 的 head 從零初始化，因此 epoch 0 的輸出正好等於 gene-mean baseline；後續訓練嘗試學到依細胞內容而變的修正量。

| 指標         | 解讀                           |
| ---------- | ---------------------------- |
| Masked MSE | 只對被遮住位置計算平方誤差平均，用來訓練及選模型     |
| MAE        | 被遮住位置的絕對誤差平均，越低越好            |
| RMSE       | 被遮住位置的均方根誤差，較大的錯誤會有較大影響，越低越好 |

三者衡量的都是 **normalize-total + log1p 表達值**，不是原始 counts。評估同時列出全部、原本為零、原本非零三組，避免大量零值掩蓋重建困難。

## 如何執行？

首次建立環境與安裝方式見 [README\_TW.md](README_TW.md#快速開始)。已建立環境後，在專案根目錄執行：

```bash
source singlecell-venv/bin/activate

# 執行不需下載資料的正確性測試
python -m pytest -q

# 實際 PBMC3k：前處理、5 epochs 訓練、validation 選模型、test 評估
python -m singlecell_baseline.train

# 探索與 QC／PCA／UMAP 圖表
python -m singlecell_baseline.explore
```

也可執行 `jupyter lab`，依序閱讀並執行兩份 Notebook。

## 本次執行結果（2026-10-23）

在現有的 Python 3.11.2 虛擬環境、CPU 上，重新執行測試與完整 PBMC3k 訓練。兩個命令的 exit code 都是 0。

### 正確性測試

```text
.......                                                                  [100%]
7 passed in 4.63s
```

測試使用小型合成 counts，無須下載資料，涵蓋：held-out 值不會影響 train 擬合結果、切分互不重疊、遮蔽位置不洩漏 target、零／非零指標、一次訓練與 checkpoint 載入，以及非法遮蔽比例的檢查。這 7 項包含遮蔽比例檢查的 3 組參數案例。

完整原始輸出：[test\_output.txt](outputs/metrics/test_output.txt)。

### 真實 PBMC3k 訓練

```text
Epoch 01/5: train MSE=0.0684, val MSE=0.0559
Epoch 02/5: train MSE=0.0683, val MSE=0.0559
Epoch 03/5: train MSE=0.0705, val MSE=0.0559
Epoch 04/5: train MSE=0.0732, val MSE=0.0571
Epoch 05/5: train MSE=0.0646, val MSE=0.0561
Selected epoch: 0; artifacts: outputs
```

epoch 0 的 validation MSE 為 **0.05578436**。5 個訓練 epochs 都沒有低於這個值，因此選回等同平均值基準的 epoch 0；不是選用最後一個 epoch。

| Test 遮蔽位置 |   位置數 | Gene mean MAE | Transformer MAE | Gene mean RMSE | Transformer RMSE |
| --------- | ----: | ------------: | --------------: | -------------: | ---------------: |
| 全部        | 7,524 |      0.066361 |        0.066361 |       0.250510 |         0.250510 |
| 原本為零      | 7,376 |      0.033832 |        0.033832 |       0.039179 |         0.039179 |
| 原本非零      |   148 |      1.687506 |        1.687506 |       1.764610 |         1.764610 |

被遮蔽位置約 98% 原本為零，因此不能只根據整體誤差判斷模型有用。目前 Transformer **未優於平均值基準**，非零值的重建誤差也明顯較大。保存前後的 test 預測完全一致，`checkpoint_reload_identical` 為 `true`。

完整訓練輸出：[baseline\_output.txt](outputs/metrics/baseline_output.txt)。

![本次訓練的學習曲線](outputs/figures/learning_curve.png)

## 到哪裡看程式與輸出？

| 檔案／目錄                                                                                                   | 內容                               |
| ------------------------------------------------------------------------------------------------------- | -------------------------------- |
| [notebooks/01\_data\_exploration.ipynb](notebooks/01_data_exploration.ipynb)                            | 資料探索、QC、counts 與處理值的差異、PCA／UMAP  |
| [notebooks/02\_masked\_expression\_transformer.ipynb](notebooks/02_masked_expression_transformer.ipynb) | Tensor、遮蔽、梯度更新、完整訓練與評估           |
| [singlecell\_baseline/data.py](singlecell_baseline/data.py)                                             | QC、切分、train-only 特徵選擇與正規化        |
| [singlecell\_baseline/model.py](singlecell_baseline/model.py)                                           | 遮蔽、模型結構與 masked MSE              |
| [singlecell\_baseline/train.py](singlecell_baseline/train.py)                                           | 訓練、基準比較、評估與 checkpoint 讀寫        |
| [tests/test\_baseline.py](tests/test_baseline.py)                                                       | 前處理無資料洩漏、遮蔽、指標與訓練／載入測試           |
| [outputs/metrics/evaluation.csv](outputs/metrics/evaluation.csv)                                        | 各模型、split、零／非零分組的 MAE／RMSE       |
| [outputs/metrics/history.csv](outputs/metrics/history.csv)                                              | 每個 epoch 的 train／validation loss |
| [outputs/metrics/results.json](outputs/metrics/results.json)                                            | 模型設定、選定 epoch、版本與完整指標            |
| `outputs/figures/`                                                                                      | QC、PCA、UMAP 與學習曲線                |
| `outputs/checkpoints/best.pt`                                                                           | 本機 checkpoint，含權重與依序排列的基因清單      |

## 目前能說明什麼、不能說明什麼？

這個專案能展示單細胞資料處理、PyTorch Transformer 與可重跑評估的實作流程。測試通過代表程式的關鍵約束符合預期，不等於模型已優於基準。

目前任務是在完整量測資料完成正規化之後，再人為遮蔽數值；不能因此宣稱能恢復真實缺失的 RNA counts。UMAP 分群也不是獨立的 cell-type 標籤。單一 donor 的細胞切分結果，只能支持有限的資料內評估。
