# CAPI：Compact Algebraic Projection Interface  
**一個可插拔、低成本的 Lie Algebra 正則化，用於幾何感知的細粒度視覺分類（FGVC）**

> 定位（走 NeurIPS 風格，而非衝榜 SOTA）：  
> 我們主張 **「一致性增益 + 泛化性 + 幾乎不增加推論成本」**，用跨資料集、跨 backbone、跨 seed 的證據與完整分析來支持貢獻，而不是只在單一設定硬拼排行榜。

---

## 1. 一句話 Pitch
**CAPI 是一個 plug-and-play 的訓練期正則化介面：把 backbone 特徵投影到 $\mathfrak{so}(m)$（SO(m) 的切空間 / Lie algebra）中，利用反對稱結構與交換子（commutator）約束「小姿態變化的可組合性」，從而在多資料集、多架構上帶來穩定的小幅提升，且推論幾乎零額外成本。**

---

## 2. 研究動機：FGVC 缺的是「可組合的幾何偏置」
FGVC 常見難點：
- 類內差異極小、但視角/姿態/局部形變造成的外觀變化很大。
- 許多方法在歐氏特徵空間用 attention / pooling 做更強辨識，但**缺少對「幾何變換如何組合」的結構性約束**。
- 走黎曼流形（SPD/Log-Euclidean 等）雖有幾何味道，但通常伴隨昂貴操作（如特徵分解）或架構限制，難以成為通用模組。

**核心問題**  
> 能不能用「足夠便宜、足夠通用」的方式，把幾何結構引入 FGVC 訓練，讓模型對姿態變化更穩定，且不犧牲部署？

---

## 3. 關鍵洞見：切空間比流形更適合當通用正則化
- 物體姿態/局部形變可視為一連串小變換的組合（composition）。
- 群（group）的乘法在流形上是非線性的，但在 **Lie algebra（切空間）** 可被線性化近似；更重要的是，**交換子 $[X, Y] = XY - YX$** 提供了「變換不可交換性」的結構訊號。
- 因此我們不直接在流形上做昂貴運算，而是：
  1) 把特徵映射到 $\mathfrak{so}(m)$ 的反對稱矩陣  
  2) 用簡單矩陣運算設計結構性 loss

---

## 4. 方法概述：CAPI 模組與 Loss 設計

### 4.1 CAPIProjection：投影到 $\mathfrak{so}(m)$
給定 backbone 特徵 $f \in \mathbb{R}^{d}$，用一個線性層投影到 $m^2$ 維並 reshape 成矩陣 $A \in \mathbb{R}^{m \times m}$，再做反對稱化：

$$
X = A - A^\top \quad\Rightarrow\quad X^\top = -X
$$

此時 $X \in \mathfrak{so}(m)$，是一個**切空間表示**。

> 重要：這個投影是「介面」而非新 backbone。CAPI 的目的不是重建主表徵，而是為訓練提供一個結構性約束通道。

---

### 4.2 Lie-Structure Loss（訓練期輔助正則化）
我們用兩類訊號來「讓表示更像小變換的切向量」：

1) **幾何可組合性（commutator-based）**  
用交換子鼓勵某些變化具有一致的組合行為（例如：不同視角變化的相容性 / 非交換性特徵），使表示捕捉到「變換的結構」，而不只是一個距離度量。

2) **類別分離/一致性（class separation / consistency）**  
在 $\mathfrak{so}(m)$ 表示上加入簡單的類內凝聚、類間分離或 margin 類的項，作為輕量替代或補強（不強迫大量負樣本的對比學習設定）。

最終訓練目標：
$$
\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{Lie}
$$

---

## 5. Plug-and-Play 與部署友善（NeurIPS-style 的賣點）
### 5.1 架構無關（architecture-agnostic）
- 任何輸出 feature 的 backbone 都能接（ResNet / ConvNeXt / ViT…）。
- CAPI 不改 backbone 的主 forward 邏輯，只在訓練時額外算一個投影與 loss。

### 5.2 幾乎零推論成本（inference-friendly）
- **推論時可以完全不使用 CAPI 分支**：只保留原本分類 head 的 logits。
- CAPI 是訓練期的 inductive bias / regularizer，不是部署依賴。

### 5.3 計算成本極低
- 核心運算為線性層 + reshape + $A - A^\top$ 與少量矩陣乘加。
- 不需要 SVD / eigendecomposition / log-map 等昂貴幾何運算。

---

## 6. 論文貢獻（用「一致性 + 分析」替代「SOTA」）
我們將貢獻寫成以下三點（符合 NeurIPS 風格）：

1. **CAPI：一個可插拔的 Lie algebra 正則化介面**  
   將任意 backbone 的特徵映射到 $\mathfrak{so}(m)$ 切空間，提供通用的幾何結構約束入口。

2. **基於交換子的結構性約束，用極低成本引入「可組合的幾何偏置」**  
   透過 commutator 捕捉小變換的組合特性，強化 FGVC 對姿態/視角變化的穩定性。

3. **全面實驗證據：跨資料集、跨架構、跨 seed 的穩定增益 + 成本分析 + 消融與可視化**  
   以「一致性提升」與「部署友善」為主軸，而不是單點衝榜。

---

## 7. 初步結果（可選放在 idea 或 draft）
目前在固定設定與多 seed 下，CAPI 相對於純 CE baseline 有穩定的小幅提升（約 +0.4% ~ +0.8% Top-1 的等級）。  
> 這類幅度若要投 NeurIPS-style，重點不在數字多大，而在「跨設定是否一致」與「分析是否完整」。

---

## 8. 實驗設計（NeurIPS-style：廣度 + 分析深度）

### 8.1 多資料集（FGVC 為主，必要時外推）
- CUB-200-2011
- NABirds
- Stanford Cars  
（可選）iNat 子集 / Aircraft（視時間與資源）

### 8.2 多 backbone（至少 2–3 種）
- ResNet-50（經典 CNN baseline）
- ConvNeXt-T / Swin-T（現代 CNN/Transformer-ish）
- ViT-B/16（純 Transformer）

### 8.3 公平比較：強 baseline + 同訓練 recipe
- 同資料增強、同 batch size、同 LR schedule、同訓練 epoch、同 seed protocol
- 報告 mean ± std（至少 3 seed；理想 5 seed）

### 8.4 代表性對照（不必追最新榜單，但要合理）
- Baseline：Cross-Entropy（必要）
- 同類正則/度量：Center loss / Triplet / SupCon / Margin-based（選 2–3 個即可）
- （可選）FGVC 常見技巧：bilinear pooling / second-order pooling 的輕量版本（若能穩定重現再放）

### 8.5 消融研究（Ablation）
- $\lambda$ 掃描：增益/穩定性 vs 正則強度
- $m$（Lie 維度）掃描：成本/表現 trade-off
- 拆解 loss：只反對稱投影 / 只 commutator / 只 separation / 全部
- 使用/不使用特定資料增強時的差異（檢驗是否互補）

### 8.6 成本與部署分析（必做）
- Training overhead：每 epoch 時間、GPU memory
- Inference overhead：  
  - **保留 CAPI 分支 vs 丟棄 CAPI 分支（部署建議）**  
  - 強調「丟棄後幾乎零成本」這個賣點

### 8.7 可視化與洞見（用來補足非 SOTA 的說服力）
- 類間/類內距離分佈（在 $\mathfrak{so}(m)$ 空間）
- 「特徵軌跡」：同一物體多視角在 CAPI 空間是否更平滑、更可解釋
- 最近鄰檢索案例：展示視角變化下的魯棒性提升

---

## 9. 可能限制與誠實敘事（加分項）
- CAPI 的增益可能是「小而穩定」，不一定在所有資料集都大幅領先。
- 對於幾何變化較少或資料偏差較強的情境，效果可能較不顯著。
- 我們將用消融與分析界定適用範圍（這是 NeurIPS reviewer 常看的成熟度指標）。

---

## 10. 參考方向（草案）
- Lie group / Lie algebra 在幾何建模與視覺表徵中的應用
- FGVC 的常見結構化表示、二階統計、attention 等方法
- Contrastive / metric learning 作為對照組的基礎文獻

（正式投稿時再補完整引用）

---
