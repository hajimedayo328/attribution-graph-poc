# attribution-graph-poc

LLMの思考回路（attribution graph）をグラフ理論で統計分析するPoC。
完全無料パイプライン: Neuronpedia生成API → JSON取得 → NetworkX解析。

## 何をしたか

gemma-2-2b のattribution graphを68枚生成し（6カテゴリ+言い換え実験）、グラフ統計・検定・分類・意味照合を実施。

主な結果（詳細は [RESULTS.md](RESULTS.md)）:

1. **重要ノード特定**: logit起点の個人化PageRankがcircuit-tracer公式の因果的重要度top10を6.7/10再現（出次数ベースラインは3.8/10）。順位相関では出次数と互角 → 大域構造は「ランキングの頭」にだけ効く
2. **架空プロンプトの構造シグネチャ**: トークン正規化後も「疎・ハブ集中」が有意（BH補正後q<0.05）。ただし要約統計での幻覚検出はAUC 0.756止まり
3. **Fact:プレフィックスは回路レベルのモード切替**: 描写系特徴が消え、参照・地理・固有名詞系特徴が入る（Neuronpedia auto-interp説明と照合）
4. **作話専用回路は存在しない**: 架空応答は通常回路の薄い使い方
5. **回路の同一性は言い回しに支配される**: 同じ言い回し・違う国のJaccard 0.668 vs 同じ国・違う言い回し 0.284（1-NNは12/12が言い回し側）

## ファイル構成

| ファイル | 内容 |
|---|---|
| `gen_batch.py` | Neuronpedia APIでグラフ量産（レジューム・429自動待機つき） |
| `analyze.py` | 単一グラフの基本統計 |
| `analyze_batch.py` | 全グラフの不変量・PageRank vs 次数・Mann-Whitney U + BH補正 |
| `followup.py` | トークン正規化検証・LOO分類（AUC） |
| `followup2.py` | レイヤー分布・最強経路・Jaccard類似度・1-NN |
| `followup3.py` | 共有回路の抽出とNeuronpedia説明APIによる意味照合 |
| `RESULTS.md` | 全実験の結果と限界 |
| `data/manifest.csv` | プロンプト一覧とS3 URL |
| `data/results*.csv`, `data/circuits.json` | 分析結果 |

グラフ本体（`data/batch/*.json`、約280MB）はリポジトリに含めない。`manifest.csv`のS3 URLから再取得するか、`gen_batch.py`で再生成できる。

## 再現方法

```bash
pip install networkx
python gen_batch.py        # グラフ生成（匿名APIは約28枚/日の制限あり）
python analyze_batch.py    # 統計分析
python followup.py
python followup2.py
python followup3.py
```

## 既知の罠

- グラフJSONの`influence`は枝刈り用累積スコアで**小さいほど重要**
- Neuronpedia特徴APIへの照会は`node_id`の中央部（層内index）を使う（`feature`フィールドは別エンコード）
- 匿名生成APIは約28枚/日でHTTP 429、解除は翌日

## 限界

n=10/群・単一モデル(gemma-2-2b)・自作プロンプト。ここでの「発見」はすべて研究の種であり、主張可能な結果ではない。attribution graph自体が置換モデルの近似である点にも注意。
