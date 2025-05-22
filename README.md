# 💧 FlowSim: 単一ノード型水フローシミュレータ - CLI版 開発資料

## 🧭 概要

ノード（全て同一種類）とエッジ（全て同一種類）から構成されるネットワーク上において、水がどのように流れ、蓄積されるかをシミュレーションするツール。CLIベースで実行・観察可能。

## 🎯 目的

* ネットワーク構造とパラメータ調整によってフローがどう変化するかを可視化する
* シンプルな構成で設計・デバッグ・ロジック理解のトレーニングを行う

## 🧱 基本構造

### ノード（Node）

* `id`: 識別子（文字列）
* `initial`: 初期水量（float）
* `capacity`: 最大保持水量（float）
* `process_factor`: 入力水量に対して何割を保持・通過させるか（0〜1）

### エッジ（Edge）

* `source`: 出発ノードID
* `target`: 到着ノードID
* `transfer_rate`: 流す比率（0〜1）
* `loss`: 流れる途中で失われる割合（0〜1）

## 📥 入力形式（YAMLまたはJSON想定）

```yaml
nodes:
  A: { initial: 100.0, capacity: 100.0, process_factor: 1.0 }
  B: { initial: 0.0, capacity: 80.0, process_factor: 1.0 }
  C: { initial: 0.0, capacity: 60.0, process_factor: 1.0 }

edges:
  - { source: A, target: B, transfer_rate: 0.5, loss: 0.1 }
  - { source: B, target: C, transfer_rate: 1.0, loss: 0.0 }
```

## 🔁 シミュレーションルール

1. 各ステップで全エッジについて水を流す：

   * `transfer = source.current * transfer_rate`
   * `after_loss = transfer * (1 - loss)`
   * `processed = after_loss * target.process_factor`
2. 各ノードに対して水量を更新：

   * 減少：source.current -= transfer
   * 増加：target.current += min(processed, capacity - current)

※ overflow（水量がcapacityを超えた分）は破棄

## 🔄 シミュレーション設定

* `steps`: 実行ステップ数（整数）

## 📤 出力

* 各ステップ終了後、全ノードの水量を表示
* 最終的な水量の分布を表示（表形式）

## 🧪 発展計画（今後の拡張用メモ）

* GUI / 可視化（matplotlib、pygame）
* ノード種類の導入（変換系・蓄積系など）
* CSV/JSON出力形式
* ユーザーインタラクションによるリアルタイム調整
