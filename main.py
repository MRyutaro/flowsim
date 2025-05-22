import yaml
from typing import Dict, List
import sys
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field


class Node(BaseModel):
    id: str
    initial: float = Field(ge=0.0)
    capacity: float = Field(ge=0.0)
    process_factor: float = Field(ge=0.0, le=1.0)
    current: float = Field(default=0.0, ge=0.0)


class Edge(BaseModel):
    source: str
    target: str
    transfer_rate: float = Field(ge=0.0, le=1.0)
    loss: float = Field(ge=0.0, le=1.0)


class FlowSimulator:
    def __init__(self, nodes: Dict[str, Node], edges: List[Edge]):
        self.nodes = nodes
        self.edges = edges
        self.current_step = 0
        self.previous_states = {}  # 前回の状態を保存
        self.stability_threshold = 1e-4  # 安定状態とみなす閾値
        self.loss_history = []  # 各ステップでの損失を記録
        self.display_steps = 20  # 表示するステップ数

        # 初期水量を設定
        for node in self.nodes.values():
            node.current = node.initial
            self.previous_states[node.id] = node.initial

        # 描画の初期設定
        plt.ion()  # インタラクティブモードをオン
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            1, 2, figsize=(12, 6)
        )  # 2つのサブプロットを作成
        self.ax1.set_position([0.02, 0.1, 0.55, 0.8])  # ネットワークグラフ領域を大きく
        self.ax2.set_position([0.65, 0.1, 0.3, 0.8])  # 損失グラフ領域を右に寄せて小さく
        self.pos = None  # ノードの位置を保持

    def is_stable(self) -> bool:
        """システムが安定状態に達したかチェック"""
        is_stable = True
        for node_id, node in self.nodes.items():
            # 前回の状態との差分を計算
            diff = abs(node.current - self.previous_states[node_id])
            if diff > self.stability_threshold:
                is_stable = False
            # 現在の状態を保存
            self.previous_states[node_id] = node.current
        return is_stable

    def step(self):
        self.current_step += 1
        total_loss = 0.0  # このステップでの総損失
        transfers = []  # 一時的に転送量を保存

        for edge in self.edges:
            source_node = self.nodes[edge.source]
            target_node = self.nodes[edge.target]

            # 転送量を計算
            transfer = source_node.current * edge.transfer_rate
            loss = transfer * edge.loss  # この経路での損失
            total_loss += loss  # 総損失に加算
            after_loss = transfer * (1 - edge.loss)
            processed = after_loss * target_node.process_factor

            transfers.append((edge.source, edge.target, transfer, processed))

        # 損失履歴に記録
        self.loss_history.append(total_loss)
        print(f"total_loss: {total_loss}")

        # 各ノードの水量を更新
        for source_id, target_id, transfer, processed in transfers:
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]

            # 減少
            source_node.current = max(0.0, source_node.current - transfer)

            # 増加（capacityを超えない範囲で）
            target_node.current = min(
                target_node.current + processed, target_node.capacity
            )

    def visualize(self):
        # ネットワークグラフの描画
        self.ax1.clear()

        G = nx.DiGraph()

        # ノードを追加
        node_colors = []  # ノードの色を格納するリスト
        for node_id, node in self.nodes.items():
            G.add_node(node_id, current=f"{node.current:.1f}", capacity=node.capacity)
            # 充填率を計算（0から1の範囲）
            fill_ratio = node.current / node.capacity if node.capacity > 0 else 0
            node_colors.append(fill_ratio)

        # エッジを追加
        for edge in self.edges:
            G.add_edge(
                edge.source,
                edge.target,
                transfer_rate=f"{edge.transfer_rate:.1f}",
                loss=f"{edge.loss:.1f}",
            )

        # 初回のみノードの位置を計算し、以降は同じ位置を使用
        if self.pos is None:
            self.pos = nx.spring_layout(G, k=1.5)  # ノード間の距離を広げる

        # ノードを描画（色は充填率に応じて変化）
        nx.draw_networkx_nodes(
            G,
            self.pos,
            node_color=node_colors,
            node_size=3000,
            ax=self.ax1,
            cmap=plt.cm.RdYlBu_r,  # 青から赤のカラーマップを使用
            vmin=0,
            vmax=1,
        )

        # エッジを描画
        nx.draw_networkx_edges(
            G,
            self.pos,
            edge_color="gray",
            width=2,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.2",
            ax=self.ax1,
            node_size=3000,
            min_source_margin=20,
            min_target_margin=20,
        )

        # ノードラベルを描画
        labels = {
            node: f"{node}\ncurrent: {data['current']}\ncapacity: {data['capacity']}\nprocess_factor: {self.nodes[node].process_factor}"
            for node, data in G.nodes(data=True)
        }
        nx.draw_networkx_labels(
            G,
            self.pos,
            labels,
            font_size=10,
            ax=self.ax1,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        )

        # エッジラベルを描画
        edge_labels = {
            (u, v): f"transfer_rate: {d['transfer_rate']}\nloss_rate: {d['loss']}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels, font_size=8)

        self.ax1.set_title(f"Water Flow Network State - Step {self.current_step}")
        self.ax1.axis("on")
        self.ax1.grid(True, linestyle="--", alpha=0.3)

        # 損失の時間変化グラフを描画
        self.ax2.clear()
        if self.loss_history:
            # 直近のデータのみを表示
            display_start = max(0, len(self.loss_history) - self.display_steps)
            display_data = self.loss_history[display_start:]
            display_steps = range(display_start + 1, len(self.loss_history) + 1)

            self.ax2.plot(
                display_steps,
                display_data,
                "b-",
                label="Water Loss",
            )

            yticks = plt.MaxNLocator(nbins=6).tick_values(
                min(display_data), max(display_data)
            )
            self.ax2.set_yticks(yticks)
            self.ax2.set_yticklabels([f"{y:.2f}" for y in yticks])

            # X軸の範囲を設定
            self.ax2.set_xlim(display_start + 1, len(self.loss_history) + 1)
        else:
            # 初期状態では0-1の範囲を表示
            self.ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            self.ax2.set_xlim(0, self.display_steps)

        self.ax2.set_xlabel("Step")
        self.ax2.set_ylabel("Water Loss")
        self.ax2.set_title("Water Loss Over Time")
        self.ax2.grid(True)
        self.ax2.legend()

        plt.draw()
        plt.pause(0.00001)  # グラフの更新を確実にする


def load_config(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    nodes = {}
    for node_id, props in config["nodes"].items():
        nodes[node_id] = Node(
            id=node_id,
            initial=props["initial"],
            capacity=props["capacity"],
            process_factor=props["process_factor"],
        )

    edges = []
    for edge_data in config["edges"]:
        edges.append(
            Edge(
                source=edge_data["source"],
                target=edge_data["target"],
                transfer_rate=edge_data["transfer_rate"],
                loss=edge_data["loss"],
            )
        )

    return nodes, edges


def main():
    if len(sys.argv) != 3:
        print("使用方法: python main.py <設定ファイル> <最大ステップ数>")
        sys.exit(1)

    config_file = sys.argv[1]
    max_steps = int(sys.argv[2])

    nodes, edges = load_config(config_file)
    simulator = FlowSimulator(nodes, edges)

    simulator.visualize()

    for _ in range(max_steps):
        simulator.step()
        simulator.visualize()

        # システムが安定状態に達したらループを終了
        if simulator.is_stable():
            print(
                f"\nシステムが安定状態に達しました（ステップ {simulator.current_step}）"
            )
            break
    else:
        print(f"\n最大ステップ数 {max_steps} に到達しました")


if __name__ == "__main__":
    main()
