# viz_rerun.py
import time
from typing import Dict, Tuple, Optional
import rerun as rr
import networkx as nx
import numpy as np


class RerunViz_v1:
    """
    Visualización con Rerun:
    - "graph/nodes": nodos (server + clients) como Points2D con labels
    - "graph/edges": aristas client->server como LineStrips2D
    - "events": TextLog con cada conexión, desconexión y mensaje
    - "metrics/...": accuracy y loss (por cliente y global) como series temporales
    """

    def __init__(self, app_name: str = "FDL Server", spawn_viewer: bool = True):
        rr.init(app_name, spawn=spawn_viewer)
        rr.set_time_seconds("time", time.time())

        self.G = nx.DiGraph()
        self.server_node = "Server"
        self.G.add_node(self.server_node)
        self.pos_cache: Dict[str, Tuple[float, float]] = {}

        rr.log("events", rr.TextLog("Server visualization started ✅"))
        self._draw_graph()

    # ---------- helpers ----------
    def _sanitize(self, label: str) -> str:
        """Convierte etiquetas en rutas válidas para Rerun (sin espacios, ':' o '/')."""
        return label.replace(" ", "_").replace(":", "_").replace("/", "_")

    def _stamp(self):
        rr.set_time_seconds("time", time.time())

    def _draw_graph(self):
        self._stamp()
        pos = nx.spring_layout(self.G, seed=42, k=0.8)
        self.pos_cache = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}

        nodes = list(self.G.nodes)
        pts = np.array([self.pos_cache[n] for n in nodes], dtype=np.float32)
        rr.log("graph/nodes", rr.Points2D(pts, labels=nodes))

        if self.G.number_of_edges() > 0:
            segments = [
                np.array([self.pos_cache[u], self.pos_cache[v]], dtype=np.float32)
                for u, v in self.G.edges
            ]
            rr.log("graph/edges", rr.LineStrips2D(segments))

    # ---------- graph / events ----------
    def add_client(self, client_label: str):
        self._stamp()
        if client_label not in self.G:
            self.G.add_node(client_label)
            self.G.add_edge(client_label, self.server_node)
            rr.log("events", rr.TextLog(f"👤 Client connected: {client_label}"))
            self._draw_graph()

    def remove_client(self, client_label: str):
        self._stamp()
        if client_label in self.G:
            self.G.remove_node(client_label)
            rr.log("events", rr.TextLog(f"🧹 Client removed: {client_label}"))
            self._draw_graph()

    def log_msg_received(self, client_label: str, nbytes: int):
        self._stamp()
        rr.log("events", rr.TextLog(f"📥 FROM {client_label} → Server | {nbytes} bytes"))
        if client_label in self.pos_cache and self.server_node in self.pos_cache:
            seg = np.array(
                [self.pos_cache[client_label], self.pos_cache[self.server_node]],
                dtype=np.float32,
            )
            rr.log("graph/edges/active_rx", rr.LineStrips2D([seg]))

    def log_msg_sent(self, client_label: str, nbytes: int):
        self._stamp()
        rr.log("events", rr.TextLog(f"📤 FROM Server → {client_label} | {nbytes} bytes"))
        if client_label in self.pos_cache and self.server_node in self.pos_cache:
            seg = np.array(
                [self.pos_cache[self.server_node], self.pos_cache[client_label]],
                dtype=np.float32,
            )
            rr.log("graph/edges/active_tx", rr.LineStrips2D([seg]))

    def log_accuracy(self, client_label: str, acc: float, global_acc: float):
        self._stamp()
        rr.log(
            "events",
            rr.TextLog(f"📌 Acc {client_label}: {acc:.2f}% | Global: {global_acc:.2f}%"),
        )

    # ---------- metrics (time series) ----------
    def log_metrics(
        self,
        client_label: str,
        acc: Optional[float] = None,
        loss: Optional[float] = None,
        global_acc: Optional[float] = None,
        global_loss: Optional[float] = None,
    ):
        """
        Registra métricas en rutas:
        - metrics/accuracy/<cliente>
        - metrics/loss/<cliente>
        - metrics/accuracy/global
        - metrics/loss/global
        """
        self._stamp()
        cl = self._sanitize(client_label)

        if acc is not None:
            rr.log(f"metrics/accuracy/{cl}", rr.Scalar(float(acc)))
        if loss is not None:
            rr.log(f"metrics/loss/{cl}", rr.Scalar(float(loss)))

        if global_acc is not None:
            rr.log("metrics/accuracy/global", rr.Scalar(float(global_acc)))
        if global_loss is not None:
            rr.log("metrics/loss/global", rr.Scalar(float(global_loss)))

    def log_note(self, text: str):
        self._stamp()
        rr.log("events", rr.TextLog(text))
