# viz_rerun.py
import time
from typing import Dict, Tuple
import rerun as rr
import networkx as nx
import numpy as np


class RerunViz:
    """
    Visualización con Rerun:
    - "graph/nodes": nodos (server + clients) como Points2D con labels
    - "graph/edges": aristas client->server como LineStrips2D
    - "events": TextLog con cada conexión, desconexión y mensaje
    """

    def __init__(self, app_name: str = "FDL Server", spawn_viewer: bool = True):
        # Inicia Rerun y lanza el viewer en otra ventana (spawn=True).
        rr.init(app_name, spawn=spawn_viewer)

        # Marca temporal inicial (reloj de pared).
        rr.set_time_seconds("time", time.time())

        # Crea el grafo base con el nodo Server.
        self.G = nx.DiGraph()
        self.server_node = "Server"
        self.G.add_node(self.server_node)
        self.pos_cache: Dict[str, Tuple[float, float]] = {}

        # Primer log de arranque.
        rr.log("events", rr.TextLog("Server visualization started ✅"))

        # Dibujo inicial.
        self._draw_graph()

    def _stamp(self):
        """Marca temporal coherente para los logs."""
        rr.set_time_seconds("time", time.time())

    def _draw_graph(self):
        """Dibuja/actualiza el grafo completo en Rerun."""
        self._stamp()

        # Layout reproducible.
        pos = nx.spring_layout(self.G, seed=42, k=0.8)
        self.pos_cache = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}

        # Nodos y etiquetas.
        nodes = list(self.G.nodes)
        pts = np.array([self.pos_cache[n] for n in nodes], dtype=np.float32)
        labels = nodes
        rr.log("graph/nodes", rr.Points2D(pts, labels=labels))

        # Aristas client -> server.
        if self.G.number_of_edges() > 0:
            segments = []
            for u, v in self.G.edges:
                segments.append(
                    np.array([self.pos_cache[u], self.pos_cache[v]], dtype=np.float32)
                )
            rr.log("graph/edges", rr.LineStrips2D(segments))

    def add_client(self, client_label: str):
        """Añade un cliente al grafo y registra evento."""
        self._stamp()
        node = client_label
        if node not in self.G:
            self.G.add_node(node)
            # Flecha de comunicación "hacia" el server (direccional).
            self.G.add_edge(node, self.server_node)
            rr.log("events", rr.TextLog(f"👤 Client connected: {node}"))
            self._draw_graph()

    def remove_client(self, client_label: str):
        """Elimina un cliente (si se desconecta) y registra evento."""
        self._stamp()
        if client_label in self.G:
            self.G.remove_node(client_label)
            rr.log("events", rr.TextLog(f"🧹 Client removed: {client_label}"))
            self._draw_graph()

    def log_msg_received(self, client_label: str, nbytes: int):
        """Loguea un mensaje recibido desde un cliente."""
        self._stamp()
        rr.log(
            "events",
            rr.TextLog(f"📥 FROM {client_label} → Server | {nbytes} bytes"),
        )

        # Línea “activa” client->server (marcador temporal visual).
        if client_label in self.pos_cache and self.server_node in self.pos_cache:
            seg = np.array(
                [self.pos_cache[client_label], self.pos_cache[self.server_node]],
                dtype=np.float32,
            )
            rr.log("graph/edges/active_rx", rr.LineStrips2D([seg]))

    def log_msg_sent(self, client_label: str, nbytes: int):
        """Loguea un mensaje enviado a un cliente."""
        self._stamp()
        rr.log(
            "events",
            rr.TextLog(f"📤 FROM Server → {client_label} | {nbytes} bytes"),
        )

        # Línea “activa” server->client.
        if client_label in self.pos_cache and self.server_node in self.pos_cache:
            seg = np.array(
                [self.pos_cache[self.server_node], self.pos_cache[client_label]],
                dtype=np.float32,
            )
            rr.log("graph/edges/active_tx", rr.LineStrips2D([seg]))

    def log_accuracy(self, client_label: str, acc: float, global_acc: float):
        """Registro de métricas de accuracy."""
        self._stamp()
        rr.log(
            "events",
            rr.TextLog(f"📌 Acc {client_label}: {acc:.2f}% | Global: {global_acc:.2f}%"),
        )

    def log_note(self, text: str):
        """Texto libre."""
        self._stamp()
        rr.log("events", rr.TextLog(text))
