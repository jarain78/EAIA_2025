# server_rerun_fdl.py
# Servidor de FL centralizado con visualización Rerun de clientes y mensajes.

import socket
import pickle
import threading
import struct
import torch
import numpy as np
import os
import time

from viz_rerun import RerunViz

# ===================== Configuración =====================
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5001
STORAGE_DIR = "FDL_Centralized_FrameWork_v1/Models_Temps/model_temp_storage"
LOG_DIR = "FDL_Centralized_FrameWork_v1/Log/"

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def print_colored(message, color_code):
    """Print con color en terminal."""
    print(f"\033[{color_code}m{message}\033[0m")


def recv_all(sock, size):
    """Recibe datos del cliente en bloques hasta completar 'size' bytes."""
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


class Server:
    def __init__(self, strategy: str = "FedAvg"):
        # Socket TCP
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((SERVER_HOST, SERVER_PORT))
        self.server_socket.listen(5)
        print_colored(f"🔹 Server listening on {SERVER_HOST}:{SERVER_PORT} ...", "34")

        # Visualización Rerun
        self.viz = RerunViz(app_name="FDL Server", spawn_viewer=True)
        self.viz.log_note("🚀 Server started; waiting for clients...")

        # Estado del servidor
        self.strategy = strategy.upper()  # "FEDAVG", "OWA", "FEDPROX", "FEDNOVA"
        self._aggregation_mode = self.strategy
        self.clients = []      # sockets "activos" de clientes que han enviado modelo
        self.models = []       # lista de dicts de pesos recibidos
        self.lock = threading.Lock()
        self.global_accuracy = 0.0

        self.start_training_time = time.time()  # t0 entrenamiento
        self.start_wait_time = None             # inicio de espera de modelos por ronda

        # Logs por estrategia
        self.WAIT_TIME_LOG = os.path.join(LOG_DIR, f"wait_time_log_{self._aggregation_mode}.txt")
        self.ACCURACY_LOG = os.path.join(LOG_DIR, f"accuracy_log_{self._aggregation_mode}.txt")

        # Particionado (si se usa)
        self.dataset_partitions = {}

    # ----------------- Aggregation mode dinámico -----------------
    def update_log_paths(self):
        self.WAIT_TIME_LOG = os.path.join(LOG_DIR, f"wait_time_log_{self._aggregation_mode}.txt")
        self.ACCURACY_LOG = os.path.join(LOG_DIR, f"accuracy_log_{self._aggregation_mode}.txt")

    @property
    def aggregation_mode(self):
        return self._aggregation_mode

    @aggregation_mode.setter
    def aggregation_mode(self, mode):
        print_colored(f"🔄 Updating aggregation mode to: {mode}", "36")
        self._aggregation_mode = mode
        self.update_log_paths()

    # ----------------- Estrategias de agregación -----------------
    def apply_owa(self, weights_list):
        """OWA sobre un conjunto de tensores numpy (todos misma forma)."""
        print_colored("🔵 [OWA] Applying OWA aggregation...", "36")
        weights_array = np.array(weights_list)
        # ordena descendente a lo largo del eje 0
        X_sorted = -np.sort(-weights_array, axis=0)
        # pesos OWA de ejemplo para 3 clientes; si hay más/menos, ajusta
        k = X_sorted.shape[0]
        if k == 1:
            return X_sorted[0]
        if k == 2:
            owa_weights = np.array([0.7, 0.3], dtype=np.float32)
        else:
            # simple triangular para k>=3
            owa_weights = np.linspace(1.0, 0.1, num=k, dtype=np.float32)
            owa_weights /= owa_weights.sum()
        owa_weights = owa_weights.reshape(k, *([1] * (X_sorted.ndim - 1)))
        aggregated = np.sum(X_sorted * owa_weights, axis=0)
        return aggregated

    def apply_fedavg(self, weights_list):
        """FedAvg clásico (media)."""
        print_colored("🟢 [FedAvg] Applying FedAvg aggregation...", "32")
        return np.mean(weights_list, axis=0)

    def apply_fedprox(self, weights_list, global_weights, mu=0.01):
        """FedProx: media - mu * media(prox_term)."""
        print_colored("🔵 [FedProx] Applying FedProx aggregation...", "36")
        global_weights_np = np.array(global_weights)
        wl = np.array(weights_list)
        prox_term = wl - global_weights_np
        prox_adjusted = np.mean(wl, axis=0) - mu * np.mean(prox_term, axis=0)
        return prox_adjusted

    def apply_fednova(self, weights_dicts_list, num_steps_list):
        """
        FedNova para listas de dicts (una entrada por cliente).
        weights_dicts_list: [ {k: tensor}, {k: tensor}, ... ]
        num_steps_list: lista del #pasos/cliente (misma longitud).
        Devuelve un dict k->tensor agregado.
        """
        print_colored("🔵 [FedNova] Applying FedNova aggregation...", "36")
        assert len(weights_dicts_list) == len(num_steps_list) and len(weights_dicts_list) > 0

        total_steps = float(np.sum(num_steps_list))
        coeffs = [s / total_steps for s in num_steps_list]

        # Inicializa dict con ceros de la forma de la primera entrada
        aggregated = {k: torch.zeros_like(next(iter(weights_dicts_list[0].values()))) for k in weights_dicts_list[0]}
        # Suma ponderada por #pasos
        for wdict, c in zip(weights_dicts_list, coeffs):
            for k in wdict:
                aggregated[k] = aggregated[k] + (wdict[k] * c)
        return aggregated

    # ----------------- Particionado (opcional, no usado por defecto) -----------------
    def partition_dataset(self):
        """Divide un dataset hipotético entre los clientes conectados."""
        num_clients = len(self.clients)
        if num_clients == 0:
            print_colored("⚠️ No clients connected. Skipping dataset partitioning.", "33")
            return
        dataset_size = 100_000
        samples_per_client = dataset_size // num_clients
        partition_info = {}
        start_idx = 0
        for idx, client in enumerate(self.clients):
            end_idx = start_idx + samples_per_client
            partition_info[idx] = (start_idx, end_idx)
            start_idx = end_idx
        self.dataset_partitions = partition_info
        print_colored(f"🔹 Dataset partitioned for {num_clients} clients.", "36")

    # ----------------- Utilidades de tiempo y logs -----------------
    def calculate_wait_time(self):
        """Calcula tiempo total de espera en la ronda y lo persiste."""
        if self.start_wait_time:
            elapsed_time = time.time() - self.start_wait_time
            print_colored(f"⏳ [WAIT TIME] Total time waiting for models: {elapsed_time:.2f}s", "33")
            with open(self.WAIT_TIME_LOG, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Wait time: {elapsed_time:.2f}s\n")

    def calculate_total_training_time(self):
        total_time = time.time() - self.start_training_time
        print_colored(f"🚀 [TOTAL TIME] Training took {total_time:.2f}s to reach 93% accuracy.", "32")
        with open(self.WAIT_TIME_LOG, "a") as f:
            f.write(f"\n[COMPLETED] Total training time: {total_time:.2f}s\n\n")

    def save_aggregated_model(self, aggregated_weights):
        """Guarda el modelo agregado con timestamp."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(STORAGE_DIR, f"aggregated_model_{ts}.pth")
        torch.save(aggregated_weights, model_path)
        print_colored(f"💾 [SAVED] Aggregated model stored in {model_path}", "36")

    def log_accuracy(self, client_label: str, accuracy: float):
        """Persiste accuracy por cliente y global."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} - {client_label}: {accuracy:.2f}% | Global Accuracy: {self.global_accuracy:.2f}%\n"
        with open(self.ACCURACY_LOG, "a") as f:
            f.write(line)
        print_colored(f"📌 [LOG] {line.strip()}", "35")

    # ----------------- Agregación y envío -----------------
    def aggregate_models(self):
        """Agrega pesos recibidos según la estrategia y hace broadcast."""
        print_colored(f"🔵 [AGGREGATION] Using strategy: {self.strategy}", "36")

        if self.strategy == "FEDNOVA":
            # Ejemplo: si los clientes envían también num_steps, aquí habría que mantenerlo.
            # Para simplificar, usamos pasos ficticios = 1 para todos.
            num_steps_list = [1 for _ in self.models]
            aggregated_weights = self.apply_fednova(self.models, num_steps_list)
        else:
            # Para el resto de estrategias, operamos tensor a tensor (clave por clave).
            keys = list(self.models[0].keys())
            aggregated_weights = {}
            # Global (para FedProx)
            global_model_weights = {k: torch.zeros_like(self.models[0][k]) for k in keys}

            for k in keys:
                weights_list_np = [m[k].cpu().numpy() for m in self.models]
                if self.strategy == "OWA":
                    combined = self.apply_owa(weights_list_np)
                elif self.strategy == "FEDAVG":
                    combined = self.apply_fedavg(weights_list_np)
                elif self.strategy == "FEDPROX":
                    combined = self.apply_fedprox(weights_list_np, global_model_weights[k].cpu().numpy())
                else:
                    # fallback a FedAvg
                    combined = self.apply_fedavg(weights_list_np)
                aggregated_weights[k] = torch.tensor(combined, dtype=torch.float32)

        # Guardado y limpieza de ronda
        self.save_aggregated_model(aggregated_weights)
        self.calculate_wait_time()
        self.models.clear()
        self.start_wait_time = None

        # Envío a clientes
        self.broadcast(aggregated_weights)

    def broadcast(self, aggregated_weights):
        """Envía los pesos agregados a todos los clientes que participaron en la ronda."""
        print_colored("🟣 [BROADCAST] Sending aggregated weights to all clients...", "35")
        serialized = pickle.dumps(aggregated_weights)
        header = struct.pack('!I', len(serialized))
        for client in list(self.clients):
            try:
                client.sendall(header + serialized)
                # Log visual de envío
                try:
                    addr = client.getpeername()
                    clabel = f"{addr[0]}:{addr[1]}"
                    self.viz.log_msg_sent(clabel, len(header) + len(serialized))
                except Exception:
                    pass
            except (ConnectionResetError, BrokenPipeError):
                print_colored("🔴 [ERROR] Could not send data to a client.", "31")
        # Limpia la lista; nueva ronda creará nueva lista.
        self.clients.clear()

    # ----------------- Negociación de estrategia -----------------
    def negotiate_strategy_with_client(self, client_socket):
        """Negocia estrategia con el cliente y responde."""
        try:
            header = recv_all(client_socket, 4)
            if not header:
                print_colored("⚠️ [Negotiation] No strategy header received.", "33")
                return
            size = struct.unpack('!I', header)[0]
            data = recv_all(client_socket, size)
            if not data:
                print_colored("⚠️ [Negotiation] No strategy received from client.", "33")
                return

            proposed = pickle.loads(data)
            if isinstance(proposed, str):
                proposed_strategy = proposed.upper()
            else:
                proposed_strategy = str(proposed).upper()

            print_colored(f"🤝 [Negotiation] Client proposes strategy: {proposed_strategy}", "36")

            valid = {"FEDAVG", "OWA", "FEDPROX", "FEDNOVA"}
            if proposed_strategy in valid:
                self.strategy = proposed_strategy
                self.aggregation_mode = proposed_strategy
                response = f"✅ Strategy '{proposed_strategy}' accepted."
            else:
                response = f"❌ Strategy '{proposed_strategy}' not valid. Keeping: {self.strategy}"

            resp = pickle.dumps(response)
            client_socket.sendall(struct.pack('!I', len(resp)) + resp)
        except Exception as e:
            print_colored(f"🔴 [Negotiation] Error: {e}", "31")

    # ----------------- Bucle por cliente -----------------
    def handle_client(self, client_socket):
        client_label = None
        try:
            client_address = client_socket.getpeername()  # (ip, port)
            client_label = f"{client_address[0]}:{client_address[1]}"
            print_colored(f"🔹 Client connected from {client_address}.", "34")
            self.viz.add_client(client_label)

            # Negocia estrategia (opcional).
            self.negotiate_strategy_with_client(client_socket)

            while self.global_accuracy < 93.0:
                if self.start_wait_time is None:
                    self.start_wait_time = time.time()

                header = recv_all(client_socket, 4)
                if not header:
                    break
                data_size = struct.unpack('!I', header)[0]
                data = recv_all(client_socket, data_size)
                if not data:
                    break

                # Log visual de bytes recibidos
                self.viz.log_msg_received(client_label, data_size)

                # Se espera (weights_dict, accuracy_float)
                payload = pickle.loads(data)
                if isinstance(payload, tuple) and len(payload) >= 2:
                    model_weights, accuracy = payload[0], float(payload[1])
                else:
                    print_colored("🔴 [ERROR] Invalid payload format from client.", "31")
                    continue

                # (Opcional) imprime tamaños
                print_colored(f"🔹 Weights received from {client_label}:", "36")
                for key, weight in model_weights.items():
                    try:
                        shape = tuple(weight.shape)
                    except Exception:
                        shape = "<?>"
                    print_colored(f"   • {key}: {shape}", "36")

                # Actualiza estado
                with self.lock:
                    self.global_accuracy = max(self.global_accuracy, accuracy)
                    # Asegurar que son tensores torch
                    model_weights_torch = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v))
                                           for k, v in model_weights.items()}
                    self.models.append(model_weights_torch)
                    if client_socket not in self.clients:
                        self.clients.append(client_socket)

                # Logs (archivo y visor)
                self.log_accuracy(client_label, accuracy)
                self.viz.log_accuracy(client_label, accuracy, self.global_accuracy)

                print_colored(f"🟡 [UPDATE] Global Accuracy: {self.global_accuracy:.2f}%", "33")

                # Aviso cuando alguien alcanza 93%
                if accuracy >= 93.0:
                    note = f"🔥 Client {client_label} reached 93% accuracy!"
                    print_colored(note, "32")
                    self.viz.log_note(note)

                # Si ya tenemos modelos de todos los clientes activos en la ronda → agregamos
                with self.lock:
                    if len(self.models) == len(self.clients) and len(self.models) > 0:
                        self.aggregate_models()

        except (ConnectionResetError, BrokenPipeError):
            print_colored("🔴 Client disconnected abruptly.", "31")
        except Exception as e:
            print_colored(f"🔴 Server error: {e}", "31")
        finally:
            try:
                client_socket.close()
            finally:
                if client_label:
                    self.viz.remove_client(client_label)

    # ----------------- Main loop -----------------
    def run(self):
        try:
            while self.global_accuracy < 93.0:
                client_socket, _ = self.server_socket.accept()
                t = threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True)
                t.start()
        except KeyboardInterrupt:
            print_colored("🛑 Server interrupted by user.", "33")
        finally:
            self.calculate_total_training_time()
            print_colored("🚀 [TRAINING COMPLETED] Global accuracy > 93%", "32")


if __name__ == "__main__":
    # strategy por defecto: "FedAvg"
    strategy = "FedAvg"
    server = Server(strategy=strategy)
    server.aggregation_mode = strategy
    server.run()
