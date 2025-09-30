# server_rerun_fdl.py
import socket
import pickle
import threading
import struct
import torch
import numpy as np
import os
import time

from viz_rerun_v1 import RerunViz_v1

# ===================== Configuración =====================
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5001
STORAGE_DIR = "FDL_Centralized_FrameWork_v1/Models_Temps/model_temp_storage"
LOG_DIR = "FDL_Centralized_FrameWork_v1/Log/"

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def print_colored(message, color_code):
    print(f"\033[{color_code}m{message}\033[0m")


def recv_all(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


class Server:
    def __init__(self, strategy: str = "FedAvg"):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((SERVER_HOST, SERVER_PORT))
        self.server_socket.listen(5)
        print_colored(f"🔹 Server listening on {SERVER_HOST}:{SERVER_PORT} ...", "34")

        self.viz = RerunViz_v1(app_name="FDL Server", spawn_viewer=True)
        self.viz.log_note("🚀 Server started; waiting for clients...")

        self.strategy = strategy.upper()
        self._aggregation_mode = self.strategy
        self.clients = []
        self.models = []
        self.lock = threading.Lock()
        self.global_accuracy = 0.0
        self.global_loss = float("inf")  # mínimo loss observado, si se reporta

        self.start_training_time = time.time()
        self.start_wait_time = None

        self.WAIT_TIME_LOG = os.path.join(LOG_DIR, f"wait_time_log_{self._aggregation_mode}.txt")
        self.ACCURACY_LOG = os.path.join(LOG_DIR, f"accuracy_log_{self._aggregation_mode}.txt")

        self.dataset_partitions = {}

    # ----- Aggregation mode dinámico -----
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

    # ----- Estrategias de agregación -----
    def apply_owa(self, weights_list):
        print_colored("🔵 [OWA] Applying OWA aggregation...", "36")
        weights_array = np.array(weights_list)
        X_sorted = -np.sort(-weights_array, axis=0)
        k = X_sorted.shape[0]
        if k == 1:
            return X_sorted[0]
        if k == 2:
            owa_weights = np.array([0.7, 0.3], dtype=np.float32)
        else:
            owa_weights = np.linspace(1.0, 0.1, num=k, dtype=np.float32)
            owa_weights /= owa_weights.sum()
        owa_weights = owa_weights.reshape(k, *([1] * (X_sorted.ndim - 1)))
        return np.sum(X_sorted * owa_weights, axis=0)

    def apply_fedavg(self, weights_list):
        print_colored("🟢 [FedAvg] Applying FedAvg aggregation...", "32")
        return np.mean(weights_list, axis=0)

    def apply_fedprox(self, weights_list, global_weights, mu=0.01):
        print_colored("🔵 [FedProx] Applying FedProx aggregation...", "36")
        global_weights_np = np.array(global_weights)
        wl = np.array(weights_list)
        prox_term = wl - global_weights_np
        prox_adjusted = np.mean(wl, axis=0) - mu * np.mean(prox_term, axis=0)
        return prox_adjusted

    def apply_fednova(self, weights_dicts_list, num_steps_list):
        print_colored("🔵 [FedNova] Applying FedNova aggregation...", "36")
        assert len(weights_dicts_list) == len(num_steps_list) and len(weights_dicts_list) > 0
        total_steps = float(np.sum(num_steps_list))
        coeffs = [s / total_steps for s in num_steps_list]
        aggregated = {k: torch.zeros_like(next(iter(weights_dicts_list[0].values()))) for k in weights_dicts_list[0]}
        for wdict, c in zip(weights_dicts_list, coeffs):
            for k in wdict:
                aggregated[k] = aggregated[k] + (wdict[k] * c)
        return aggregated

    # ----- Tiempos & logs -----
    def calculate_wait_time(self):
        if self.start_wait_time:
            elapsed = time.time() - self.start_wait_time
            print_colored(f"⏳ [WAIT TIME] Total time waiting for models: {elapsed:.2f}s", "33")
            with open(self.WAIT_TIME_LOG, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Wait time: {elapsed:.2f}s\n")

    def calculate_total_training_time(self):
        total_time = time.time() - self.start_training_time
        print_colored(f"🚀 [TOTAL TIME] Training took {total_time:.2f}s to reach 93% accuracy.", "32")
        with open(self.WAIT_TIME_LOG, "a") as f:
            f.write(f"\n[COMPLETED] Total training time: {total_time:.2f}s\n\n")

    def save_aggregated_model(self, aggregated_weights):
        ts = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(STORAGE_DIR, f"aggregated_model_{ts}.pth")
        torch.save(aggregated_weights, model_path)
        print_colored(f"💾 [SAVED] Aggregated model stored in {model_path}", "36")

    def log_accuracy(self, client_label: str, accuracy: float):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} - {client_label}: {accuracy:.2f}% | Global Accuracy: {self.global_accuracy:.2f}%\n"
        with open(self.ACCURACY_LOG, "a") as f:
            f.write(line)
        print_colored(f"📌 [LOG] {line.strip()}", "35")

    # ----- Agregación y envío -----
    def aggregate_models(self):
        print_colored(f"🔵 [AGGREGATION] Using strategy: {self.strategy}", "36")

        if self.strategy == "FEDNOVA":
            num_steps_list = [1 for _ in self.models]
            aggregated_weights = self.apply_fednova(self.models, num_steps_list)
        else:
            keys = list(self.models[0].keys())
            aggregated_weights = {}
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
                    combined = self.apply_fedavg(weights_list_np)
                aggregated_weights[k] = torch.tensor(combined, dtype=torch.float32)

        self.save_aggregated_model(aggregated_weights)
        self.calculate_wait_time()
        self.models.clear()
        self.start_wait_time = None
        self.broadcast(aggregated_weights)

    def broadcast(self, aggregated_weights):
        print_colored("🟣 [BROADCAST] Sending aggregated weights to all clients...", "35")
        serialized = pickle.dumps(aggregated_weights)
        header = struct.pack('!I', len(serialized))
        for client in list(self.clients):
            try:
                client.sendall(header + serialized)
                try:
                    addr = client.getpeername()
                    clabel = f"{addr[0]}:{addr[1]}"
                    self.viz.log_msg_sent(clabel, len(header) + len(serialized))
                except Exception:
                    pass
            except (ConnectionResetError, BrokenPipeError):
                print_colored("🔴 [ERROR] Could not send data to a client.", "31")
        self.clients.clear()

    # ----- Negociación -----
    def negotiate_strategy_with_client(self, client_socket):
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
            proposed_strategy = proposed.upper() if isinstance(proposed, str) else str(proposed).upper()
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

    # ----- Bucle por cliente -----
    def handle_client(self, client_socket):
        client_label = None
        try:
            client_address = client_socket.getpeername()
            client_label = f"{client_address[0]}:{client_address[1]}"
            print_colored(f"🔹 Client connected from {client_address}.", "34")
            self.viz.add_client(client_label)

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

                self.viz.log_msg_received(client_label, data_size)

                # Payload esperado:
                #   (model_weights: dict, accuracy: float, [loss: float opcional])
                payload = pickle.loads(data)
                if isinstance(payload, tuple) and len(payload) >= 2:
                    model_weights = payload[0]
                    accuracy = float(payload[1])
                    loss = float(payload[2]) if len(payload) >= 3 and payload[2] is not None else None
                else:
                    print_colored("🔴 [ERROR] Invalid payload format from client.", "31")
                    continue

                print_colored(f"🔹 Weights received from {client_label}:", "36")
                for key, weight in model_weights.items():
                    try:
                        shape = tuple(weight.shape)
                    except Exception:
                        shape = "<?>"
                    print_colored(f"   • {key}: {shape}", "36")

                with self.lock:
                    # Actualiza métricas globales
                    self.global_accuracy = max(self.global_accuracy, accuracy)
                    if loss is not None:
                        self.global_loss = min(self.global_loss, loss)
                    # Normaliza a tensores torch
                    model_weights_torch = {
                        k: (v if isinstance(v, torch.Tensor) else torch.tensor(v))
                        for k, v in model_weights.items()
                    }
                    self.models.append(model_weights_torch)
                    if client_socket not in self.clients:
                        self.clients.append(client_socket)

                # Logs
                self.log_accuracy(client_label, accuracy)
                self.viz.log_accuracy(client_label, accuracy, self.global_accuracy)
                # Series temporales (accuracy/loss por cliente y global)
                self.viz.log_metrics(
                    client_label=client_label,
                    acc=accuracy,
                    loss=loss,  # puede ser None
                    global_acc=self.global_accuracy,
                    global_loss=(self.global_loss if self.global_loss != float("inf") else None),
                )


                print_colored(f"🟡 [UPDATE] Global Accuracy: {self.global_accuracy:.2f}%", "33")
                if loss is not None and self.global_loss != float('inf'):
                    print_colored(f"🟠 [UPDATE] Global Loss (min): {self.global_loss:.4f}", "33")

                if accuracy >= 93.0:
                    note = f"🔥 Client {client_label} reached 93% accuracy!"
                    print_colored(note, "32")
                    self.viz.log_note(note)

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

    # ----- Main loop -----
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
    strategy = "FedAvg"
    server = Server(strategy=strategy)
    server.aggregation_mode = strategy
    server.run()
