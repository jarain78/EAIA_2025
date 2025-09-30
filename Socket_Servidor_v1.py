# This server establishes a negotiation with clients to partition the dataset
# so that clients can perform local model training.
# Then, the server receives the trained models and aggregates them to obtain a global model.
# The server also stores temporary models in a directory and logs waiting times 
# and accuracy metrics into log files.

import socket
import pickle
import threading
import struct
import torch
import numpy as np
import os
import time
import networkx as nx
import matplotlib.pyplot as plt
#from GSKit import OWA as aggF  # Custom library for OWA

# Server configuration
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5001
STORAGE_DIR = "FDL_Centralized_FrameWork_v1/Models_Temps/model_temp_storage"
LOG_DIR = "FDL_Centralized_FrameWork_v1/Log/"
#WAIT_TIME_LOG = os.path.join(LOG_DIR, "wait_time_log.txt")
#ACCURACY_LOG = os.path.join(LOG_DIR, "accuracy_log.txt")

# Ensure directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def print_colored(message, color_code):
    """Print messages with colors in the terminal."""
    print(f"\033[{color_code}m{message}\033[0m")

def recv_all(sock, size):
    """Receive data from the client in chunks."""
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

# Graph to visualize communication
G = nx.DiGraph()

def update_graph():
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    plt.title("Client-Server Communication Flow")
    plt.show(block=False)

def add_client_to_graph(client_id, client_address):
    G.add_node(f"Client {client_id}\n{client_address}")
    G.add_edge(f"Client {client_id}\n{client_address}", "Server")
    update_graph()


class Server:
    def __init__(self, strategy):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((SERVER_HOST, SERVER_PORT))
        self.server_socket.listen(5)
        print_colored("🔹 Server waiting for connections...", "34")

        self.strategy = strategy  
        self.clients = []  
        self.models = []
        self.lock = threading.Lock()
        self.global_accuracy = 0.0

        self.start_training_time = time.time()  # Training start timestamp
        self.start_wait_time = None  # Start timestamp for waiting for models

        self.agregation_mode = "FedAvg"  # Default aggregation mode
        self.WAIT_TIME_LOG = os.path.join(LOG_DIR, f"wait_time_log_{self.agregation_mode}.txt")
        self.ACCURACY_LOG = os.path.join(LOG_DIR, f"accuracy_log_{self.agregation_mode}.txt")

        self.client_counter = 0

    def update_log_paths(self):
        """Update log file paths based on aggregation mode."""
        self.WAIT_TIME_LOG = os.path.join(LOG_DIR, f"wait_time_log_{self._aggregation_mode}.txt")
        self.ACCURACY_LOG = os.path.join(LOG_DIR, f"accuracy_log_{self._aggregation_mode}.txt")


    @property
    def aggregation_mode(self):
        return self._aggregation_mode

    @aggregation_mode.setter
    def aggregation_mode(self, mode):
        """Dynamically update aggregation mode and log filenames."""
        print(f"🔄 Updating aggregation mode to: {mode}")
        self._aggregation_mode = mode
        self.update_log_paths()  # Update log paths dynamically

    def apply_owa(self, weights_list):
        """Apply OWA to a list of weight matrices."""
        print_colored("🔵 [OWA] Applying OWA aggregation...", "36")
        weights_array = np.array(weights_list)
        X_sorted = -np.sort(-weights_array, axis=0)
        owa_weights = np.array([0.2, 0.5, 0.3])[:, np.newaxis, np.newaxis]
        aggregated_weights = np.sum(X_sorted * owa_weights, axis=0)
        return aggregated_weights  

    def apply_fedavg(self, weights_list):
        """Apply FedAvg (Federated Averaging)."""
        print_colored("🟢 [FedAvg] Applying FedAvg aggregation...", "32")
        return np.mean(weights_list, axis=0)  

    def apply_fedprox(self, weights_list, global_weights, mu=0.01):
        """Apply FedProx by adding a proximity term."""
        print_colored("🔵 [FedProx] Applying FedProx aggregation...", "36")
        global_weights_np = np.array(global_weights)
        prox_term = np.array(weights_list) - global_weights_np
        prox_adjusted = np.mean(weights_list, axis=0) - mu * np.mean(prox_term, axis=0)
        return prox_adjusted
    
    
    def apply_fednova(self, weights_list, num_steps=3):
        """Apply FedNova (Federated Normalized Averaging)."""
        print_colored("🔵 [FedNova] Applying FedNova aggregation...")
        
        num_clients = len(weights_list)
        total_steps = np.sum(num_steps)
        
        # Compute normalization coefficients
        normalized_coeffs = np.array(num_steps) / total_steps
        
        # Initialize the aggregated model
        aggregated_weights = {key: torch.zeros_like(weights_list[0][key]) for key in weights_list[0]}
        
        for i, client_weights in enumerate(weights_list):
            for key in client_weights:
                aggregated_weights[key] += normalized_coeffs[i] * client_weights[key]
        
        return aggregated_weights
    

    def partition_dataset(self):
        """Evenly split the dataset among connected clients."""
        num_clients = len(self.clients)
        if num_clients == 0:
            print_colored("⚠️ No clients connected. Skipping dataset partitioning.", "33")
            return
        
        dataset_size = 100000  # Example: Assume dataset has 100,000 samples
        samples_per_client = dataset_size // num_clients  # Even split

        partition_info = {}
        start_idx = 0

        for client_id in self.clients.keys():
            end_idx = start_idx + samples_per_client
            partition_info[client_id] = (start_idx, end_idx)
            start_idx = end_idx

        self.dataset_partitions = partition_info
        print_colored(f"🔹 Dataset partitioned for {num_clients} clients.", "36")

    def send_dataset_partition(self, client_socket, client_id):
        """Send dataset partition details to the client."""
        if client_id not in self.dataset_partitions:
            print_colored(f"⚠️ No partition found for Client {client_id}.", "31")
            return

        partition_info = self.dataset_partitions[client_id]
        serialized_data = pickle.dumps(partition_info)
        size = struct.pack('!I', len(serialized_data))

        try:
            client_socket.sendall(size + serialized_data)
            print_colored(f"📤 Sent dataset partition {partition_info} to Client {client_id}", "32")
        except Exception as e:
            print_colored(f"🔴 Error sending dataset partition to Client {client_id}: {e}", "31")



    def calculate_wait_time(self):
        """Calculate and display total waiting time for the round."""
        if self.start_wait_time:
            elapsed_time = time.time() - self.start_wait_time
            print_colored(f"⏳ [WAIT TIME] Total time waiting for models: {elapsed_time:.2f} seconds", "33")  # Yellow
            
            # Save wait time to log
            with open(self.WAIT_TIME_LOG, "a") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Wait time: {elapsed_time:.2f} seconds\n")

    def calculate_total_training_time(self):
        """Calculate and display total training time once 93% accuracy is reached."""
        total_time = time.time() - self.start_training_time
        print_colored(f"🚀 [TOTAL TIME] Training took {total_time:.2f} seconds to reach 93% accuracy.", "32")

        # Save total time to log
        with open(self.WAIT_TIME_LOG, "a") as log_file:
            log_file.write(f"\n[COMPLETED] Total training time: {total_time:.2f} seconds\n\n")



    def save_aggregated_model(self, aggregated_weights):
        """Save the aggregated model in temporary storage with a timestamp."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(STORAGE_DIR, f"aggregated_model_{timestamp}.pth")

        torch.save(aggregated_weights, model_path)
        print_colored(f"💾 [SAVED] Aggregated model stored in {model_path}", "36")

    def log_accuracy(self, client_id, accuracy):
        """Log each client's accuracy and the global accuracy to a log file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - Client {client_id}: {accuracy:.2f}% | Global Accuracy: {self.global_accuracy:.2f}%\n"

        with open(self.ACCURACY_LOG, "a") as log_file:
            log_file.write(log_entry)

        print_colored(f"📌 [LOG] {log_entry.strip()}", "35")  # Magenta

    def aggregate_models(self):
        """Apply the selected aggregation strategy and send the weights to clients."""
        print_colored(f"🔵 [AGGREGATION] Using strategy: {self.strategy}", "36")

        aggregated_weights = {}
        global_model_weights = {key: torch.zeros_like(self.models[0][key]) for key in self.models[0]}

        for key in self.models[0]:  
            weights_list = [model[key].cpu().numpy() for model in self.models]  

            if self.strategy == "OWA":
                aggregated_weights[key] = torch.tensor(self.apply_owa(weights_list), dtype=torch.float32)
            elif self.strategy == "FedAvg":
                aggregated_weights[key] = torch.tensor(self.apply_fedavg(weights_list), dtype=torch.float32)
            elif self.strategy == "FedProx":
                aggregated_weights[key] = torch.tensor(self.apply_fedprox(weights_list, global_model_weights[key]), dtype=torch.float32)
            elif self.strategy == "FedNova":
                aggregated_weights[key] = torch.tensor(self.apply_fednova(weights_list), dtype=torch.float32) #, device='cuda')


        self.save_aggregated_model(aggregated_weights)
        # 🔹 Calculate and display waiting time before clearing
        self.calculate_wait_time()

        self.models.clear()
        self.start_wait_time = None  

        self.broadcast(aggregated_weights)

    def broadcast(self, aggregated_weights):
        """Send aggregated weights to all clients."""
        print_colored("🟣 [BROADCAST] Sending aggregated weights to all clients...", "35")
        serialized_weights = pickle.dumps(aggregated_weights)
        size = struct.pack('!I', len(serialized_weights))

        for client in self.clients:
            try:
                client.sendall(size + serialized_weights)
            except (ConnectionResetError, BrokenPipeError):
                print_colored("🔴 [ERROR] Could not send data to a client.", "31")
        
        self.clients.clear()

    def negotiate_strategy_with_client(self, client_socket):
        """Negotiate the preferred aggregation strategy with the client and respond accordingly."""
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
            
            proposed_strategy = pickle.loads(data).upper()
            print_colored(f"🤝 [Negotiation] Client proposes strategy: {proposed_strategy}", "36")

            valid_strategies = ["FEDAVG", "OWA", "FEDPROX", "FEDNOVA"]
            if proposed_strategy in valid_strategies:
                self.strategy = proposed_strategy
                self.aggregation_mode = proposed_strategy
                response = f"✅ Strategy '{proposed_strategy}' accepted."
            else:
                response = f"❌ Strategy '{proposed_strategy}' not valid. Keeping: {self.strategy}"

            response_data = pickle.dumps(response)
            client_socket.sendall(struct.pack('!I', len(response_data)) + response_data)

        except Exception as e:
            print_colored(f"🔴 [Negotiation] Error processing client strategy: {e}", "31")


    def handle_client(self, client_socket):
        """Receive a client's model and update the aggregation process."""
        try:
            client_address = client_socket.getpeername()
            print_colored(f"🔹 Client connected from {client_address}.", "34")
            
            self.negotiate_strategy_with_client(client_socket)

            #print_colored("🔹 Client connected.", "34")
            # Partition dataset when a new client joins
            #self.partition_dataset()
            
            # Send dataset partition to client
            #self.send_dataset_partition(client_socket, client_id)

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
                
                model_weights, accuracy = pickle.loads(data)

                # 🔹 Print shapes of received weights
                print_colored(f"🔹 Weights received from {client_address}: ", "36")
                for key, weight in model_weights.items():
                    print_colored(f"🔹 {key}: {weight.shape}", "36")

                with self.lock:
                    self.global_accuracy = max(self.global_accuracy, accuracy)
                    self.models.append(model_weights)
                    self.clients.append(client_socket)

                # Log client accuracy
                self.log_accuracy(client_socket.getpeername(), accuracy)

                print_colored(f"🟡 [UPDATE] Global Accuracy: {self.global_accuracy:.2f}%", "33")

                if accuracy >= 93.0:
                    self.broadcast_message(f"🔥 A client has reached 93% accuracy! 🔥")

                with self.lock:
                    if len(self.models) == len(self.clients):
                        self.aggregate_models()

        except (ConnectionResetError, BrokenPipeError):
            print_colored("🔴 Client disconnected abruptly.", "31")
        except Exception as e:
            print_colored(f"🔴 Server error: {e}", "31")
        finally:
            client_socket.close()

    def run(self):
        while self.global_accuracy < 93.0:
            try:
                client_socket, _ = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(client_socket,)).start()
            except KeyboardInterrupt:
                print_colored("🚀 Server stopped manually.", "32")
                break
        # 🔹 Calculate and display total training time
        self.calculate_total_training_time()

        print_colored("🚀 [TRAINING COMPLETED] Global accuracy > 93%", "32")

if __name__ == "__main__":
    #strategy = input("Select aggregation strategy (OWA / FedAvg / FedProx / FedNova): ").strip().upper()
    #if strategy not in ["OWA", "FEDAVG", "FEDPROX", "FEDNOVA"]:
    #    print_colored("❌ Invalid strategy. Using FedAvg as default.", "31")
    
    strategy = "FedAvg"

    server = Server(strategy)
    server.aggregation_mode = strategy
    server.run()
