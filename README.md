# EAIA_2025

Perfect 👌 — here’s a **README in English** that explains the **server**, **client**, and **visualizer (`viz_rerun.py`)**. I’ll keep it clear and structured so you can drop it into your repo.

---

# Federated Learning Server with Rerun Visualization

This project implements a **centralized Federated Learning (FL) server** in Python with support for multiple aggregation strategies and **real-time visualization** of client activity, messages, and training metrics using [Rerun](https://rerun.io).

---

## 📂 Project Components

### 1. `server_rerun_fdl.py` (Server)

* A TCP server that coordinates multiple clients in a Federated Learning setup.
* Responsibilities:

  * Accept connections from clients.
  * Negotiate aggregation strategy (`FedAvg`, `OWA`, `FedProx`, `FedNova`).
  * Receive local model weights + metrics (`accuracy`, optional `loss`).
  * Aggregate models according to the chosen strategy.
  * Save aggregated models to disk (`.pth`).
  * Broadcast the aggregated model back to all connected clients.
  * Log wait times and accuracy in text files.

**Aggregation modes supported:**

* **FedAvg** – simple average of weights.
* **OWA** – Ordered Weighted Averaging.
* **FedProx** – FedAvg with a proximity term.
* **FedNova** – normalized averaging based on local steps.

---

### 2. `viz_rerun.py` (Visualizer)

* Provides a wrapper around [Rerun](https://rerun.io) to visualize:

  * **Graph view** of server and connected clients (nodes + edges).
  * **Event logs** (connections, disconnections, messages, notes).
  * **Training metrics**:

    * Client-level accuracy/loss (`metrics/accuracy/<client>`, `metrics/loss/<client>`).
    * Global accuracy/loss (`metrics/accuracy/global`, `metrics/loss/global`).
* Each client event is timestamped and appears in the Rerun viewer in real time.
* Useful for debugging and teaching how Federated Learning works.

---

### 3. Client (example outline)

Clients are responsible for:

1. Connecting to the server via TCP.
2. Negotiating the aggregation strategy (optional).
3. Receiving dataset partition information (optional).
4. Training a local model.
5. Sending `(model_weights, accuracy, loss)` back to the server.
6. Receiving aggregated global weights from the server and continuing training.

A **minimal client workflow**:

```python
import socket, pickle, struct

HOST, PORT = "127.0.0.1", 5001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

# Send preferred strategy
strategy = "FedAvg"
msg = pickle.dumps(strategy)
sock.sendall(struct.pack('!I', len(msg)) + msg)

# Dummy payload: model weights dict + accuracy + loss
dummy_weights = {"layer1": ...}  # torch tensors or numpy arrays
payload = (dummy_weights, 85.3, 0.42)
data = pickle.dumps(payload)
sock.sendall(struct.pack('!I', len(data)) + data)

# Receive aggregated model
header = sock.recv(4)
size = struct.unpack('!I', header)[0]
global_weights = pickle.loads(sock.recv(size))
```

---

## 🚀 How to Run

1. **Install dependencies**

   ```bash
   pip install torch numpy networkx matplotlib rerun-sdk
   ```

2. **Run the server**

   ```bash
   python server_rerun_fdl.py
   ```

   * The server starts listening on `127.0.0.1:5001`.
   * A Rerun viewer window opens automatically.

3. **Run one or more clients** (custom implementations).

   * Each client connects, trains locally, and sends weights + metrics.
   * The server aggregates and updates the Rerun viewer.

---

## 📊 Visualization Features

* **Graph View**

  * Blue node: Server.
  * Light-blue nodes: Connected clients.
  * Directed edges: Communication flows (client → server, server → client).

* **Events Log**

  * Shows connections/disconnections, message size, aggregation rounds.

* **Metrics Dashboard**

  * Time-series plots for accuracy and loss (per client and global).
  * Select `metrics/accuracy/*` and `metrics/loss/*` channels in Rerun to see trends.

---

## 📁 Logs & Models

* Aggregated models are saved in:

  ```
  FDL_Centralized_FrameWork_v1/Models_Temps/model_temp_storage/
  ```

* Logs are stored in:

  ```
  FDL_Centralized_FrameWork_v1/Log/
  ├── wait_time_log_<strategy>.txt
  └── accuracy_log_<strategy>.txt
  ```

---

## ✅ Example Workflow

1. Start the server:

   ```bash
   python server_rerun_fdl.py
   ```

   > Rerun viewer opens: `Server visualization started ✅`

2. Start two or more clients. Each client trains locally and sends updates:

   ```
   👤 Client connected: 127.0.0.1:55231
   📥 FROM 127.0.0.1:55231 → Server | 128456 bytes
   📌 Acc 127.0.0.1:55231: 85.3% | Global: 85.3%
   ```

3. Watch Rerun:

   * Clients appear as nodes.
   * Events populate in the log.
   * Accuracy and loss curves update live.

---

