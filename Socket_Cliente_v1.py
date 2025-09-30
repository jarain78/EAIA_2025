import socket
import pickle
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from torchvision import datasets
import sys

# Configuración del cliente
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5001
MODEL_PATH = "Models/"

# Función para imprimir con colores en la terminal
def print_colored(message, color_code):
    print(f"\033[{color_code}m{message}\033[0m")

def recv_all(sock, size):
    """Recibe datos del servidor en fragmentos."""
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

# 🔹 Permitir seleccionar el dataset
def select_dataset():
    print("Seleccione el dataset:")
    print("1. MNIST (28x28 imágenes en escala de grises, 10 clases)")
    print("2. CIFAR-100 (32x32 imágenes a color, 100 clases)")
    print("3. CIFAR-10 (32x32 imágenes a color, 10 clases)")

    choice = input("Ingrese el número correspondiente (1, 2 o 3): ").strip()
    
    if choice == "1":
        return "MNIST"
    elif choice == "2":
        return "CIFAR-100"
    elif choice == "3":
        return "CIFAR-10"
    else:
        print_colored("Selección inválida. Se usará MNIST por defecto.", "31")  # Rojo
        return "MNIST"

# 🔹 Permitir seleccionar el mecanismo de agregación
def select_strategy():
    print("\nSeleccione el mecanismo de agregación:")
    print("1. FedAvg (Promedio Federado)")
    print("2. OWA (Ordered Weighted Averaging)")
    print("3. FedProx (Federated Proximal)")
    print("4. FedNova (Normalized Averaging)")

    choice = input("Ingrese el número correspondiente (1, 2, 3 o 4): ").strip()

    if choice == "1":
        return "FedAvg"
    elif choice == "2":
        return "OWA"
    elif choice == "3":
        return "FedProx"
    elif choice == "4":
        return "FedNova"
    else:
        print_colored("Selección inválida. Se usará FedAvg por defecto.", "31")  # Rojo
        return "FedAvg"


# 🔹 Definir la arquitectura de la red neuronal dinámicamente
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Aumentamos la cantidad de neuronas para más robustez
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Client:
    def __init__(self, client_id):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((SERVER_HOST, SERVER_PORT))

            # Negociar estrategia de agregación
            # OWA / FedAvg / FedProx / FedNova
            #self.negotiate_strategy(self.client_socket, strategy_name="FedAvg")

            # Mostrar menú de selección de estrategia
            self.strategy_name = select_strategy()

            # Negociar estrategia de agregación
            self.negotiate_strategy(self.client_socket, strategy_name=self.strategy_name)

        except ConnectionRefusedError:
            print_colored("No se pudo conectar al servidor. Asegúrate de que el servidor está en ejecución.", "31")  # Rojo
            return
        
        self.client_id = client_id
        self.dataset_name = select_dataset()  # Preguntar al usuario qué dataset usar
        self.train_loader, input_size, num_classes = self.get_data()  # Obtener los datos, input_size y número de clases
        
        # 🔹 Configurar el modelo de acuerdo al dataset seleccionado
        self.model = NeuralNet(input_size, num_classes)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

        #self.partition_range = self.receive_dataset_partition()  # Get assigned dataset


    # 🔹 Negociación de estrategia de agregación con el servidor
    def negotiate_strategy(self, client_socket, strategy_name="FedAvg"):
        try:
            print(f"🤝 Proponiendo estrategia '{strategy_name}' al servidor...")
            serialized = pickle.dumps(strategy_name)
            client_socket.sendall(struct.pack('!I', len(serialized)) + serialized)

            # Esperar confirmación del servidor
            header = recv_all(client_socket, 4)
            if header:
                size = struct.unpack('!I', header)[0]
                response = recv_all(client_socket, size)
                response_msg = pickle.loads(response)
                print(f"📨 Respuesta del servidor: {response_msg}")
            else:
                print("⚠️ No se recibió respuesta del servidor.")
        except Exception as e:
            print(f"🔴 Error durante la negociación: {e}")


    def get_data(self):
        """Carga el dataset seleccionado y devuelve DataLoader, input_size y num_classes."""
        if self.dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))  # Convertir en vector plano
            ])
            dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            input_size = 28 * 28  # Dimensiones de la imagen aplanada
            num_classes = 10

        elif self.dataset_name == "CIFAR-100":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))  # Convertir en vector plano
            ])
            dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
            input_size = 32 * 32 * 3  # 32x32 imágenes con 3 canales (RGB)
            num_classes = 100

        elif self.dataset_name == "CIFAR-10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))  # Convertir en vector plano
            ])
            dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
            input_size = 32 * 32 * 3  # 32x32 imágenes con 3 canales (RGB)
            num_classes = 10

        return DataLoader(dataset, batch_size=64, shuffle=True), input_size, num_classes
    
    def receive_dataset_partition(self):
            """Receives dataset partition from the server."""
            header = recv_all(self.client_socket, 4)
            if not header:
                print_colored("⚠️ No dataset partition received.", "31")
                return None
            
            data_size = struct.unpack('!I', header)[0]
            data = recv_all(self.client_socket, data_size)
            if not data:
                print_colored("⚠️ Failed to receive dataset partition.", "31")
                return None

            partition_range = pickle.loads(data)
            print_colored(f"📥 Received dataset partition: {partition_range}", "32")
            return partition_range

    def evaluate(self):
        """Evalúa la precisión del modelo en el dataset de entrenamiento (por simplicidad)."""
        correct = 0
        total = 0
        for images, labels in self.train_loader:
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return (correct / total) * 100
    
    def train(self):
        """Entrena el modelo localmente hasta alcanzar el accuracy requerido."""
        if not self.client_socket:
            return
        
        print_colored(f"🟢 [DEBUG] Cliente {self.client_id} ({self.dataset_name}): Iniciando entrenamiento...", "32")  # Verde

        while True:
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
            
            accuracy = self.evaluate()
            print_colored(f"🔹 Cliente {self.client_id} ({self.dataset_name}): Accuracy = {accuracy:.2f}%", "34")  # Azul
            
            print_colored(f"🟢 [DEBUG] Cliente {self.client_id} ({self.dataset_name}): Entrenando con Nuevos Pesos...", "32")  # Verde

            if accuracy >= 90.0:
                print_colored(f"💾 Cliente {self.client_id}: Entrenamiento completado con accuracy {accuracy:.2f}%", "36")  # Cian
                
                # Guardar el modelo entrenado localmente
                model_filename = MODEL_PATH + f"Cliente_{self.client_id}_{self.dataset_name}_modelo_entrenado.pth"
                torch.save(self.model.state_dict(), model_filename)
                print_colored(f"[GUARDADO] Modelo entrenado guardado como '{model_filename}'", "36")  # Cian

                # Enviar mensaje al servidor
                message = f"Cliente {self.client_id} ({self.dataset_name}) ha alcanzado el accuracy requerido del 90%."
                try:
                    self.client_socket.sendall(message.encode())  # Enviar mensaje de texto al servidor
                    print_colored("🟣 [DEBUG] Mensaje enviado al servidor.", "35")  # Magenta
                except (ConnectionResetError, BrokenPipeError):
                    print_colored(f"🔴 Cliente {self.client_id}: Error al enviar el mensaje al servidor.", "31")  # Rojo
                return
                 
            self.send_model(accuracy)
    
    def send_model(self, accuracy):
        """Envía los pesos del modelo al servidor."""
        if not self.client_socket:
            return
        
        print_colored("🟡 [DEBUG] Enviando modelo al servidor...", "33")  # Amarillo

        model_weights = {key: value.clone().detach() for key, value in self.model.state_dict().items()}
        serialized_data = pickle.dumps((model_weights, accuracy))
        size = struct.pack('!I', len(serialized_data))
        
        time.sleep(1)  # Espera 1 segundo antes de enviar los datos

        try:
            self.client_socket.sendall(size + serialized_data)
        except (ConnectionResetError, BrokenPipeError):
            print_colored("🔴 Error de conexión. Reintentando...", "31")  # Rojo
            return

    def close(self):
        """Cierra la conexión del cliente."""
        if self.client_socket:
            self.client_socket.close()

if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    client = Client(client_id)
    client.train()
    client.close()
