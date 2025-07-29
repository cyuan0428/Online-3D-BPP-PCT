from fastapi import FastAPI
import torch
from model import DRL_GAT
from tools import load_policy

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

class Args:
    setting = 2
    lnes = 'EMS'
    internal_node_holder = 80
    leaf_node_holder = 50
    shuffle = True
    continuous = True
    no_cuda = False
    device = 0
    seed = 4
    use_acktr = True
    num_processes = 64
    num_steps = 5
    learning_rate = 1e-6
    actor_loss_coef = 1.0
    critic_loss_coef = 1.0
    max_grad_norm = 0.5
    embedding_size = 64
    hidden_size = 128
    gat_layer_num = 1
    gamma = 1.0
    model_save_interval = 200
    model_update_interval = int(2e4)
    model_save_path = './logs/experiment'
    print_log_interval = 10
    evaluate = False
    evaluation_episodes = 100
    load_model = False
    model_path = None
    load_dataset = False
    dataset_path = None
    sample_from_distribution = False
    sample_left_bound = None
    sample_right_bound = None
    id = 'PctContinuous-v0' if continuous else 'PctDiscrete-v0'
    container_size = [10, 10, 10]
    item_size_set = [[2, 3, 4], [1, 1, 1]]
    internal_node_length = 6
    normFactor = 1.0 / max(container_size)

@app.on_event("startup")
def load_model():
    global model
    args = Args()
    model = DRL_GAT(args)
    model = load_policy("PCT-cont_test1-2025.07.26-16-47-19_2025.07.27-06-39-42.pt", model)
    model.to(device)
    model.eval()

@app.post("/predict")
def predict():
    return {"message": f"模型已經載入到 {device}，可以推論！"}
