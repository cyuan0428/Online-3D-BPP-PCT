from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from model import DRL_GAT
from tools import load_policy

app = FastAPI()

# 自動偵測運算裝置
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 模型初始化參數 ====
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

# ==== 輸入格式 ====
class Item(BaseModel):
    color: str
    size: List[float]  # [長, 寬, 高]

class PredictRequest(BaseModel):
    items: List[Item]

# ==== 啟動時載入模型（含try/except印traceback） ====
@app.on_event("startup")
def load_model():
    global model
    try:
        args = Args()
        model = DRL_GAT(args)
        model = load_policy("PCT-cont_test1-2025.07.26-16-47-19_2025.07.27-06-39-42.pt", model)
        model.to(device)
        model.eval()
        print(f"模型載入完成，設備：{device}")
    except Exception as e:
        import traceback
        print("=== 載入模型發生錯誤 ===")
        traceback.print_exc()
        raise e

# ==== API 端點 ====
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # 轉成模型可接受的 tensor 格式
        input_sizes = [item.size for item in req.items]
        input_tensor = torch.tensor([input_sizes], dtype=torch.float32).to(device)

        # 模型推論
        with torch.no_grad():
            output = model(input_tensor)

        # 處理輸出結果 (根據你的模型調整)
        output_result = [x.cpu().numpy().tolist() for x in output]

        return {
            "received": [item.dict() for item in req.items],
            "model_output": output_result,
            "message": f"推論成功，模型運行於 {device}！"
        }
    except Exception as e:
        import traceback
        print("=== 推論過程發生錯誤 ===")
        traceback.print_exc()
        return {
            "error": f"推論失敗: {str(e)}"
        }
