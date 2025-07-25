import torch

# 設定箱子尺寸
container_size = [80, 90, 50]

# 你的 4 個物品
items = [
    [20, 44, 7],   # item1
    [20, 88, 7],   # item2
    [40, 44, 7],   # item3
    [20, 22, 7],   # item4
]

# 組合成一個資料集（只有一筆裝箱任務）
my_dataset = [
    [container_size, items]
]

# 存成 .pt 檔案
torch.save(my_dataset, 'my_4items_dataset.pt')

print("已成功產生 my_4items_dataset.pt！")
