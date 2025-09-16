import random
import numpy as np
import json
from copy import deepcopy

# 超参数存储模块
class HyperparameterStore:
    def __init__(self, initial_params=None, save_path="hyperparameters.json"):
        self.save_path = save_path
        self.params = initial_params if initial_params else {}

    def load(self):
        """加载超参数集合"""
        try:
            with open(self.save_path, "r") as file:
                self.params = json.load(file)
        except FileNotFoundError:
            print("No existing hyperparameters found. Using initial parameters.")

    def save(self):
        """保存超参数集合"""
        with open(self.save_path, "w") as file:
            json.dump(self.params, file, indent=4)

    def update(self, new_params):
        """更新超参数集合"""
        self.params.update(new_params)
        self.save()

# 模型训练与评估模块
class ModelTrainer:
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    def train_and_evaluate(self, hyperparams):
        """
        模拟模型训练并计算验证集损失。
        用户需要替换为具体的模型训练和评估代码。
        """
        # 示例：生成一个随机的验证损失
        loss = sum(hyperparams.values()) + random.uniform(-0.5, 0.5)  # 模拟损失
        return loss

# 进化优化模块
class EvolutionaryOptimizer:
    def __init__(self, initial_params, trainer, mutation_rate=0.1, exploration_decay=0.95, min_exploration=0.01):
        self.params = initial_params
        self.trainer = trainer
        self.mutation_rate = mutation_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.best_params = deepcopy(initial_params)
        self.best_loss = float("inf")

    def mutate(self, param_key, param_value):
        """根据参数类型对超参数进行变异"""
        if param_key == "learning_rate":
            # 学习率缩放在 0.1 倍到 10 倍之间
            factor = random.uniform(0.1, 10)
            return max(1e-6, param_value * factor)
        elif param_key in ["d_model", "d_ff"]:
            # 模型维度参数整数随机扰动
            return max(1, param_value + random.randint(-10, 10))
        else:
            # 默认随机加减
            return param_value + random.uniform(-0.1, 0.1)

    def evolve(self, generations=50):
        """运行进化优化算法"""
        current_params = deepcopy(self.params)
        exploration_rate = self.mutation_rate

        for generation in range(generations):
            # Step 1: 训练和评估
            loss = self.trainer.train_and_evaluate(current_params)

            # Step 2: 自然选择
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = deepcopy(current_params)
                print(f"Generation {generation}: New best loss = {loss}")
            else:
                # Step 3: 遗传变异
                mutated_params = {}
                for key, value in current_params.items():
                    mutated_params[key] = self.mutate(key, value)
                current_params = mutated_params

            # Step 4: 动态调整探索强度
            exploration_rate = max(self.min_exploration, exploration_rate * self.exploration_decay)

        print("Best parameters found:", self.best_params)
        print("Best loss:", self.best_loss)
        return self.best_params, self.best_loss

# 主程序
if __name__ == "__main__":
    # 初始化超参数
    initial_hyperparameters = {
        "learning_rate": 0.01,
        "d_model": 512,
        "d_ff": 2048,
    }

    # 加载和保存超参数
    store = HyperparameterStore(initial_hyperparameters)
    store.load()

    # 假设的训练和验证数据（用户需要替换为实际数据）
    train_data = None
    val_data = None

    # 模型训练器
    trainer = ModelTrainer(None, train_data, val_data)

    # 进化优化器
    optimizer = EvolutionaryOptimizer(store.params, trainer)
    best_params, best_loss = optimizer.evolve(generations=20)

    # 保存最终的超参数
    store.update(best_params)
