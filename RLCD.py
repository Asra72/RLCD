import torch
import torch.nn as nn
import torch.optim as optim
import random, time, math
from itertools import product
from docplex.mp.model import Model

def get_cvrp_data():
    num_customers = 5
    num_vehicles = 2
    vehicle_capacity = 15
    depot_index = 0
    points = [(0.0,0.0),(0.3,0.2),(0.1,0.5),(0.6,0.4),(0.7,0.1),(0.2,0.8)]
    demands = [0,4,6,5,7,3]
    return num_customers, num_vehicles, vehicle_capacity, depot_index, points, demands

def instance_features():
    num_customers, num_vehicles, vehicle_capacity, depot_index, points, demands = get_cvrp_data()
    customers = list(range(1, num_customers+1))
    max_dist = math.sqrt(2.0)
    d_cc = []
    for i in customers:
        for j in customers:
            if i < j:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d_cc.append(math.hypot(dx, dy)/max_dist)
    d_cd = []
    for i in customers:
        dx = points[i][0] - points[depot_index][0]
        dy = points[i][1] - points[depot_index][1]
        d_cd.append(math.hypot(dx, dy)/max_dist)
    total_demand = sum(demands[i] for i in customers)
    utilization = total_demand / (num_vehicles * vehicle_capacity)
    mean_cc = sum(d_cc)/len(d_cc)
    mean_cd = sum(d_cd)/len(d_cd)
    mean_cc_sq = sum(x*x for x in d_cc)/len(d_cc)
    std_cc = math.sqrt(max(0.0, mean_cc_sq - mean_cc*mean_cc))
    f1 = float(min(1.5, utilization))
    f2 = float(mean_cc)
    f3 = float(mean_cd)
    f4 = float(std_cc)
    return torch.tensor([f1, f2, f3, f4], dtype=torch.float32)

EMPHASIS_OPTIONS = [0, 1, 2]
HEURISTIC_FREQ_OPTIONS = [-1, 0, 10, 20, 50]
THREADS_OPTIONS = [1, 2, 4, 8]
GAP_OPTIONS = [0.02, 0.05, 0.10, 0.15, 0.20]

ACTIONS = list(product(EMPHASIS_OPTIONS, HEURISTIC_FREQ_OPTIONS, THREADS_OPTIONS, GAP_OPTIONS))

class ActorNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.head_actions = nn.Linear(hidden_dim, len(ACTIONS))
    def forward(self, x):
        z = self.feature_extractor(x)
        return self.head_actions(z)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.value_network(x).squeeze(-1)

def sample_configuration(logits_actions):
    dist = torch.distributions.Categorical(logits=logits_actions)
    idx = dist.sample()
    e, h, t, g = ACTIONS[int(idx)]
    config = {"emphasis": int(e), "heuristic_freq": int(h), "threads": int(t), "mip_gap": float(g)}
    logprob = dist.log_prob(idx)
    entropy_value = dist.entropy()
    picks = int(idx)
    return config, logprob, entropy_value, picks

def evaluate_configuration_with_docplex(config, time_limit_sec=30.0):
    num_customers, num_vehicles, vehicle_capacity, depot_index, points, demands = get_cvrp_data()
    num_nodes = num_customers + 1
    costs_matrix = [[0.0]*num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                costs_matrix[i][j] = math.hypot(dx, dy)
    model = Model(name="cvrp_small", log_output=False)
    x_variables = {(i,j): model.binary_var(name=f"x_{i}_{j}") for i in range(num_nodes) for j in range(num_nodes) if i != j}
    load_variables = {i: model.continuous_var(lb=0, ub=vehicle_capacity, name=f"u_{i}") for i in range(num_nodes)}
    model.add_constraint(model.sum(x_variables[depot_index, j] for j in range(1, num_nodes)) == num_vehicles)
    model.add_constraint(model.sum(x_variables[i, depot_index] for i in range(1, num_nodes)) == num_vehicles)
    for k in range(1, num_nodes):
        model.add_constraint(model.sum(x_variables[i, k] for i in range(num_nodes) if i != k) == 1)
        model.add_constraint(model.sum(x_variables[k, j] for j in range(num_nodes) if j != k) == 1)
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                model.add_constraint(load_variables[i] - load_variables[j] + vehicle_capacity * x_variables[i, j] <= vehicle_capacity - demands[j])
    model.add_constraint(load_variables[depot_index] == 0)
    for i in range(1, num_nodes):
        model.add_constraint(load_variables[i] >= demands[i])
        model.add_constraint(load_variables[i] <= vehicle_capacity)
    model.minimize(model.sum(costs_matrix[i][j] * x_variables[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j))
    model.context.cplex_parameters.emphasis.mip = int(config["emphasis"])
    model.context.cplex_parameters.mip.strategy.heuristicfreq = int(config["heuristic_freq"])
    model.parameters.mip.tolerances.mipgap = float(config["mip_gap"])
    model.context.cplex_parameters.threads = int(config["threads"])
    model.parameters.timelimit = float(time_limit_sec)
    t0 = time.time()
    model.solve()
    t1 = time.time() - t0
    if not model.solution:
        return -(1_000_000 + 10.0 * t1)
    objective_value = model.objective_value
    reward_value = -(objective_value + 0.1 * t1)
    return reward_value

def train_ppo(num_iterations=40, batch_size=12, num_epochs=3, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, learning_rate=3e-3):
    torch.manual_seed(1); random.seed(1)
    features = instance_features()
    actor_network = ActorNetwork(input_dim=features.numel())
    critic_network = CriticNetwork(input_dim=features.numel())
    optimizer_actor = optim.SGD(actor_network.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_critic = optim.SGD(critic_network.parameters(), lr=learning_rate, momentum=0.9)
    best_reward_overall = -1e18
    best_config_overall = None
    for iteration in range(1, num_iterations + 1):
        with torch.no_grad():
            value_estimate = critic_network(features).item()
        old_logprobs_list, rewards_list, values_list, picks_list, configs_list = [], [], [], [], []
        for _ in range(batch_size):
            logits_actions = actor_network(features)
            config, logprob, entropy_value, picks = sample_configuration(logits_actions)
            reward_value = evaluate_configuration_with_docplex(config)
            old_logprobs_list.append(logprob.detach())
            rewards_list.append(reward_value)
            values_list.append(value_estimate)
            picks_list.append(picks)
            configs_list.append(config)
        old_logprobs_tensor = torch.stack(old_logprobs_list)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
        values_tensor = torch.tensor(values_list, dtype=torch.float32)
        advantages = rewards_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / advantages.std()
        returns = rewards_tensor
        for _ in range(num_epochs):
            logits_actions_now = actor_network(features)
            dist = torch.distributions.Categorical(logits=logits_actions_now)
            new_logprobs = torch.stack([dist.log_prob(torch.tensor(i)) for i in picks_list])
            entropy_mean = dist.entropy()
            ratio = torch.exp(new_logprobs - old_logprobs_tensor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            actor_loss = policy_loss - entropy_coef * entropy_mean
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
            value_prediction = critic_network(features).repeat(batch_size)
            critic_loss = (returns - value_prediction).pow(2).mean() * value_coef
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
        best_index = int(torch.argmax(rewards_tensor).item())
        if rewards_list[best_index] > best_reward_overall:
            best_reward_overall = rewards_list[best_index]
            best_config_overall = configs_list[best_index]
        if iteration % 5 == 0:
            print(f"iter {iteration:03d}  best {best_reward_overall:.2f}  mean {rewards_tensor.mean():.2f}")
    print("done.")
    print("best_config:", best_config_overall)
    return best_config_overall

if __name__ == "__main__":
    train_ppo(num_iterations=40, batch_size=12, num_epochs=3, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, learning_rate=3e-3)

