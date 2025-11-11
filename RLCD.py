import torch
import torch.nn as nn
import torch.optim as optim
import random

def make_state():
    return torch.tensor([random.random(), random.random()], dtype=torch.float32)

def next_state(s, a):
    x, y = s.tolist()
    if a == 0:
        r = 1.0 - x + random.uniform(-0.3, 0.3)
    elif a == 1:
        r = 1.4 - abs(x - 0.5) + random.uniform(-0.2, 0.2)
    else:
        r = 1.8 - (x + y) + random.uniform(-0.3, 0.3)
    if r < 0: r = 0
    done = random.random() < 0.07
    ns = torch.tensor([random.random(), random.random()], dtype=torch.float32)
    return ns, r, done

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.Linear(2, 32)
        self.p = nn.Linear(32, 3)
        self.v = nn.Linear(32, 1)
    def forward(self, x):
        z = torch.relu(self.h(x))
        return self.p(z), self.v(z).squeeze(-1)

def one_episode(model, gamma=0.97):
    s = make_state()
    s_list, a_list, r_list, v_list, logp_list = [], [], [], [], []
    done = False
    total = 0.0
    while not done:
        logits, val = model(s)
        exp_logits = torch.exp(logits)
        probs = exp_logits / torch.sum(exp_logits)
        a = torch.multinomial(probs, 1).item()
        logp = torch.log(probs[a] + 1e-8)
        ns, r, done = next_state(s, a)
        s_list.append(s)
        a_list.append(a)
        r_list.append(r)
        v_list.append(val)
        logp_list.append(logp)
        total += r
        s = ns
    G = 0.0
    returns = []
    for r in reversed(r_list):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.stack(s_list), torch.tensor(a_list), torch.tensor(returns), torch.stack(v_list), torch.stack(logp_list), total

def train():
    torch.manual_seed(3)
    random.seed(3)
    net = TinyNet()
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    gamma = 0.97
    ep_total = 300
    print("training")
    for ep in range(1, ep_total + 1):
        S, A, R, V, L, tr = one_episode(net, gamma)
        pg_loss = -(L * (R - V.detach())).mean()
        v_loss = (R - V).pow(2).mean()
        loss = pg_loss + 0.5 * v_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 25 == 0:
            s = make_state()
            done = False
            ev_r = 0
            while not done:
                logits, _ = net(s)
                ex = torch.exp(logits)
                pr = ex / torch.sum(ex)
                a = int(torch.argmax(pr).item())
                s, r, done = next_state(s, a)
                ev_r += r
            print("ep", ep, "train", round(tr, 2), "eval", round(ev_r, 2))
    print("done")

if __name__ == "__main__":
    train()
