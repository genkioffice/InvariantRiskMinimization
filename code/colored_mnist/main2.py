import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()


print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

# for restart in range(flags.n_restarts):
mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)

# rng_state = np.random.get_state()
# # 追加
# rng_state = list(rng_state)
# random_ints = np.random.randint(0, max(rng_state[1]), len(rng_state[1]))
# del rng_state[1]
# rng_state.insert(1, random_ints)
# np.random.set_state(rng_state)



# np.random.shuffle(mnist.data)
# # np.random.set_state(rng_state)
# np.random.shuffle(mnist.targets)



mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_test = (mnist.data[50000:], mnist.targets[50000:])



def make_environment(images, labels, p_color):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
    
    def torch_xor(a,b):
        return (a-b).abs()

    images = images.reshape((-1,28,28))[:, ::2, ::2]
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels))) # accurecyのmaxはcolorの情報を使わなければ最大が.75である。

    colors = torch_xor(labels, torch_bernoulli(p_color, len(labels)))

    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
        'images': (images.float() / 255.).cuda(),
        'labels': labels[:, None].cuda()
    }
    

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def pretty_print(*values):
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=5, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        lin1 = nn.Linear(2*14*14, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1) # 0, 1のラベルを学習する
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight) #初期化
            nn.init.zeros_(lin.bias) #bias初期化
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    
    def forward(self, input):
        out = input.view(input.shape[0], 2*14*14)
        out = self._main(out)
        return out


for restart in range(flags.n_restarts):
    print("Restart", restart)
    np.random.seed(restart + 40)
    np.random.shuffle(mnist_train[0].numpy())
    np.random.shuffle(mnist_train[1].numpy())


    mlp = MLP().cuda()
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2), # 0.8の確率でlabel->colorを
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1), # 0.9の確率
        make_environment(mnist_test[0], mnist_test[1], 0.9) # 0.1の確率. テストにおいては擬似相関が小さくなる
    ]

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc' 'loss')
    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])
        
        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
        
        weight_norm = torch.tensor(0.).cuda()
        # これは何のためにある？ => たぶんERMとpenaltyの比率
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)
        
        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty

        if penalty_weight > 1.0:
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_acc = envs[2]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
                loss.detach().cpu().numpy()
        )