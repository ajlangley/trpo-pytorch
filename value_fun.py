class ValueFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, state):
        return self.model(state)

    def to(self, device):
        self.model.to(device)

    def apply_update(self, step):
        n = 0

        for param in self.parameters():
            update = step[n: n + param.numel()].view(param.size())
            param.data += update
            n += param.numel()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
