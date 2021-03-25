import math


def get_flat():

    def flat(epoch):
        return 1.

    return flat


def get_warmup_exp_decay(warmup_epochs: int=5, gamma: float=0.95):
    
    def warmup_exp_decay(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1.) / warmup_epochs
        return gamma ** (epoch - warmup_epochs)
    
    return warmup_exp_decay


def get_warmup_linear_decay(warmup_epochs: int=5, total_epochs: int=100):
    
    def warmup_linear_decay(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1.) / warmup_epochs
        return float(total_epochs - epoch) / (total_epochs - warmup_epochs)
    
    return warmup_linear_decay


def get_warmup_sqrt_decay(warmup_epochs: int=5):

    def warmup_sqrt_decay(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1.) / warmup_epochs
        return (epoch - warmup_epochs + 1) ** -0.5

    return warmup_sqrt_decay


def get_warmup_cosine_decay(warmup_epochs: int=5, total_epochs: int=100):

    def warmup_cosine_decay(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1.) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return warmup_cosine_decay