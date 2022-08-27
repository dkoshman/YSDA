class WeightDecayGradientHook:
    def __init__(self, weight, regularization_lambda):
        self.decay = regularization_lambda * weight.detach().clone()

    def __call__(self, grad):
        try:
            grad.data += self.decay
        except RuntimeError:
            grad.data = grad.data.clone() + self.decay
        return grad


def weight_decay(tensor, regularization_lambda):
    """Adds weight decay gradient hook to tensor corresponding to L2 loss"""
    if not tensor.requires_grad:
        return tensor

    hook = WeightDecayGradientHook(tensor, regularization_lambda)
    tensor.register_hook(hook)
    return tensor
