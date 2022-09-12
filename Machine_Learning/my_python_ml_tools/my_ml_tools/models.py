class WeightDecayGradientHook:
    def __init__(self, weight, regularization_lambda):
        self.decay = regularization_lambda * weight.detach().clone()

    def __call__(self, grad):
        return grad.clone().detach() + self.decay


def l2_regularization(tensor, regularization_lambda):
    """Adds weight decay gradient hook to tensor corresponding to L2 loss"""
    if tensor.requires_grad:
        hook = WeightDecayGradientHook(tensor, regularization_lambda)
        tensor.register_hook(hook)
    return tensor
