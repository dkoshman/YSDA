class RegularizationGradientHook:
    def __init__(self, tensor, l2_coefficient, l1_coefficient):
        tensor = tensor.clone().detach()
        self.decay = 2 * l2_coefficient * tensor + l1_coefficient * tensor.sign()

    def __call__(self, grad):
        return grad.clone().detach() + self.decay


def register_regularization_hook(tensor, l2_coefficient, l1_coefficient=0):
    """Adds weight decay gradient hook to tensor in place"""
    if tensor.requires_grad:
        hook = RegularizationGradientHook(tensor, l2_coefficient, l1_coefficient)
        tensor.register_hook(hook)
