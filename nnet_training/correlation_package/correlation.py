import torch
torch.ops.load_library("correlation_package/build/lib.linux-x86_64-3.6/correlation_pkg.cpython-36m-x86_64-linux-gnu.so")

class CorrelationTorch(torch.nn.Module):
    def __init__(self, max_displacement=4, *args, **kwargs):
        super(CorrelationTorch, self).__init__()
        self.max_displacement = max_displacement
        self.output_dim = 2 * self.max_displacement + 1
        self.pad_size = self.max_displacement

    def forward(self, x1, x2):
        B, C, H, W = x1.size()

        x2 = torch.nn.functional.pad(x2, [self.pad_size] * 4)
        cv = []
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                cost = x1 * x2[:, :, i:(i + H), j:(j + W)]
                cost = torch.mean(cost, 1, keepdim=True)
                cv.append(cost)
        return torch.cat(cv, 1)

class CorrelationFunction(torch.autograd.Function):
    """
    Typical Parameters: pad_size=3, kernel_size=3, max_displacement=20,
    stride1=1, stride2=2, corr_multiply=1
    """

    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3,
                max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        ctx.save_for_backward(input1, input2)

        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        output = torch.ops.cerberus.correlation(
            input1, input2, pad_size, kernel_size,
            max_displacement, stride1, stride2, corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input1, input2 = ctx.saved_tensors

        grad_input1, grad_input2 = torch.ops.cerberus.correlation_backward(
            input1, input2, grad_outputs, ctx.pad_size, ctx.kernel_size,
            ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(torch.nn.Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0,
                 stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()

        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        # if self.train():
        #     return CorrelationFunction.apply(
        #         input1, input2, self.pad_size, self.kernel_size,
        #         self.max_displacement, self.stride1, self.stride2, self.corr_multiply)

        return torch.ops.cerberus.correlation(
            input1, input2, self.pad_size, self.kernel_size,
            self.max_displacement, self.stride1, self.stride2, self.corr_multiply)

if __name__ == '__main__':
    import time
    import random

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    CORR = Correlation(pad_size=4, kernel_size=1, max_displacement=10,
                       stride1=1, stride2=1, corr_multiply=1).to(DEVICE)

    T_SUM = 0.0

    for i in range(10):
        C = random.choice([128, 256])
        H = random.choice([128, 256])
        W = random.choice([64, 128])
        x1 = torch.randn(16, C, H, W, requires_grad=True).to(DEVICE)
        x2 = torch.randn(16, C, H, W, requires_grad=True).to(DEVICE)

        end = time.time()
        y = CORR(x1, x2)
        t_f = time.time() - end

        end = time.time()
        y.sum().backward()
        t_b = time.time() - end

        print(f'Forward: {t_f * 1000:.3f}ms, Backward: {t_b * 1000:.3f}ms')

        if i < 3:
            continue
        T_SUM += t_b + t_f

    print(f'Sum: {T_SUM:.3f}s')
