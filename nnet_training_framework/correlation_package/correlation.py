import torch
import correlation_arf, correlation_pwc

torch.backends.cudnn.deterministic = True
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = False

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
    Typical Parameters: pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1
    """

    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        ctx.save_for_backward(input1, input2)
        
        with torch.cuda.device_of(input1):
            ctx.pad_size = pad_size
            ctx.kernel_size = kernel_size
            ctx.max_displacement = max_displacement
            ctx.stride1 = stride1
            ctx.stride2 = stride2
            ctx.corr_multiply = corr_multiply

            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_pwc.forward(input1, input2, rbot1, rbot2, output, 
                pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

            return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()

            print("Grad Output", grad_output.shape)
            print("Input Dims", input1.shape)

            correlation_pwc.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

            return grad_input1, grad_input2, None, None, None, None, None, None 


class Correlation(torch.nn.Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()

        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        return CorrelationFunction.apply(input1, input2, self.pad_size, self.kernel_size, \
                            self.max_displacement, self.stride1, self.stride2, self.corr_multiply)

if __name__ == '__main__':
    import time
    import random

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    corr = Correlation(max_displacement=2, kernel_size=3, stride1=1, stride2=1, corr_multiply=1).to(device)
    # corr = CorrelationTorch(2)

    t_sum = 0

    for i in range(50):
        # C = random.choice([128, 256])
        # H = random.choice([128, 256])  # , 512
        # W = random.choice([64, 128])  # , 256
        C = 3
        C = H = W = 124
        x1 = torch.randn(4, C, H, W, requires_grad=True).to(device)
        x2 = torch.randn(4, C, H, W, requires_grad=True).to(device)

        print("original dims", x1.shape)

        end = time.time()
        y = corr(x1, x2)
        t_f = time.time() - end

        end = time.time()
        y.sum().backward()
        t_b = time.time() - end

        print('Forward: {:.3f}ms, Backward: {:.3f}ms'.format(t_f * 100, t_b * 100))

        if i < 3:
            continue
        t_sum += t_b + t_f

    print('Sum: {:.3f}s'.format(t_sum))