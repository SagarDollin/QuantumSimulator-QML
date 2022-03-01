import MyQuantumSimulator
import torch

def get_device(gpu_no):
    if torch.cuda.is_available():
        return torch.device('cuda', gpu_no)
    else:
        return torch.device('cpu')
device = get_device(0)

def circuit(params):
    qc = MyQuantumSimulator.Circuit(1)
    qc.Rx(params[0],nth_qubit=0)
    qc.Ry(params[1],nth_qubit=0)
    return qc.expected_value_Z()

def cost(params):
    return circuit(params)

params = torch.tensor([[0.011], [0.012]], requires_grad=True, device=device, dtype=torch.cfloat)
print("cost before training",cost(params))
print("###############################")
optimizer = torch.optim.Adagrad([params], lr=0.2)
steps = 100
for i in range(steps):
    # update the circuit parameters
    loss = cost(params)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)[0][0]))

print("################################################")
print("Optimized rotation angles: {}".format(params))



