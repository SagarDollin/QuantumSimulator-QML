import MyQuantumSimulator
import torch
import numpy as np
def get_device(gpu_no):
    if torch.cuda.is_available():
        return torch.device('cuda', gpu_no)
    else:
        return torch.device('cpu')
device = get_device(0)
qc = MyQuantumSimulator.Circuit(3) #This will create a Circuit with three qubits
print(qc.state_vector)
qc.initialize([0,1],torch.tensor([[0,1], [1/pow(2,0.5),1/pow(2,0.5)]], device=device, dtype=torch.cfloat))  #applies to qubit 0 and 1
print(qc.state_vector)
qc.x(0) #applies X gate on 0th qubit
qc.h(1) #applies H gate on 1st qubit
qc.y(2) #applies Y gate on the 2nd qubit
qc.z(0) #applies Z gate on the oth qubit
print(qc.state_vector) #remember the previous state wasn't the default state, you can compare the result by running the same
                #steps on qiskit's statevector simulator

print("#############################################")
qc.cx(2,0)
print(qc.state_vector)
qc.cx([2,0],1)  # since qubit 2 and 0 are set to state |1> the target qubit 1 is also flipped from state |0> to |1>
print(qc.state_vector)
qc.cz(0,1)  # if 0th qubit is in state |1> then the sign of the 1st qubit is changed
print(qc.state_vector)


print('#################################################')
c = MyQuantumSimulator.Circuit(1) #This will create a Circuit with one qubit
c.initialize(list_qubits=[0],vectors=torch.tensor([[0,1]], device=device, dtype=torch.cfloat))
print(c.state_vector)
torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.pi = torch.tensor([torch.pi], device=device, dtype=torch.cfloat)
c.Rx(theta=torch.pi,nth_qubit=0)
print(c.state_vector)

print('#################################################')
cq = MyQuantumSimulator.Circuit(1) #This will create a Circuit with one qubit
print(cq.state_vector)
torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.pi = torch.tensor([torch.pi], device=device, dtype=torch.cfloat)
cq.R(theta=torch.pi, phi=torch.pi, lamda=torch.pi, nth_qubit=0)
print(cq.state_vector)
norm = 0
for y in cq.state_vector:
    print(y[0])
    norm += (y[0]*np.conjugate(y[0].to('cpu')))
    print(norm)


