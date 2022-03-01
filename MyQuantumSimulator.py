import torch
import math



def get_device(gpu_no):
    if torch.cuda.is_available():
        return torch.device('cuda', gpu_no)
    else:
        return torch.device('cpu')


class Circuit:
    # initialize qubits and state vector
    def __init__(self, n_qubits, device='cuda', gpu_no=0):
        """This functions takes the number of qubits in the circuit and initializes all the qubits in state |0> by
        default and the state vector corresponding to the qubits.

        parameters:
        n_qubits <- int()
        device <- str()  (cpu or gpu)
        gpu_no <- int()
        """

        self.n = n_qubits
        if device != 'cuda':
            self.device = torch.device(device)
        else:
            self.device = get_device(gpu_no)

        self.qubits = torch.tensor([[1, 0]] * n_qubits, device=self.device, dtype=torch.cfloat)
        self.initialize(range(self.n), self.qubits)
        self.I = torch.tensor([[1., 0.], [0., 1.]], device=self.device, dtype=torch.cfloat)
        self.z_gate = torch.tensor(([[1, 0], [0, -1]]), device=self.device, dtype=torch.cfloat)
        self.y_gate = torch.tensor(([[0, -1j], [1j, 0]]), device=self.device, dtype=torch.cfloat)
        self.h_gate = 1 / math.sqrt(2) * torch.tensor([[1., 1.], [1., -1.]], device=self.device, dtype=torch.cfloat)
        self.x_gate = torch.tensor(([[0., 1.], [1., 0.]]), device=self.device, dtype=torch.cfloat)

    def initialize(self, list_qubits, vectors):
        """You can use this function to initialize qubits manually. It takes in a list of qubits and their
        corresponding unit vectors as arguments and initializes the state vector accordingly.

        parameters:
        list_qubits <- list()
        vectors <- torch.tensor().dtype == torch.cfloat  (keep on same device)
         """
        try:
            for i in range(len(list_qubits)):
                self.qubits[list_qubits[i]] = vectors[i]
        except IndexError:
            print("Length of list_qubits and vectors must be equal!")
            return
        if self.n > 1:
            self.state_vector = torch.kron(self.qubits[0], self.qubits[1])
            for i in range(2, self.n):
                self.state_vector = torch.kron(self.state_vector, self.qubits[i])
        else:
            self.state_vector = self.qubits
        self.state_vector = self.state_vector.T

    def apply_single_gate(self, nth_qubit, gate):
        """This function  takes in arguments the qubit on which you want to perform operation and the gate/type of
        operation you want to perform on the given qubit, and performs the matrix operation to update the
        state_vector. """
        if nth_qubit != 0:
            gate_matrix = self.I
        else:
            gate_matrix = gate

        for i in range(1, self.n):
            if i != nth_qubit:
                gate_matrix = torch.kron(gate_matrix, self.I)
            else:
                gate_matrix = torch.kron(gate_matrix, gate)

        self.state_vector = torch.matmul(gate_matrix, self.state_vector)
        return self.state_vector

    def apply_multiple_gate(self, control_list, target_list, gate):
        """This function  takes in arguments the list of control qubits and list of target qubits on which you want to
        perform operation and the gate/type of operation you want to perform on the given qubit, and performs the
        matrix operation to update the state_vector. """

        self.p_zero = torch.tensor(([[1, 0], [0, 0]]), device=self.device, dtype=torch.cfloat)
        self.p_one = torch.tensor(([[0, 0], [0, 1]]), device=self.device, dtype=torch.cfloat)
        controls = [self.p_zero, self.p_one]
        gate_matrix = torch.tensor([1.], device=self.device, dtype=torch.cfloat)

        for i in range(2 ** len(control_list)):
            binary_str = str(bin(i))
            binary_str = binary_str[2:]
            padding = len(control_list) - len(binary_str)
            if padding != 0:
                binary_str = '0' * padding + binary_str
            k = 0

            for j in range(self.n):
                if j in control_list:
                    gate_matrix = torch.kron(gate_matrix, controls[int(binary_str[k])])
                    k += 1
                elif (i < (2 ** len(control_list) - 1) or j not in target_list):
                    gate_matrix = torch.kron(gate_matrix, self.I)

                elif (i == (2 ** len(control_list) - 1) and j in target_list):
                    gate_matrix = torch.kron(gate_matrix, gate)

            if i == 0:
                linear_gate_matrix = gate_matrix
            else:
                linear_gate_matrix += gate_matrix

            gate_matrix = torch.tensor([1.], device=self.device, dtype=torch.cfloat)
        self.state_vector = torch.matmul(linear_gate_matrix, self.state_vector)
        return self.state_vector

    def x(self, nth_qubit):
        """This function performs x gate operation on given qubit by passing the given qubit and the x gate matrix for
        single qubit to the apply_single_gate() function. """
        self.apply_single_gate(nth_qubit, self.x_gate)

    def h(self, nth_qubit):
        """This function performs h gate operation on given qubit by passing the given and the h gate matrix for
        single qubit to the apply_single_gate() function. """
        self.apply_single_gate(nth_qubit, self.h_gate)

    def y(self, nth_qubit):
        """This function performs y gate operation on given qubit by passing the given and the y gate matrix for
        single qubit to the apply_single_gate() function. """
        self.apply_single_gate(nth_qubit, self.y_gate)

    def z(self, nth_qubit):
        """This function performs z gate operation on given qubit by passing the given and the z gate matrix for
        single qubit to the apply_single_gate() function. """
        self.apply_single_gate(nth_qubit, self.z_gate)

    def Rx(self, theta, nth_qubit):
        """This function performs RX rotation gate operation on given qubit by passing the given qubit index and the
        z gate matrix for single qubit to the apply_single_gate() function. """
        co = torch.cos(theta / 2)
        si = torch.sin(theta / 2)
        self.Rx_gate = torch.stack([torch.cat([co, -si], dim=-1),
                                    torch.cat([-si, co], dim=-1)], dim=-2).squeeze(0)
        self.apply_single_gate(nth_qubit, self.Rx_gate)

    def Ry(self, theta, nth_qubit):
        """This function performs RY rotation gate operation on given qubit by passing the given qubit index and the
            Calculated gate matrix for single qubit to the apply_single_gate() function. """
        co = torch.cos(theta / 2)
        si = torch.sin(theta / 2)
        self.Ry_gate = torch.stack([torch.cat([co, -si]),
                                    torch.cat([si, co])], dim=-2).squeeze(0)
        self.apply_single_gate(nth_qubit, self.Ry_gate)

    def R(self, theta, phi, lamda, nth_qubit):
        """This function performs U3 general rotation gate operation on given qubit and angles: theta, phi,
        lamda by passing the given qubit index and the Calculated gate matrix for single qubit to the
        apply_single_gate() function. """
        a = torch.cos(theta / 2)
        b = -torch.exp(1j * lamda) * torch.sin(theta / 2)
        c = torch.exp(1j * phi) * torch.sin(theta / 2)
        d = torch.exp(1j * (phi + lamda)) * torch.cos(theta / 2)
        self.R_gate = torch.stack([torch.cat([a, b]),
                                    torch.cat([c, d])], dim=-2).squeeze(0)
        self.apply_single_gate(nth_qubit, self.R_gate)

    def cz(self, control_list, target_list):
        """This function takes in arguments as list of control qubits and list of target qubits to perform controlled
        z operation on the qubits by passing the given arguments and z gate for single qubit to the
        apply_multiuple_gate() function. """
        if (type(control_list) is not list):
            control_list = [control_list]
        if type(target_list) is not list:
            target_list = [target_list]

        self.apply_multiple_gate(control_list, target_list, self.z_gate)

    def cx(self, control_list, target_list):
        """This function takes in arguments as list of control qubits and list of target qubits to perform controlled
        x operation on the qubits by passing the given arguments and x gate for single qubit to the
        apply_multiuple_gate() function. """
        if type(control_list) is not list:
            control_list = [control_list]
        if type(target_list) is not list:
            target_list = [target_list]

        self.apply_multiple_gate(control_list, target_list, self.x_gate)

    def expected_value_Z(self):
        """This function returns the expectaed Pauli Z vaules of each qubit in the circuit"""
        for i in range(self.n):
            if i != 0:
                right_matrix = torch.eye(2 ** (self.n - 1 - i), device=self.device, dtype=torch.cfloat)
                left_matrix = torch.eye(2 ** i, device=self.device, dtype=torch.cfloat)
                matrix = torch.kron(torch.kron(left_matrix, self.z_gate), right_matrix)
                concatenated_matrix = torch.cat((concatenated_matrix, matrix), 0)
            else:
                right_matrix = torch.eye(2 ** (self.n - 1), device=self.device, dtype=torch.cfloat)
                concatenated_matrix = torch.kron(self.z_gate, right_matrix)

        transformation = concatenated_matrix @ self.state_vector
        transformation = transformation.view(self.n, -1).T
        expectation = self.state_vector.T @ transformation
        return expectation

    def probabilities(self):
        return torch.pow(self.state_vector, 2)
