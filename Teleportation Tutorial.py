# Strawberry Fields Quantum Teleportation Tutorial
# https://strawberryfields.readthedocs.io/en/latest/tutorials/blackbird.html

import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import scale
from numpy import pi, sqrt


# Quantum Teleportation
prog = sf.Program(3)

@sf.convert
def custom(x):
    return-x*sqrt(2)

with prog.context as q:
    # Prepares initial states
    Coherent(1+0.5j) | q[0]
    Squeezed(-2) | q[1]
    Squeezed(2) | q[2]
    
    # apply gates
    BS = BSgate(pi/4, pi)
    BS | (q[1], q[2])
    BS | (q[0], q[1])
    
    # Perform Homodyne Measurements
    MeasureX | q[0]
    MeasureP | q[1]
    
    # Displacement gates conditioned on the measurements
    Xgate(scale(q[0], sqrt(2))) | q[2]
    Zgate(custom(q[1])) | q[2]
    
eng = sf.Engine('fock', backend_options = {"cutoff_dim": 15})

result = eng.run(prog)

print(result.samples, "\n", result.state)

state = result.state
print(state.dm().shape)

# Find reduced density matrix
rho_q2 = np.einsum('kkllij->ij',state.dm())
print(rho_q2.shape)

# Fock state has reduced_dm() method to automatically find reduced density matrix
T = False
if state.reduced_dm(2).all() == rho_q2.all():
    T = True
print(T)

# probabilities and probability density plots

probs = np.real_if_close(np.diagonal(rho_q2))
print(probs)

plt.bar(range(15), probs[:])
plt.xlabel('Fock state')
plt.ylabel('Marginal probability')
plt.title('Mode 2')
plt.show()

# Can automatically do via Fock state method all_fock_probs()

fock_probs = state.all_fock_probs()
print(fock_probs.shape, '\n' , np.sum(fock_probs, axis=(0,1)))


