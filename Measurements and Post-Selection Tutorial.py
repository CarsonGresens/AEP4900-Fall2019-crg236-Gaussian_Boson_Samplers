# Measurement and Post-Selection tutorial

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *

# Measurement
prog = sf.Program(2)
eng = sf.Engine("fock", backend_options={"cutoff_dim": 12})

with prog.context as q:
    Fock(5)       | q[0]
    Fock(6)       | q[1]
    BSgate()      | (q[0], q[1])
    MeasureFock() | q[0]

results = eng.run(prog)

print(results.samples[0],'\n',results.samples)

prog2 = sf.Program(2)
with prog2.context as q:
    MeasureFock() | q[1]
    
results = eng.run(prog2)

print(results.samples[1])

# Use Post Selection

prog3 = sf.Program(2)
eng = sf.Engine("fock", backend_options={"cutoff_dim": 6})

with prog3.context as q:
    Fock(2)               | q[0]
    Fock(3)               | q[1]
    BSgate()              | (q[0],q[1])
    MeasureFock(select=0) | q[0]
    MeasureFock()         | q[1]
    
result = eng.run(prog3)
print(result.samples)


# Gaussian Backend Example, 2 mode squeezed gate

prog4 = sf.Program(2)
eng = sf.Engine("gaussian")

with prog4.context as q:
#with eng:
    S2gate(1)                   | (q[0],q[1])
    MeasureHomodyne(0,select=1) | q[0]

state = eng.run(prog4).state
mu, cov = state.reduced_gaussian([1])
print(mu[0], mu, cov)

# User Defined Provessing Functions

@sf.convert
def log(x):
    if 0.5<x< 1:
        return np.log(x)
    else:
        return x

prog5 = sf.Program(2)
with prog5.context as q:
    MeasureX      | q[0]
    Xgate(log(q[0])) | q[1]
    
print(eng.run(prog5))



