import sympy as sp

import numpy as np
import pandas as pd
sp.init_printing()

x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')

def assemble_gradU(U,V,W):
    dUdx = sp.Derivative(U,x)
    dUdy = sp.Derivative(U,y)
    dUdz = sp.Derivative(U,z)

    dVdx = sp.Derivative(V,x)
    dVdy = sp.Derivative(V,y)
    dVdz = sp.Derivative(V,z)

    dWdx = sp.Derivative(W,x)
    dWdy = sp.Derivative(W,y)
    dWdz = sp.Derivative(W,z)
    gradU = sp.Matrix([[dUdx, dUdy, dUdz],[dVdx, dVdy, dVdz],[dWdx, dWdy, dWdz]])
    return gradU

def evaluate_tensor_basis_components(gradU,cont_terms=None):
    if cont_terms == None:
        cont_terms = [gradU[0,0],gradU[1,1],gradU[2,2]]
    
    
    S = 1/2*(gradU + gradU.transpose())
    R = 1/2*(gradU - gradU.transpose())
    nonzero_components = np.ones((10,3,3))
    I = np.eye(3)
    T1 = S
    T2 = (S * R) - (R * S)
    T3 = (S * S) - 1/3 * sp.Trace((S * S))*I
    T4 = (R * R) - 1/3 * sp.Trace((R * R))*I
    T5 = (R * S * S) - (S * S * R)
    T6 = (R * R * S) + (S * R * R) - 2/3* sp.Trace((S * R * R))*I
    T7 = (R * S * R * R) - (R * R * S * R)
    T8 = (S * R * S * S) - (S * S * R * S)
    T9 = (R * R * S * S) + (S * S * R * R) - 2/3 * sp.Trace((S * S * R * R))*I
    T10 = (R * S * S * R * R) - (R * R * S * S * R)

    for Ti, tensor in enumerate([T1,T2,T3,T4,T5,T6,T7,T8,T9,T10]):
        print(f'\n Tensor: {Ti+1}')
        for i in range(3):
            for j in range(3):
                tensor[i,j] = sp.factor(sp.simplify(tensor[i,j])).subs(sum([x * 1.0 for x in cont_terms]),0).subs(sum(cont_terms),0)
                if tensor[i,j] == 0.0:
                    nonzero_components[Ti,i,j] = 0
        print(nonzero_components[Ti])

    return nonzero_components

cases = {
    '2D_XY': {
        'U': sp.Function('U')(x,y),
        'V': sp.Function('V')(x,y),
        'W': 0,
        'a': sp.Function('a')(x,y),
        'b': sp.Function('b')(x,y),
        'cont_terms': [0,1]
    },
    '2D_XZ': {
        'U': sp.Function('U')(x,z),
        'V': 0,
        'W': sp.Function('W')(x,z),
        'a': sp.Function('a')(x,z),
        'b': sp.Function('b')(x,z),
        'cont_terms': [0,2]
    },
    '2D_YZ': {
        'U': 0,
        'V': sp.Function('V')(y,z),
        'W': sp.Function('W')(y,z),
        'a': sp.Function('a')(y,z),
        'b': sp.Function('b')(y,z),
        'cont_terms': [1,2],
    },
    'DUCT': {
        'U': sp.Function('U')(y,z),
        'V': 0,
        'W': 0,
        'a': sp.Function('a')(y,z),
        'b': sp.Function('b')(y,z),
        'cont_terms': None
    }
}

for case in cases.keys():
    print(f'\n=========== CASE: {case} ===========\n')
    gradU = assemble_gradU(cases[case]['U'],cases[case]['V'],cases[case]['W'])
    if cases[case]['cont_terms'] is not None: cont_terms = [gradU[cases[case]['cont_terms'][0],cases[case]['cont_terms'][0]], gradU[cases[case]['cont_terms'][1],cases[case]['cont_terms'][1]]]
    else: cont_terms = None
    nonzero_components = evaluate_tensor_basis_components(gradU,cont_terms)



