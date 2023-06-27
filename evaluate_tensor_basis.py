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

def assemble_Av(v):
    dvdx = sp.Derivative(v,x)
    dvdy = sp.Derivative(v,y)
    dvdz = sp.Derivative(v,z)
    Av = sp.Matrix([[0, -dvdz, dvdy],[dvdz, 0, -dvdx],[-dvdy, dvdx, 0]])
    return Av

def evaluate_basis_tensor_invariants(gradU,Aa,Ab,cont_terms=None):
    if cont_terms == None:
        cont_terms = [gradU[0,0],gradU[1,1],gradU[2,2]]
    
    #cont_eqn =[]
    #cont_eqn.append()
    #cont_eqn.append()
    
    S = 1/2*(gradU + gradU.transpose())
    R = 1/2*(gradU - gradU.transpose())
    nonzero_I1 = np.ones(47,dtype=int)
    nonzero_I2 = np.ones(47,dtype=int)

    B1 = S * S
    B2 = S * S * S
    B3 = R * R
    B4 = Aa * Aa
    B5 = Ab * Ab
    B6 = R * R * S
    B7 = R * R * S * S
    B8 = R * R * S * R * S * S
    B9 = Aa * Aa * S
    B10= Aa * Aa * S * S
    B11= Aa * Aa * S * Aa * S * S
    B12= Ab * Ab * S
    B13= Ab * Ab * S * S
    B14= Ab * Ab * S * Ab * S * S
    B15= R * Aa
    B16= Aa * Ab
    B17= R * Ab

    B18= R * Aa * S
    B19= R * Aa * S * S

    B20= R * R * Aa * S
    B21= Aa * Aa * R * S

    B22= R * R * Aa * S * S
    B23= Aa * Aa * R * S * S	

    B24= R * R * S * Aa * S * S
    B25= Aa * Aa * S * R * S * S

    B26= R * Ab * S
    B27= R * Ab * S * S

    B28= R * R * Ab * S
    B29= Ab * Ab * R * S

    B30= R * R * Ab * S * S
    B31= Ab * Ab * R * S * S	

    B32= R * R * S * Ab * S * S
    B33= Ab * Ab * S * R * S * S

    B34= Aa * Ab * S
    B35= Aa * Ab * S * S

    B36= Aa * Aa * Ab * S
    B37= Ab * Ab * Aa * S

    B38= Aa * Aa * Ab * S * S
    B39= Ab * Ab * Aa * S * S	

    B40= Aa * Aa * S * Ab * S * S
    B41= Ab * Ab * S * Aa * S * S

    B42= R * Aa * Ab

    B43= R * Aa * Ab * S
    B44= R * Ab * Aa * S
    B45= R * Aa * Ab * S * S
    B46= R * Ab * Aa * S * S
    B47= R * Aa * S * Ab * S * S

    for i, basis_tensor in enumerate([B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,
                        B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,B33,B34,B35,B36,B37,B38,
                        B39,B40,B41,B42,B43,B44,B45,B46,B47]):
        print(f'I1(B{i+1}):')
        expr = sp.factor(sp.simplify(sp.trace(basis_tensor))).subs(sum([x * 1.0 for x in cont_terms]),0).subs(sum(cont_terms),0)
        if expr == 0.0:
            nonzero_I1[i] = 0

        sp.pprint(expr)

    for i, basis_tensor in enumerate([B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,
                        B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,B33,B34,B35,B36,B37,B38,
                        B39,B40,B41,B42,B43,B44,B45,B46,B47]):
        print(f'I2(B{i+1}):')
        expr = sp.factor(sp.simplify(0.5*((sp.trace(basis_tensor))**2 - sp.trace(basis_tensor**2) ))).subs(sum([x * 1.0 for x in cont_terms]),0).subs(sum(cont_terms),0)
        sp.pprint(expr)
        if expr == 0.0:
            nonzero_I2[i] = 0
    return nonzero_I1, nonzero_I2





results = pd.DataFrame()
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
        'cont_terms': [1,2]
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

results['Tensor_name'] =['B%d' % i for i in range(1, 48)]

for case in cases.keys():
    print(f'=========== CASE: {case} ===========')
    gradU = assemble_gradU(cases[case]['U'],cases[case]['V'],cases[case]['W'])
    Aa = assemble_Av(cases[case]['a'])
    Ab = assemble_Av(cases[case]['b'])
    if cases[case]['cont_terms'] is not None: cont_terms = [gradU[cases[case]['cont_terms'][0],cases[case]['cont_terms'][0]], gradU[cases[case]['cont_terms'][1],cases[case]['cont_terms'][1]]]
    else: cont_terms = None
    results[f'{case}_I1'], results[f'{case}_I2'] = evaluate_basis_tensor_invariants(gradU,Aa,Ab,cont_terms)

print(results)


