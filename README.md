# integrity-basis-input-features

The purpose of this repository is to analytically calculate the input feature expressions for integrity basis invariants in data-driven turbulence modelling. Using the analytical expressions, we can check which input features are zero for a given case. The integrity basis is constructed using 4 tensors: *S*, *R*, *Aa*, and *Ab*, where *a* and *b* are two scalars, and *Ai* is the antisymmetric tensor associated with the gradient of a given *i*. For certain flows, the partial derivatives of these variables in certain directions are zero, which produces zero input features.

Sample code and results are provided for the case of 2D flows, and flow through a duct. The code outputs results to `output.log`, and summarizes the zero/non-zero results in a .csv file.

# Checking a new flow case
The main input is the `cases` dictionary. In this dictionary, you specify which mean flow components of the flow are zero, and the coordinate directions that the other variables depend on. For example, the entry

```
'2D_XY': {
    'U': sp.Function('U')(x,y),
    'V': sp.Function('V')(x,y),
    'W': 0,
    'a': sp.Function('a')(x,y),
    'b': sp.Function('b')(x,y),
    'cont_terms': [0,1]
},
```
Means that the *z*-component of the velocity vector (*U*,*V*,*W*) is zero, and *U* and *V* only depend on the *x* and *y* coordinates. Therefore, the partial derivative ∂*U*/∂*z* is zero (likewise for *V*). The scalars *a* and *b* also only depend on the *x* and *y* coordinates. Lastly, the `cont_terms` entry needs to be provided. The purpose of this entry is to check if a given input feature is zero due to the continuity condition: ∂*U*/∂*x* + ∂*V*/∂*y* + ∂*W*/∂*z* = 0. We specify which terms appear in the continuity equation for this flow. In the above example, the *x* (which is index 0), and *y* (index 1) terms appear. The *z* term is index 2.



