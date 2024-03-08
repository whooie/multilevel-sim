import sympy as sy
ph = sy.symbols("phi", real=True)

eiph = sy.exp(sy.I * ph)
rt2 = sy.sqrt(2)
S = sy.S
I = sy.I

P1 = sy.Matrix([
    [1, 0,          0,          0           ],
    [0, eiph / rt2, 0,          0           ],
    [0, 0,          eiph / rt2, 0           ],
    [0, I / rt2,    0,          0           ],
    [0, 0,          I / rt2,    0           ],
    [0, 0,          0,          eiph / 2    ],
    [0, 0,          0,          I * eiph / 2],
    [0, 0,          0,          I * eiph / 2],
    [0, 0,          0,          -S(1) / 2   ]
])

P2 = sy.Matrix([
    [1, 0,       0,       0,       0,       0,         0,       0,         0        ],
    [0, I / rt2, 0,       0,       0,       0,         0,       0,         0        ],
    [0, 0,       I / rt2, 1 / rt2, 0,       0,         0,       0,         0        ],
    [0, 0,       0,       0,       1 / rt2, 0,         0,       0,         0        ],
    [0, 1 / rt2, 0,       0,       0,       0,         0,       0,         0        ],
    [0, 0,       1 / rt2, I / rt2, 0,       0,         0,       0,         0        ],
    [0, 0,       0,       0,       I / rt2, 0,         0,       0,         0        ],
    [0, 0,       0,       0,       0,       -S(1) / 2, 0,       0,         0        ],
    [0, 0,       0,       0,       0,       I / 2,     0,       0,         0        ],
    [0, 0,       0,       0,       0,       I / 2,     0,       0,         0        ],
    [0, 0,       0,       0,       0,       S(1) / 2,  0,       0,         0        ],
    [0, 0,       0,       0,       0,       0,         I / rt2, 0,         0        ],
    [0, 0,       0,       0,       0,       0,         I / rt2, 0,         0        ],
    [0, 0,       0,       0,       0,       0,         0,       I / 2,     0        ],
    [0, 0,       0,       0,       0,       0,         0,       -S(1) / 2, 0        ],
    [0, 0,       0,       0,       0,       0,         0,       S(1) / 2,  0        ],
    [0, 0,       0,       0,       0,       0,         0,       I / 2,     0        ],
    [0, 0,       0,       0,       0,       0,         0,       0,         S(1) / 2 ],
    [0, 0,       0,       0,       0,       0,         0,       0,         I / 2    ],
    [0, 0,       0,       0,       0,       0,         0,       0,         I / 2    ],
    [0, 0,       0,       0,       0,       0,         0,       0,         -S(1) / 2]
])

print(P2 * P1)
print(sy.latex(P2 * P1))
