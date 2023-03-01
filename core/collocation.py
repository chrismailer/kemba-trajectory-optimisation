from pyomo.environ import Constraint


cp_error_msg = 'Number of collocation points incorrect for selected integrator'

def explicit_euler(q, dq):
    def func(m,n,cp,dof):
        assert cp == 1 and len(m.cp) == 1, cp_error_msg
        if n > 1:
            return q[n,cp,dof] == q[n-1,cp,dof] + m.hm*m.h[n] * dq[n-1,cp,dof]
        else: # n=1
            return Constraint.Skip
    return func


def implicit_euler(q, dq):
    def func(m,n,cp,dof):
        assert cp == 1 and len(m.cp) == 1, cp_error_msg
        if n > 1:
            return q[n,cp,dof] == q[n-1,cp,dof] + m.hm*m.h[n] * dq[n,cp,dof]
        else: # n=1
            return Constraint.Skip
    return func


def radau_2(q, dq):
    R = [   [0.416666125000187, -0.083333125000187],
            [0.749999625000187,  0.250000374999812],    ]
    
    def func(m,n,cp,dof):
        assert 1 <= cp <= 2 and len(m.cp) == 2, cp_error_msg
        if n > 1:
            inc = sum(R[cp-1][pp-1]*dq[n,pp,dof] for pp in m.cp)
            return q[n,cp,dof] == q[n-1,m.cp[-1],dof] + m.hm*m.h[n] * inc
        else: # n=1
            return Constraint.Skip
    return func


def radau_3(q, dq):
    R = [   [0.19681547722366, -0.06553542585020,  0.02377097434822],
            [0.39442431473909,  0.29207341166523, -0.04154875212600],
            [0.37640306270047,  0.51248582618842,  0.11111111111111],   ]
    
    def func(m,n,cp,dof):
        assert 1 <= cp <= 3 and len(m.cp) == 3, cp_error_msg
        if n > 1:
            inc = sum(R[cp-1][pp-1]*dq[n,pp,dof] for pp in m.cp)
            return q[n,cp,dof] == q[n-1,m.cp[-1],dof] + m.hm*m.h[n] * inc
        else: # n=1
            return Constraint.Skip
    return func
