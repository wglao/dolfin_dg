import enum
import ufl
import dolfinx
from petsc4py import PETSc


class MatrixType(enum.Enum):
    monolithic = enum.auto()
    block = enum.auto()
    nest = enum.auto()


class SeparateSpaceFormSplitter(ufl.corealg.multifunction.MultiFunction):

    def split(self, form, v, u=None):
        self.vu = tuple((v, u))
        return ufl.algorithms.map_integrands.map_integrand_dags(self, form)

    def argument(self, obj):
        if not obj in self.vu:
            return ufl.constantvalue.Zero(shape=obj.ufl_shape)
        return obj

    expr = ufl.corealg.multifunction.MultiFunction.reuse_if_untouched


def extract_rows(F, v):
    vn = len(v)
    L = [None for _ in range(vn)]

    fs = SeparateSpaceFormSplitter()

    for vi in range(vn):
        L[vi] = fs.split(F, v[vi])
        L[vi] = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(L[vi])
        L[vi] = ufl.algorithms.apply_derivatives.apply_derivatives(L[vi])

    return L


def extract_blocks(F, u, v):
    un, vn = len(u), len(v)
    a = [[None for _ in range(un)] for _ in range(vn)]

    fs = SeparateSpaceFormSplitter()

    for vi in range(vn):
        for ui in range(un):
            a[vi][ui] = fs.split(F, v[vi], u[ui])
            a[vi][ui] = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(a[vi][ui])
            a[vi][ui] = ufl.algorithms.apply_derivatives.apply_derivatives(a[vi][ui])

    return a


def extract_block_linear_system(F, u, v):
    F_a = extract_blocks(F, u, v)
    F_L = extract_rows(F, u, v)

    a = list(list(map(ufl.lhs, row)) for row in F_a)
    L = list(map(ufl.rhs, F_L))

    return a, L


def derivative_block(F, u, du=None, coefficient_derivatives=None):
    import ufl
    if isinstance(F, ufl.Form):
        return ufl.derivative(F, u, du, coefficient_derivatives)

    if not isinstance(F, (list, tuple)):
        raise TypeError("Expecting F to be a list of Forms. Found: %s" % str(F))

    if not isinstance(u, (list, tuple)):
        raise TypeError("Expecting u to be a list of Coefficients. Found: %s" % str(u))

    if du is not None:
        if not isinstance(du, (list, tuple)):
            raise TypeError("Expecting du to be a list of Arguments. Found: %s" % str(u))

    import itertools
    from ufl.algorithms.apply_derivatives import apply_derivatives
    from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering

    m, n = len(u), len(F)

    if du is None:
        du = [None] * m

    J = [[None for _ in range(m)] for _ in range(n)]

    for (i, j) in itertools.product(range(n), range(m)):
        gateaux_derivative = ufl.derivative(F[i], u[j], du[j], coefficient_derivatives)
        gateaux_derivative = apply_derivatives(apply_algebra_lowering(gateaux_derivative))
        if gateaux_derivative.empty():
            gateaux_derivative = None
        J[i][j] = gateaux_derivative

    return J


class NonlinearPDE_SNESProblem():
    def __init__(self, F, J, soln_vars, bcs, P=None):
        self.L = F
        self.a = J
        self.a_precon = P
        self.bcs = bcs
        self.soln_vars = soln_vars

    def F_mono(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x, self.soln_vars.vector.localForm() as _u:
            _u[:] = _x
        with F.localForm() as f_local:
            f_local.set(0.0)
        dolfinx.fem.assemble_vector(F, self.L)
        dolfinx.fem.apply_lifting(F, [self.a], [self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        J.zeroEntries()
        dolfinx.fem.assemble_matrix(J, self.a, self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfinx.fem.assemble_matrix(P, self.a_precon, self.bcs, diagonal=1.0)
            P.assemble()

    def F_block(self, snes, x, F):
        assert x.getType() != "nest"
        assert F.getType() != "nest"
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)

        offset = 0
        for var in self.soln_vars:
            size_local = var.vector.getLocalSize()
            var.vector.array[:] = x.array_r[offset:offset + size_local]
            var.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        dolfinx.fem.assemble_vector_block(F, self.L, self.a, self.bcs, x0=x, scale=-1.0)

    def J_block(self, snes, x, J, P):
        assert x.getType() != "nest" and J.getType() != "nest" and P.getType() != "nest"
        J.zeroEntries()
        dolfinx.fem.assemble_matrix_block(J, self.a, self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfinx.fem.assemble_matrix_block(P, self.a_precon, self.bcs, diagonal=1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        assert x.getType() == "nest" and F.getType() == "nest"
        # Update solution
        x = x.getNestSubVecs()
        for x_sub, var_sub in zip(x, self.soln_vars):
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x, var_sub.vector.localForm() as _u:
                _u[:] = _x

        # Assemble
        bcs1 = dolfinx.cpp.fem.bcs_cols(dolfinx.fem.assemble._create_cpp_form(self.a), self.bcs)
        for L, F_sub, a, bc in zip(self.L, F.getNestSubVecs(), self.a, bcs1):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            dolfinx.fem.assemble_vector(F_sub, L)
            dolfinx.fem.apply_lifting(F_sub, a, bc, x0=x, scale=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form(self.L), self.bcs)
        for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, x):
            dolfinx.fem.set_bc(F_sub, bc, x_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(self, snes, x, J, P):
        assert x.getType() == "nest" and J.getType() == "nest" and P.getType() == "nest"
        J.zeroEntries()
        dolfinx.fem.assemble_matrix_nest(J, self.a, self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfinx.fem.assemble_matrix_nest(P, self.a_precon, self.bcs, diagonal=1.0)
            P.assemble()
