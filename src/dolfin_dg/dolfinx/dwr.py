import numpy as np
from petsc4py import PETSc

import ufl
import dolfinx as dfx
import dolfinx.fem as fem


def dual(form, w, z=None):
    if z is not None:
        v, u = form.arguments()
        return ufl.replace(form, {u: w, v: z})
    u = form.arguments()[0]
    return ufl.replace(form, {u: w})


class NonlinearAPosterioriEstimator:

    def __init__(self, J, F, j, u_h, V_star, bcs_star=None):
        assert(len(F.arguments()) == 1)
        assert(len(J.arguments()) == 2)
        assert(len(j.arguments()) == 0)

        if bcs_star is None:
            bcs_star = []

        if not hasattr(bcs_star, "__len__"):
            bcs_star = (bcs_star,)

        self.F = F
        self.J = J
        self.j = j
        self.u_h = u_h
        self.V_star = V_star
        self.bcs_star = bcs_star

    def compute_indicators(self, petsc_options=None):
        z = self.compute_dual_solution(petsc_options=petsc_options)
        eta = self.compute_error_indicators(z)
        eta_cf = self.compute_cell_indicators(eta)
        return eta_cf

    def compute_dual_solution(self, petsc_options=None):
        # Replace the components of the bilinear and linear formulations to
        # formulate the dual problem
        # u -> w, v -> z
        w = ufl.TestFunction(self.V_star)
        z = ufl.TrialFunction(self.V_star)

        dual_M = dual(self.J, w, z)
        dual_j = ufl.derivative(self.j, self.u_h, w)

        # Solve the dual problem
        # z_s = fem.Function(self.V_star)
        if petsc_options is None:
            petsc_options = {"ksp_type": "preonly",
                             "pc_type": "lu"}
        problem = fem.petsc.LinearProblem(
            dual_M, dual_j, bcs=self.bcs_star, petsc_options=petsc_options)
        z_s = problem.solve()

        return z_s

    def compute_error_indicators(self, z):
        # Evaluate the residual in the enriched space
        v = self.F.arguments()[0]

        DG0 = fem.functionspace(self.V_star.mesh, ("DG", 0))
        dg_0 = ufl.TestFunction(DG0)

        dwr = ufl.replace(self.F, {v: z*dg_0})

        dwr_vec = fem.petsc.assemble_vector(
            fem.form(dwr))
        dwr_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dwr_vec.abs()
        return dwr_vec

    def compute_cell_indicators(self, eta):
        mesh = self.V_star.mesh

        # Put the values of the projection into cell tags
        cf = dfx.mesh.meshtags(
            mesh, mesh.topology.dim,
            np.arange(mesh.topology.index_map(
                mesh.topology.dim).size_local, dtype=np.int32), eta)
        return cf


class LinearAPosterioriEstimator(NonlinearAPosterioriEstimator):

    def __init__(self, a, L, j, u_h, V_star, bcs_star=None):
        assert(len(L.arguments()) == 1)
        assert(len(a.arguments()) == 2)
        assert(len(j.arguments()) == 1)

        F = a - L
        v, u = F.arguments()
        F = ufl.replace(F, {u: u_h})
        J = ufl.derivative(F, u_h)

        u = j.arguments()[0]
        j = ufl.replace(j, {u: u_h})

        NonlinearAPosterioriEstimator.__init__(
            self, J, F, j, u_h, V_star, bcs_star=bcs_star)
