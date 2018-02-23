from firedrake import *


def test_repeated_custom_transfer():
    mesh = UnitIntervalMesh(2)
    mh = MeshHierarchy(mesh, 1)
    mesh = mh[-1]
    count = [0]

    def myprolong(coarse, fine):
        prolong(coarse, fine)
        count[0] += 1

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx
    L = v*dx

    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh)

    options = {"ksp_type": "preonly",
               "pc_type": "mg"}

    solver = LinearVariationalSolver(problem, solver_parameters=options)

    with dmhooks.transfer_operators(V, prolong=myprolong):
        solver.solve()

    assert count[0] == 1

    uh.assign(0)

    # from IPython import embed; embed()
    # For this test to work, need to rebuild PETSc MG transfer ops.
    # with dmhooks.transfer_operators(V, prolong=myprolong):
    solver.solve()

    assert count[0] == 1
