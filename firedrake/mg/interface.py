from pyop2 import op2

from . import utils
from . import kernels


__all__ = ["prolong", "restrict", "inject"]


def check_arguments(coarse, fine):
    cfs = coarse.function_space()
    ffs = fine.function_space()
    hierarchy, lvl = utils.get_level(cfs.mesh())
    if hierarchy is None:
        raise ValueError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(ffs.mesh())
    if lvl >= flvl:
        raise ValueError("Coarse function must be from coarser space")
    if hierarchy is not fhierarchy:
        raise ValueError("Can't transfer between functions from different hierarchies")
    if coarse.ufl_shape != fine.ufl_shape:
        raise ValueError("Mismatching function space shapes")


def prolong(coarse, fine):
    check_arguments(coarse, fine)
    Vc = coarse.function_space()
    Vf = fine.function_space()
    if len(Vc) > 1:
        if len(Vc) != len(Vf):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(coarse.split(), fine.split()):
            prolong(in_, out)
        return

    coarse_coords = Vc.ufl_domain().coordinates
    fine_to_coarse = utils.fine_node_to_coarse_node_map(Vf, Vc)
    fine_to_coarse_coords = utils.fine_node_to_coarse_node_map(Vf, coarse_coords.function_space())
    kernel = kernels.prolong_kernel(coarse)

    # XXX: Should be able to figure out locations by pushing forward
    # reference cell node locations to physical space.
    # x = \sum_i c_i \phi_i(x_hat)
    node_locations = utils.physical_node_locations(Vf)
    # Have to do this, because the node set core size is not right for
    # this expanded stencil
    for d in [coarse, coarse_coords]:
        d.dat.global_to_local_begin(op2.READ)
        d.dat.global_to_local_end(op2.READ)
    op2.par_loop(kernel, fine.node_set,
                 fine.dat(op2.WRITE),
                 coarse.dat(op2.READ, fine_to_coarse),
                 node_locations.dat(op2.READ),
                 coarse_coords.dat(op2.READ, fine_to_coarse_coords))


def restrict(fine_dual, coarse_dual):
    check_arguments(coarse_dual, fine_dual)
    Vf = fine_dual.function_space()
    Vc = coarse_dual.function_space()
    if len(Vc) > 1:
        if len(Vc) != len(Vf):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(fine_dual.split(), coarse_dual.split()):
            restrict(in_, out)
        return
    coarse_dual.dat.zero()
    # XXX: Should be able to figure out locations by pushing forward
    # reference cell node locations to physical space.
    # x = \sum_i c_i \phi_i(x_hat)
    node_locations = utils.physical_node_locations(Vf)

    coarse_coords = Vc.ufl_domain().coordinates
    fine_to_coarse = utils.fine_node_to_coarse_node_map(Vf, Vc)
    fine_to_coarse_coords = utils.fine_node_to_coarse_node_map(Vf, coarse_coords.function_space())
    # Have to do this, because the node set core size is not right for
    # this expanded stencil
    for d in [coarse_coords]:
        d.dat.global_to_local_begin(op2.READ)
        d.dat.global_to_local_end(op2.READ)
    kernel = kernels.restrict_kernel(Vf, Vc)
    op2.par_loop(kernel, fine_dual.node_set,
                 coarse_dual.dat(op2.INC, fine_to_coarse),
                 fine_dual.dat(op2.READ),
                 node_locations.dat(op2.READ),
                 coarse_coords.dat(op2.READ, fine_to_coarse_coords))


def inject(fine, coarse):
    check_arguments(coarse, fine)
    Vf = fine.function_space()
    Vc = coarse.function_space()
    if len(Vc) > 1:
        if len(Vc) != len(Vf):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(fine.split(), coarse.split()):
            inject(in_, out)
        return
    # Algorithm:
    # Loop over coarse nodes
    # Have list of candidate fine cells for each coarse node
    # For each fine cell, pull back to reference space, determine if
    # coarse node location is inside.
    # With candidate cell found, evaluate fine dofs from relevant
    # function at coarse node location.
    #
    # For DG, for each coarse cell, instead:
    # solve inner(u_c, v_c)*dx_c == inner(f, v_c)*dx_c

    kernel, dg = kernels.inject_kernel(Vf, Vc)
    if not dg:
        node_locations = utils.physical_node_locations(Vc)

        fine_coords = Vf.ufl_domain().coordinates
        coarse_node_to_fine_nodes = utils.coarse_node_to_fine_node_map(Vc, Vf)
        coarse_node_to_fine_coords = utils.coarse_node_to_fine_node_map(Vc, fine_coords.function_space())

        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        for d in [fine, fine_coords]:
            d.dat.global_to_local_begin(op2.READ)
            d.dat.global_to_local_end(op2.READ)
        coarse.dat.zero()
        op2.par_loop(kernel, coarse.node_set,
                     coarse.dat(op2.INC),
                     node_locations.dat(op2.READ),
                     fine.dat(op2.READ, coarse_node_to_fine_nodes),
                     fine_coords.dat(op2.READ, coarse_node_to_fine_coords))
    else:
        coarse.dat.zero()
        coarse_coords = Vc.mesh().coordinates
        fine_coords = Vf.mesh().coordinates
        coarse_cell_to_fine_nodes = utils.coarse_cell_to_fine_node_map(Vc, Vf)
        coarse_cell_to_fine_coords = utils.coarse_cell_to_fine_node_map(Vc, fine_coords.function_space())
        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        for d in [fine, fine_coords]:
            d.dat.global_to_local_begin(op2.READ)
            d.dat.global_to_local_end(op2.READ)
        op2.par_loop(kernel, Vc.mesh().cell_set,
                     coarse.dat(op2.INC, coarse.cell_node_map()),
                     fine.dat(op2.READ, coarse_cell_to_fine_nodes),
                     fine_coords.dat(op2.READ, coarse_cell_to_fine_coords),
                     coarse_coords.dat(op2.READ, coarse_coords.cell_node_map()))
