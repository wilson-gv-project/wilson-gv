from wilson_suite.wilson_main.simulation_core import SimulationCore
from wilson_suite.wilson_experiment.canned_experiments.evv_2scan import evv_2scan

def test_simulation_core():

    # Testing with the "canned" EVV-2-scan experiment
    tot_anharm_limit = 1
    mech_anharm_limit = 1
    el_anharm_limit = 1

    sim_core = SimulationCore(experiment = evv_2scan(), total_anharm_limit = tot_anharm_limit,
                              mech_anharm_limit = mech_anharm_limit, el_anharm_limit = el_anharm_limit)

    # One unique order of perturbation is common to this experiment
    assert len(sim_core.terms_by_order_comp_magn.keys()) == 1


    # That order is 3 and there is only one compounding pattern (1, 1, 2), which during the scan
    # visits three different magnitude condition regimes
    assert len(sim_core.terms_by_order_comp_magn[3]) == 1
    assert len(sim_core.terms_by_order_comp_magn[3][(1, 1, 2)]) == 3
    assert len(sim_core.coll_magn_bank.keys()) == 3

    # The tuples containing (2) (here corresponds to the carrier freq of the third pulse) are
    # not relevant for understanding the below comments about magnitude conditions

    # This is the "typical" EVV setup (-k1 + k2 + k3 with |w_2| > |w_1|)
    assert sim_core.coll_magn_bank[0] == (((1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)), ((0,),))

    # This corresponds to a situation with no definite sign between w_1 and w_2
    assert sim_core.coll_magn_bank[1] == (((1,), (2,), (0, 2), (1, 2), (0, 1, 2)), ((0,),))

    # This corresponds to |w_1| > |w_2|
    assert sim_core.coll_magn_bank[2] == (((1,), (2,), (0, 2), (1, 2), (0, 1, 2)), ((0,), (0, 1)))

    # Magnitude condition key 0 has 14 terms (should correspond to the "typical" EVV terms)
    assert len(sim_core.terms_by_order_comp_magn[3][(1, 1, 2)][0].terms) == 14

    # Magnitude condition key 1 has 47 terms (they are not inspected in detail - here testing the organizing)
    assert len(sim_core.terms_by_order_comp_magn[3][(1, 1, 2)][1].terms) == 47

    # Magnitude condition key 2 has 34 terms (they are not inspected in detail - here testing the organizing)
    assert len(sim_core.terms_by_order_comp_magn[3][(1, 1, 2)][2].terms) == 34

    # The evaluation should need these five derivative properties and a maximum of three-quantum energy levels
    assert sorted(sim_core.props.names()) == sorted(['dipgrad', 'diphess', 'polgrad', 'cff', 'polhess'])
    assert sim_core.max_state_lvl == 3

    # Several different resonance condition pair quanta are collected over the three categories of terms
    assert sim_core.pair_quanta == {(0, 1), (1, 2), (2, 1), (0, 0), (1, 1), (0, 2)}
