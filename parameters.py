import numpy as np
import brian2 as b2

# PARAMETERS
def network_parameters():
    ''' Define neural and synaptic parameters '''

    # All neurons
    shared_neural_params = {
            'V_rest'     : [-70.0, 'mV'],
            'V_reset'    : [-70.0, 'mV'],
            'V_thres'    : [-38.0, 'mV'],

            'tau_memb'  : [150.0, 'ms'],
            'w_ampa'    : [  6.0,   ''],
            'w_nmda'    : [  1.5,   ''],
            'w_glyc'    : [ 10.0,   ''],
        }

    # Axial neurons
    variable_neural_params_axs = {
            'gleak'     : [   5.6,   ''],
            'alpha1'    : [  45.0,   ''],
            'alpha2'    : [  15.0,   ''],
            'tau1'      : [ 150.0, 'ms'],
            'tau2'      : [2000.0, 'ms'],
            'delta_w1'   : [  0.99, 'mV'],
            'delta_w2'   : [ 0.025, 'mV'],
            'R_memb'     : [[89.0*1e6, 91.0*1e6], 'ohm'],
        }

    # Limb neurons
    variable_neural_params_lmb = {
            'gleak'     : [   4.4,   ''],
            'alpha1'    : [  25.0,   ''],
            'alpha2'    : [  15.0,   ''],
            'tau1'      : [ 400.0, 'ms'],
            'tau2'      : [3200.0, 'ms'],
            'delta_w1'   : [  0.65, 'mV'],
            'delta_w2'   : [ 0.025, 'mV'],
            'R_memb'     : [[85.0*1e6, 86.0*1e6], 'ohm'],
        }

    # Shared synaptic parameters stored in neurons
    shared_syn_ex_params = {
            'E_ampa'     : [ 0.0,  'mV'],
            'E_nmda'     : [ 0.0,  'mV'],
            'deltag_ampa': [ 0.1,    ''],
            'deltag_nmda': [ 0.1,    ''],
            'tau_ampa'   : [20.0,  'ms'],
            'tau_nmda'   : [100.0, 'ms'],
        }

    shared_syn_in_params = {
            'E_glyc'     : [-85.0, 'mV'],
            'deltag_glyc': [  0.1,   ''],
            'tau_glyc'   : [ 20.0, 'ms'],
        }

    return (
        shared_neural_params,
        variable_neural_params_axs,
        variable_neural_params_lmb,
        shared_syn_ex_params,
        shared_syn_in_params
    )

# PARAMETERS ASSIGNMENT
def set_neural_identifiers(
    pop            : b2.NeuronGroup,
    pools_indices  : dict[str, tuple[tuple[int]]],
    limbs_positions: tuple[int]
) -> None:
    '''
    Define internal parameters to distinguish neurons

    SIDE_ID
    'ax' : +- 1, # Axial (left, right)
    'lb' : +- 2, # Limb  (left, right)

    NER_ID
    'ex': 0,  # excitatory
    'in': 1,  # inhibitory
    '''

    # Pools indices
    ex_ind_l = pools_indices['el']
    ex_ind_r = pools_indices['er']
    in_ind_l = pools_indices['il']
    in_ind_r = pools_indices['ir']

    # Axial and limb segments
    segments_limbs = len(limbs_positions)
    segments_axial = len(ex_ind_l) - segments_limbs

    # AXIS
    for seg in range(segments_axial):

        # ner_id
        pop[ex_ind_l[seg]].ner_id = 0
        pop[in_ind_l[seg]].ner_id = 1
        pop[ex_ind_r[seg]].ner_id = 0
        pop[in_ind_r[seg]].ner_id = 1

        # side_id
        pop[ex_ind_l[seg]].side_id = -1
        pop[in_ind_l[seg]].side_id = -1
        pop[ex_ind_r[seg]].side_id = +1
        pop[in_ind_r[seg]].side_id = +1

        # y_position
        pop[ex_ind_l[seg]].y = seg
        pop[in_ind_l[seg]].y = seg
        pop[ex_ind_r[seg]].y = seg
        pop[in_ind_r[seg]].y = seg

    # LIMBS
    for seg, lmb_pos in enumerate(limbs_positions):

        lb_seg = segments_axial + seg

        # ner_id
        pop[ex_ind_l[lb_seg]].ner_id = 0
        pop[in_ind_l[lb_seg]].ner_id = 1
        pop[ex_ind_r[lb_seg]].ner_id = 0
        pop[in_ind_r[lb_seg]].ner_id = 1

        # side_id
        pop[ex_ind_l[lb_seg]].side_id = -2
        pop[in_ind_l[lb_seg]].side_id = -2
        pop[ex_ind_r[lb_seg]].side_id = +2
        pop[in_ind_r[lb_seg]].side_id = +2

        # y_position
        pop[ex_ind_l[lb_seg]].y = lmb_pos
        pop[in_ind_l[lb_seg]].y = lmb_pos
        pop[ex_ind_r[lb_seg]].y = lmb_pos
        pop[in_ind_r[lb_seg]].y = lmb_pos

def set_neural_parameters(
    pool      : b2.NeuronGroup,
    ids       : list[int],
    std_value : float,
    parameters: dict[str, list[float, str]] = None
):
    '''
    Set the value of the parameters for the specified indeces. Used for values that are
    specific for a pupulation subset.\n
    parameters = { 'parameter1' : [ value1, unit1] }.\n
    - Numbers are considered as fixed quantities\n
    - Lists of one element are considered gaussian distributed with specified std_val\n
    - Lists of two elements are considered uniform distributed between the two values\n
    '''

    if not bool(ids) \
        or not isinstance(ids, (list, range, np.ndarray) ) \
            or not parameters:
        return pool

    for param, data in parameters.items():
        value = data[0]
        unit = data[1]

        try:
            getattr(pool[0], param)
        except AttributeError:
            continue

        if isinstance(value, list):
            if len(value) == 1:
                # Normal distribution
                if unit == '':
                    setattr(pool[ids], param, 'randn()*{std}*{M} + {M}'.format(
                        M=value[0],
                        std=std_value))
                else:
                    setattr(pool[ids], param, 'randn()*{std}*{M}*{un} + {M}*{un}'.format(
                        M=value[0],
                        un=unit,
                        std=std_value))

            if len(value) == 2:
                # Uniform distribution
                if unit == '':
                    setattr(pool[ids], param, 'rand()*({M}-{m}) + {m}'.format(
                        m=value[0],
                        M=value[1]))
                else:
                    setattr(pool[ids], param, 'rand()*({M}-{m})*{un} + {m}*{un}'.format(
                        m=value[0],
                        M=value[1],
                        un=unit))
        else:
            if unit == '':
                setattr(pool[ids], param, f"{value}")
            else:
                setattr(pool[ids], param, f"{value}*{unit}")

    return pool

def set_synaptic_parameters(
    syn       : b2.Synapses,
    ilimits   : list[int],
    jlimits   : list[int],
    std_value : float,
    parameters: dict[str, list[float, str]] = None
):
    '''
    Set the parameters for the synaptic synaptic connections. Used for values that are specific
    for connections between pupulation subsets.\n
    parameters = { 'parameter1' : [ value1, unit1] }.\n
    - Numbers are considered as fixed quantities\n
    - Lists of one element are considered gaussian distributed with specified std_val\n
    - Lists of two elements are considered uniform distributed between the two values\n
    '''

    if not bool(ilimits) or not bool(jlimits) or not parameters:
        return syn

    # Substitute the desired values
    for param, data in parameters.items():

        try:
            attr = getattr(syn, param)
        except AttributeError:
            continue

        value = data[0]
        unit = data[1]

        if isinstance(value, list):
            if len(value) == 1:
                # Normal distribution
                if unit == '':
                    valuetoset = 'randn()*{std}*{M} + {M}'.format(M=value[0],
                                                                  std=std_value)
                else:
                    valuetoset = 'randn()*{std}*{M}*{un} + {M}*{un}'.format(M=value[0],
                                                                            un=unit,
                                                                            std=std_value)

            elif len(value) == 2:
                # Uniform distribution
                if unit == '':
                    valuetoset = 'rand()*({M}-{m}) + {m}'.format(m=value[0],
                                                                 M=value[1])
                else:
                    valuetoset = 'rand()*({M}-{m})*{un} + {m}*{un}'.format(m=value[0],
                                                                           M=value[1],
                                                                           un=unit)
        else:
            if unit == '':
                valuetoset = f"{value}"
            else:
                valuetoset = f"{value}*{unit}"

        # Check if the synapse is connecting neurons of interest
        target_syn = np.intersect1d(np.where((syn.i >= ilimits[0]) * (syn.i <= ilimits[-1])),
                                    np.where((syn.j >= jlimits[0]) * (syn.j <= jlimits[-1])))

        attr[target_syn] = valuetoset
        setattr(syn, param, attr)

    return syn
