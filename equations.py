# MODELS
def neural_equations():
    ''' Define equations for the neuron models '''

    neuron_eqs = '''
    dv/dt = 1/ tau_memb * ( - (v- V_rest ) - R_memb * w1 + I_tot ) : volt (unless refractory)

    dw1/dt = ( -w1 )/tau1 : ampere (unless refractory)

    I_tot = ( R_memb * I1  + I_ex + I_in ) : volt

    I_ex = w_ex * g_ex_tot * (E_ex - v) : volt
    I_in = w_in * g_in_tot * (E_in - v) : volt

    dg_ex_tot/dt = - (g_ex_tot)/( tau_ex ) : 1
    dg_in_tot/dt = - (g_in_tot)/( tau_in ) : 1

    I1 : amp

    # Neuronal parameters
    tau_memb  : second (constant)
    V_rest    :   volt (constant)
    V_reset   :   volt (constant)
    V_thres   :   volt (constant)
    std_val   :      1 (constant)
    R_memb    :    ohm (constant)
    tau1      : second (constant)
    delta_w1  : ampere (constant)

    # Synaptic parameters
    w_ex      :    1   (constant)
    E_ex      :  volt  (constant)
    tau_ex    : second (constant)

    w_in      :    1   (constant)
    E_in      :  volt  (constant)
    tau_in    : second (constant)

    # Neuronal identifiers
    y        :      1 (constant)
    side_id  :      1 (constant)
    ner_id   :      1 (constant)

    '''

    reset_eqs = '''
    v = V_reset
    w1 = w1 + delta_w1
    '''

    threshold_eqs = '( v >= V_thres )'

    return neuron_eqs, reset_eqs, threshold_eqs

def synaptic_equations():
    ''' Define equations for the synapses models '''

    syn_ex_eq     = '''link = 1 : 1'''
    syn_ex_on_pre = '''g_ex_tot_post += int(not_refractory_post)'''
    syn_in_eq     = '''link = 1 : 1'''
    syn_in_on_pre = '''g_in_tot_post += int(not_refractory_post)'''

    return syn_ex_eq, syn_ex_on_pre, syn_in_eq, syn_in_on_pre
