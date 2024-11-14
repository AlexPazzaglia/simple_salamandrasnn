# MODELS
def neural_equations():
    ''' Define equations for the neuron models '''

    neuron_eqs = '''
    dv/dt = 1/ tau_memb * ( -gleak*(v- V_rest ) - alpha1*w1 - alpha2*w2 + I_tot ) : volt (unless refractory)

    dw1/dt = (-w1)/tau1 : volt (unless refractory)
    dw2/dt = (-w2)/tau2 : volt (unless refractory)

    I_tot = ( R_memb*I1  + I_ampa + I_nmda + I_glyc ) : volt

    I_ampa = w_ampa * g_ampa_tot * (E_ampa - v) : volt
    I_nmda = w_nmda * g_nmda_tot * (E_nmda - v) : volt
    I_glyc = w_glyc * g_glyc_tot * (E_glyc - v) : volt

    dg_ampa_tot/dt = - (g_ampa_tot)/( tau_ampa ) : 1
    dg_nmda_tot/dt = - (g_nmda_tot)/( tau_nmda ) : 1
    dg_glyc_tot/dt = - (g_glyc_tot)/( tau_glyc ) : 1

    I1 : amp

    # Neuronal parameters
    tau_memb : second (constant)
    V_rest   :   volt (constant)
    V_reset   :   volt (constant)
    V_thres   :   volt (constant)
    std_val  :      1 (constant)
    R_memb    :    ohm (constant)
    gleak    :      1 (constant)
    alpha1   :      1 (constant)
    alpha2   :      1 (constant)
    tau1     : second (constant)
    tau2     : second (constant)
    delta_w1  :   volt (constant)
    delta_w2  :   volt (constant)

    # Synaptic parameters
    w_ampa      :    1   (constant)
    E_ampa      :  volt  (constant)
    tau_ampa    : second (constant)
    deltag_ampa :    1   (constant)

    w_nmda      :    1   (constant)
    E_nmda      :  volt  (constant)
    tau_nmda    : second (constant)
    deltag_nmda :    1   (constant)

    w_glyc      :    1   (constant)
    E_glyc      :  volt  (constant)
    tau_glyc    : second (constant)
    deltag_glyc :    1   (constant)

    # Neuronal identifiers
    y        :      1 (constant)
    side_id  :      1 (constant)
    ner_id   :      1 (constant)

    '''

    reset_eqs = '''
    v = V_reset
    w1 = w1 + delta_w1
    w2 = w2 + delta_w2
    '''

    threshold_eqs = '( v >= V_thres )'

    return neuron_eqs, reset_eqs, threshold_eqs

def synaptic_equations():
    ''' Define equations for the synapses models '''

    syn_ex_eq = '''link = 1 : 1'''

    syn_ex_on_pre = '''
    g_ampa_tot_post += deltag_ampa_post * int(not_refractory_post)
    g_nmda_tot_post += deltag_nmda_post * int(not_refractory_post)
    '''

    syn_in_eq = '''link = 1 : 1'''

    syn_in_on_pre = '''
    g_glyc_tot_post += deltag_glyc_post * int(not_refractory_post)
    '''

    return syn_ex_eq, syn_ex_on_pre, syn_in_eq, syn_in_on_pre
