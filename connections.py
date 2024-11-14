import numpy as np
import brian2 as b2

# CONNECTIVITY
def concond_byidentity( condlist: list,
                        extraconditions: str ='') -> str:
    '''
    Condition based on direction, crossing, location and synaptic type\n
    - extraconditions (if included) contains additional
    constrains to be linked by and operators\n
    Ex: [ [ 'up', 'ipsi', 'ax','ex', 'ax', ['ex','in'] ] ]\n
    Note: elements of the list can also be lists themselves
    (will be linked by and operators)\n
    Note: equality considered with tolerance to avoid errors
    due to numerical approximations
    '''

    def exp_from_dict(cond_elem, dic: dict) -> str:
        if isinstance(cond_elem, list):
            exp = '( '
            for i, cond in enumerate(cond_elem):
                if i == 0:
                    exp += dic[cond]
                else:
                    exp += ' or ' + dic[cond]

            exp += ' )'
        else:
            exp = dic[cond_elem]
        return exp

    dir_dict = {
        'up'  : '(y_pre > y_post + 0.001)',
        'dw'  : '(y_pre < y_post - 0.001)',
        'upeq': '(y_pre > y_post - 0.001)',
        'dweq': '(y_pre < y_post + 0.001)',
        'eq'  : '( abs( y_pre - y_post ) < 0.001  )',
    }

    contype_dict = {
        'ipsi'  : '( (side_id_pre * side_id_post) > 0 )',   # Ipsilateral
        'contra': '( (side_id_pre * side_id_post) < 0 )',   # Contralateral
    }

    axial_dict = {
        'ax' : '( abs(side_id) == 1 )',
        'lb' : '( abs(side_id) >= 2 )',
    }

    type_dict = {
        'ex': '(ner_id == 0)',  # excitatory
        'in': '(ner_id == 1)',  # inhibitory
    }

    output = '( '
    for i_x, cond in enumerate(condlist):

        # Substitute conditions with corresponding expression (AND)
        xycondlist = []

        if cond[0] != '':
            xycondlist.append(exp_from_dict(cond[0], dir_dict))

        if cond[1] != '':
            xycondlist.append(exp_from_dict(cond[1], contype_dict))

        xycond_exp = ''
        for i_c, xy_cond in enumerate(xycondlist):
            if i_c == 0:
                xycond_exp += xy_cond
            else:
                xycond_exp += ' and ' + xy_cond

        if xycond_exp != '':
            xycond_exp = ' ( ' + xycond_exp + ' ) '

        # Substitute conditions with corresponding expression (AND)
        xcondlist = []
        if cond[2] != '':
            xcondlist.append(
                exp_from_dict(cond[2], axial_dict).replace('side_id', 'side_id_pre'))

        if cond[3] != '':
            xcondlist.append(
                exp_from_dict(cond[3], type_dict).replace('ner_id', 'ner_id_pre'))

        xcond_exp = ''
        for i_c, x_cond in enumerate(xcondlist):
            if i_c == 0:
                xcond_exp += x_cond
            else:
                xcond_exp += ' and ' + x_cond

        if xcond_exp != '':
            xcond_exp = ' ( ' + xcond_exp + ' ) '

        # Substitute conditions with corresponding expression (AND)
        ycondlist = []

        if cond[4] != '':
            ycondlist.append(
                exp_from_dict(cond[4], axial_dict).replace('side_id', 'side_id_post'))

        if cond[5] != '':
            ycondlist.append(
                exp_from_dict(cond[5], type_dict).replace('ner_id', 'ner_id_post'))

        ycond_exp = ''
        for i_c, y_cond in enumerate(ycondlist):
            if i_c == 0:
                ycond_exp += y_cond
            else:
                ycond_exp += ' and ' + y_cond

        if ycond_exp != '':
            ycond_exp = ' ( ' + ycond_exp + ' ) '

        # Insert conditions in the output string (OR)
        cond = ''
        if xycond_exp != '':
            cond += '( ' + xycond_exp

        if xcond_exp != '' and cond == '':
            cond += '( ' + xcond_exp
        elif xcond_exp != '' and cond != '':
            cond += ' and ' + xcond_exp

        if ycond_exp != '' and cond == '':
            cond += '( ' + ycond_exp
        elif ycond_exp != '' and cond != '':
            cond += ' and ' + ycond_exp
        cond += ' )'

        if i_x == 0:
            output += cond
        else:
            output += ' or ' + cond
    output += ')'

    # Check if additional conditions were selected
    if extraconditions:
        output = extraconditions + ' and ' + output

    return '(i!=j) and ' + output

def connect_byidentity( syn: b2.Synapses,
                        condlist: list,
                        prob: float,
                        extraconditions: str ='') -> None:
    '''
    Connect according to a set of conditions regarding the identity of the neurons,
    considering the whole network.\n
    Fixed probability used.\n\n
    Ex: conditions = [ [ 'eq', 'ipsi', 'ax', 'ex', 'ax', 'ex' ],
    [ 'eq', 'ipsi', 'ax', 'ex', 'ax', 'in' ] ]\n
        connect_byidentity(self.S_E, conditions, 0.20)
    '''

    cond = concond_byidentity(condlist, extraconditions)
    syn.connect(condition=cond, p=prob, skip_if_invalid=True)
    return

def connectivity_define(syn_ex: b2.Synapses, syn_in: b2.Synapses)-> None:
    ''' Define the connectivity (NOTE: NETWORK 1 in Knusel 2013)'''

    extracond_up_2 = '( y_pre == y_post + 2 )'
    extracond_up_1 = '( y_pre == y_post + 1 )'
    extracond_eq_0 = '( y_pre == y_post - 0 )'
    extracond_dw_1 = '( y_pre == y_post - 1 )'
    extracond_dw_2 = '( y_pre == y_post - 2 )'

    # AXIAL NETWORK
    # Excitatory
    condlist_axe_axe = [ [ '', 'ipsi', 'ax', 'ex', 'ax', 'ex'] ]
    condlist_axe_axi = [ [ '', 'ipsi', 'ax', 'ex', 'ax', 'in'] ]
    connect_byidentity( syn_ex, condlist_axe_axe,  0.12, extraconditions= extracond_eq_0 )
    connect_byidentity( syn_ex, condlist_axe_axe,  0.10, extraconditions= extracond_dw_1 )
    connect_byidentity( syn_ex, condlist_axe_axe,  0.05, extraconditions= extracond_dw_2 )
    connect_byidentity( syn_ex, condlist_axe_axi,  0.20, extraconditions= extracond_eq_0 )

    # Inhibitory
    condlist_axi_axa = [ [ '', 'contra', 'ax', 'in', 'ax', ['ex','in'] ] ]
    connect_byidentity( syn_in, condlist_axi_axa, 0.45, extraconditions= extracond_eq_0 )
    connect_byidentity( syn_in, condlist_axi_axa, 0.15, extraconditions= extracond_dw_1 )
    connect_byidentity( syn_in, condlist_axi_axa, 0.10, extraconditions= extracond_dw_2 )

    # LIMB-AXIS
    # Excitatory
    condlist_lbe_axa = [ [ '', 'ipsi', 'lb', 'ex', 'ax', ['ex','in'] ] ]
    connect_byidentity( syn_ex, condlist_lbe_axa,  0.80, extraconditions= extracond_eq_0 )
    connect_byidentity( syn_ex, condlist_lbe_axa,  0.50, extraconditions= extracond_dw_1 )

    # Inhibitory
    condlist_lbi_axa = [ [ '', 'contra', 'lb', 'in', 'ax', ['ex','in'] ] ]
    connect_byidentity( syn_in, condlist_lbi_axa,  0.80, extraconditions= extracond_eq_0 )
    connect_byidentity( syn_in, condlist_lbi_axa,  0.50, extraconditions= extracond_dw_1 )

    # INTRA-LIMB
    # Excitatory
    condlist_lbe_lbe = [ [ '', 'ipsi', 'lb', 'ex', 'lb', 'ex'] ]
    condlist_lbe_lbi = [ [ '', 'ipsi', 'lb', 'ex', 'lb', 'in'] ]
    connect_byidentity( syn_ex, condlist_lbe_lbe,  0.12, extraconditions= extracond_eq_0 )
    connect_byidentity( syn_ex, condlist_lbe_lbi,  0.20, extraconditions= extracond_eq_0 )

    # Inhibitory
    condlist_lbi_lba = [ [ '', 'contra', 'lb', 'in', 'lb', ['ex','in'] ] ]
    connect_byidentity( syn_in, condlist_lbi_lba, 0.22, extraconditions= extracond_eq_0 )

def connectivity_from_file(
    syn_ex: b2.Synapses,
    syn_in: b2.Synapses,
    wmat_file: str) -> None:
    '''
    Defines the connectivity based on input weight matrix
    '''

    # Load weight matrix
    wmat = np.load(wmat_file)

    # Excitatory connections
    wmat_ex = np.zeros(wmat.shape)
    wmat_ex[wmat>0] = 1

    # Inhibitory connections
    wmat_in = np.zeros(wmat.shape)
    wmat_in[wmat<0] = 1

    # Connect
    ex_sources, ex_targets = np.nonzero(wmat_ex)
    syn_ex.connect(i=ex_sources, j=ex_targets)

    in_sources, in_targets = np.nonzero(wmat_in)
    syn_in.connect(i=in_sources, j=in_targets)

def connectivity_to_file(wmat_file, wmat):
    ''' Saves connectivity matrix to specified file '''
    np.save(wmat_file, wmat)

def get_connectivity_matrix(n_tot, syn_ex: b2.Synapses, syn_in: b2.Synapses) -> np.ndarray:
    ''' Create a sparse matrix to store the connections '''
    wmat = np.zeros((n_tot, n_tot))
    wmat[syn_ex.i[:], syn_ex.j[:]] = + 1
    wmat[syn_in.i[:], syn_in.j[:]] = - 1
    return wmat
