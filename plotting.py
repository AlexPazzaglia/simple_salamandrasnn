import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

# PLOTTING
def plot_results(
    statemon      : b2.StateMonitor,
    spikemon      : b2.SpikeMonitor,
    monitored_vars: tuple[str],
    wmat          : np.ndarray,
    segments_tot  : int,
    n_e_semi      : int,
    n_i_semi      : int,
    limb_positions: tuple[int],
    pools_indices : dict[str, tuple[tuple[int]]],
):
    ''' Plot the results '''

    # Neuronal values
    if len(monitored_vars) > 0:
        plt.figure('Internal variables')
        plot_internal_variables(
            statemon,
            monitored_vars,
        )

    # Spikes
    plt.figure('Raster plot')
    plot_raster_plot(
        spikemon,
        segments_tot,
        n_e_semi,
        n_i_semi,
        limb_positions,
        pools_indices,
    )

    # Connectivity matrix
    plt.figure('Connectivity matrix')
    plot_connectivity_matrix(
        wmat,
        segments_tot,
        n_e_semi,
        n_i_semi,
        limb_positions
    )
    plt.show()

def plot_internal_variables(
    statemon      : b2.StateMonitor,
    monitored_vars: tuple[str]
):
    ''' Evolution of neuronal variables '''
    n_vars = len(monitored_vars)

    for i, var in enumerate(monitored_vars):
        var_evolution = getattr(statemon, var)
        plt.subplot(n_vars, 1, i+1)
        plt.plot(statemon.t/b2.ms, var_evolution.T)
        plt.ylabel(var.upper())

    plt.xlabel('Time [ms]')

def plot_raster_plot(
    spikemon      : b2.SpikeMonitor,
    segments_tot  : int,
    n_e_semi      : int,
    n_i_semi      : int,
    limb_positions: tuple[int],
    pools_indices : dict[str, tuple[tuple[int]]],

):
    ''' Spiking activity '''

    spike_trains = spikemon.spike_trains()

    inds_el = np.array( pools_indices['el'] ).flatten()
    inds_er = np.array( pools_indices['er'] ).flatten()
    inds_il = np.array( pools_indices['il'] ).flatten()
    inds_ir = np.array( pools_indices['ir'] ).flatten()

    n_el = len(inds_el)
    n_er = len(inds_er)

    # Plot the left excitatory neurons
    for id, ner_ind in enumerate(inds_el):
        train = spike_trains[ner_ind] / b2.ms
        plt.scatter(
            train,
            - id * np.ones_like(train),
            color     = 'r',
            marker    = '.',
            s = 0.1,
        )

    # Plot the left inhibitory neurons
    for id, ner_ind in enumerate(inds_il):
        train = spike_trains[ner_ind] / b2.ms
        plt.scatter(
            train,
            - id * np.ones_like(train) - n_el,
            color     = 'b',
            marker    = '.',
            s = 0.1,
        )


    # Plot the right excitatory neurons
    for id, ner_ind in enumerate(inds_er):
        train = spike_trains[ner_ind] / b2.ms
        plt.scatter(
            train,
            + id * np.ones_like(train),
            color     = 'r',
            marker    = '.',
            s = 0.1,
        )

    # Plot the right inhibitory neurons
    for id, ner_ind in enumerate(inds_ir):
        train = spike_trains[ner_ind] / b2.ms
        plt.scatter(
            train,
            + id * np.ones_like(train) + n_er,
            color     = 'b',
            marker    = '.',
            s = 0.1,
        )


    duration = spikemon.t[-1]/b2.ms
    n_tot    = 2 * (n_e_semi + n_i_semi) * segments_tot

    plt.xlim(0, duration)
    plt.ylim(-n_tot/2, +n_tot/2)
    plt.hlines(0, 0, duration, color='k', linewidth=0.5)

    plt.xlabel('Time [ms]')
    plt.ylabel('Neuron index')
    plt.title('Raster plot')

def plot_connectivity_matrix(
    w_syn         : np.ndarray,
    segments_tot  : int,
    n_e_semi      : int,
    n_i_semi      : int,
    limb_positions: list[int]
) -> None:
    '''
    Connectivity matrix showing the links in the network.
    '''

    segments_limbs = len(limb_positions)
    segments_axial = segments_tot - segments_limbs

    n_hemiseg = n_e_semi + n_i_semi
    n_seg = 2 * n_hemiseg
    n_axial = segments_axial * n_seg
    n_limbs = segments_limbs * n_seg
    n_tot = n_axial + n_limbs

    # Separate pools
    plot_grid(segments_tot, n_e_semi, n_i_semi, limb_positions, 'h', n_tot)
    plot_grid(segments_tot, n_e_semi, n_i_semi, limb_positions, 'v', n_tot)

    # Decorate plot
    locs, labs = get_plot_labels(segments_tot, n_e_semi, n_i_semi, limb_positions)

    plt.yticks(locs, labs)
    plt.xticks(locs, labs, rotation = 90)

    plt.title('Connectivity matrix')
    plt.xlabel('Pre-synaptic nurons')
    plt.ylabel('Post-synaptic neurons')

    plt.xlim(0,n_tot - 0.5)
    plt.ylim(0,n_tot - 0.5)

    plt.imshow(w_syn.T, cmap = 'seismic') #origin ='upper')
    return

def plot_grid(
    segments_tot  : int,
    n_e_semi      : int,
    n_i_semi      : int,
    limb_positions: list[int],
    orientation   : str,
    limit         : int,
):
    ''' Grid to separate pools '''
    segments_limbs = len(limb_positions)
    segments_axial = segments_tot - segments_limbs

    n_seg = 2 * ( n_e_semi + n_i_semi )
    n_axial = segments_axial * n_seg

    if orientation == 'v':
        plt.plot([n_axial, n_axial], [0,limit], color= 'k', linewidth= 0.5)

        # Limbs
        for lmb_ind in range(segments_limbs):
            lmb_ex = n_axial + lmb_ind * n_seg + 2 * n_e_semi
            lmb_in = n_axial + lmb_ind * n_seg + n_seg

            plt.plot([lmb_ex, lmb_ex], [0,limit], c='k', lw= 0.5, ls='--')
            plt.plot([lmb_in, lmb_in], [0,limit], c='k', lw= 0.5, ls='--')

    if orientation == 'h':
        plt.plot([0,limit], [n_axial, n_axial], color= 'k', linewidth= 0.5)    # Between axis and limbs

        # Separate limbs
        for lmb_ind in range(segments_limbs):
            lmb_ex = n_axial + lmb_ind * n_seg + 2 * n_e_semi
            lmb_in = n_axial + lmb_ind * n_seg + n_seg

            plt.plot([0,limit], [lmb_ex, lmb_ex], c='k', lw= 0.5, ls='--')
            plt.plot([0,limit], [lmb_in, lmb_in], c='k', lw= 0.5, ls='--')

def get_plot_labels(
    segments_tot  : int,
    n_e_semi      : int,
    n_i_semi      : int,
    limb_positions: list[int],
):
    ''' Labels for the plots '''

    segments_limbs = len(limb_positions)
    segments_axial = segments_tot - segments_limbs

    n_hemiseg = n_e_semi + n_i_semi
    n_seg = 2 * n_hemiseg
    n_axial = segments_axial * n_seg

    # Locations and Labels
    locs_ax = [ seg_ind * n_seg + n_seg // 2 for seg_ind in range(segments_axial) ]
    labs_ax = [ f'AX_{seg_ind}' for seg_ind in range(segments_axial)]
    locs_lb = [ n_axial + lb_ind * n_seg + n_seg // 2 for lb_ind in range(segments_limbs) ]
    labs_lb = [ f'LB_{1 + seg_ind}_{2 + seg_ind}' for seg_ind in range(segments_limbs)]

    locs = locs_ax + locs_lb
    labs = labs_ax + labs_lb

    return locs, labs
