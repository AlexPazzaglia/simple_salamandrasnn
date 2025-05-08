'''
Example program to run the model proposed by Knusel et al. 2013 with Brian2.
Alessandro Pazzaglia
30/06/2022
'''
import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import logging
import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

import equations
import parameters
import connections
import plotting
import time

class SimulationSNN():

    def __init__(
        self,
        load_connectivity : bool,
        save_connectivity : bool,
        connectivity_file : str,
        sim_dt            : float,
        duration          : float,
        drive_current     : float,
        n_ex_hemiseg      : int,
        n_in_hemiseg      : int,
        seg_axial         : int,
        limbs_positions   : int,
        state_monitor     : bool,
        include_callback  : bool,
        callback_dt       : float,
    ):

        # Connections
        self.load_connectivity = load_connectivity
        self.save_connectivity = save_connectivity
        self.connectivity_file = connectivity_file

        # Simulation
        self.sim_dt        = sim_dt
        self.duration      = duration
        self.state_monitor = state_monitor

        # Drive
        self.drive_current    = drive_current

        # Topology
        self.n_ex_hemiseg     = n_ex_hemiseg
        self.n_in_hemiseg     = n_in_hemiseg
        self.seg_axial        = seg_axial
        self.limbs_positions  = limbs_positions

        # Callback
        self.include_callback = include_callback
        self.callback_dt      = callback_dt

        # Complete network definition
        self.define_topology()
        self.define_parameters()
        self.define_equations()
        self.define_neuron_population()
        self.define_synaptic_populations()
        self.define_connectivity()
        self.define_monitors()
        self.initialize_network()
        self.define_callback_function()

    def define_topology(self):
        ''' Define the network topology '''

        self.n_ex_seg = self.n_ex_hemiseg * 2
        self.n_in_seg = self.n_in_hemiseg * 2

        self.n_hemiseg = self.n_ex_hemiseg + self.n_in_hemiseg
        self.n_seg     = self.n_hemiseg * 2

        self.seg_limbs = len(self.limbs_positions)
        self.seg_tot = self.seg_axial + self.seg_limbs

        self.n_axial = self.n_seg * self.seg_axial
        self.n_limbs = self.n_seg * self.seg_limbs
        self.n_tot = self.n_seg * self.seg_tot

        self.pools_indices = {
            'el': [[x + y * self.n_seg                                     for x in range(self.n_ex_hemiseg)] for y in range(self.seg_tot)],
            'er': [[x + y * self.n_seg + self.n_ex_hemiseg                 for x in range(self.n_ex_hemiseg)] for y in range(self.seg_tot)],
            'il': [[x + y * self.n_seg + self.n_ex_seg                     for x in range(self.n_in_hemiseg)] for y in range(self.seg_tot)],
            'ir': [[x + y * self.n_seg + self.n_ex_seg + self.n_in_hemiseg for x in range(self.n_in_hemiseg)] for y in range(self.seg_tot)],
        }

    def define_parameters(self):
        ''' Define the network parameters '''
        (
            self.shared_neural_params,
            self.variable_neural_params_axs,
            self.variable_neural_params_lmb,
            self.shared_syn_ex_params,
            self.shared_syn_in_params,
        ) = parameters.network_parameters()

    def define_equations(self):
        ''' Define the network equations '''
        (
            self.neuron_eqs,
            self.reset_eqs,
            self.threshold_eqs,
        )= equations.neural_equations()
        (
            self.syn_ex_eq,
            self.syn_ex_on_pre,
            self.syn_in_eq,
            self.syn_in_on_pre,
        ) = equations.synaptic_equations()

    def define_neuron_population(self):
        ''' Define the neuron populations '''

        # Define the neuron population
        self.pop = b2.NeuronGroup(
            N          = self.n_tot,
            model      = self.neuron_eqs,
            threshold  = self.threshold_eqs,
            reset      = self.reset_eqs,
            refractory = 5 * b2.ms,
            method     = 'euler'
        )

        # Define the neuronal identifiers
        parameters.set_neural_identifiers(
            self.pop,
            self.pools_indices,
            self.limbs_positions
        )

        # Assign neuronal parameters
        syn_params   = self.shared_syn_ex_params | self.shared_syn_in_params
        axial_params = self.shared_neural_params | self.variable_neural_params_axs | syn_params
        limbs_params = self.shared_neural_params | self.variable_neural_params_lmb | syn_params

        # Axial neurons
        parameters.set_neural_parameters(
            pool       = self.pop,
            ids        = range(0, self.n_axial),
            std_value  = 0,
            parameters = axial_params,
        )

        # Limbs neurons
        parameters.set_neural_parameters(
            pool       = self.pop,
            ids        = range(self.n_axial, self.n_tot),
            std_value  = 0,
            parameters = limbs_params,
        )

    def define_synaptic_populations(self):
        ''' Define the synaptic populations '''

        self.syn_ex = b2.Synapses(
            source = self.pop,              # Origin
            target = self.pop,              # Target
            model  = self.syn_ex_eq,        # Internal behavior
            on_pre = self.syn_ex_on_pre,    # Effect of a spike
            delay  = 5*b2.ms,               # Synaptic delay
            method = 'euler'                # Integration method
        )

        self.syn_in = b2.Synapses(
            source = self.pop,              # Origin
            target = self.pop,              # Target
            model  = self.syn_in_eq,        # Internal behavior
            on_pre = self.syn_in_on_pre,    # Effect of a spike
            delay = 5*b2.ms,                # Synaptic delay
            method = 'euler'                # Integration method
        )

    def define_connectivity(self):
        ''' Define the network connectivity '''

        if self.load_connectivity:
            logging.info('Loading connectivity matrix from %s', self.connectivity_file)
            connections.connectivity_from_file(self.syn_ex, self.syn_in, self.connectivity_file)
        else:
            logging.info('Defining connectivity matrix')
            connections.connectivity_define(self.syn_ex, self.syn_in)

        # Save connectivity
        self.wmat = connections.get_connectivity_matrix(self.n_tot, self.syn_ex, self.syn_in)

        if self.save_connectivity:
            logging.info('Saving connectivity matrix to %s', self.connectivity_file)
            connections.connectivity_to_file(self.connectivity_file, self.wmat)

    def define_monitors(self):
        ''' Define the objects to monitor the variables '''

        if self.state_monitor:
            self.monitored_vars = ('v','w1','I_ex','I_in')
        else:
            self.monitored_vars = ()

        self.spikemon = b2.SpikeMonitor(self.pop)                                      # Record spike times
        self.statemon = b2.StateMonitor(self.pop, self.monitored_vars, record=True)    # Record variables

    def initialize_network(self):
        ''' Initialize network variables '''

        # Initialize variables
        self.pop.v = self.shared_neural_params['V_rest'][0] * b2.mV
        self.pop.w1 = 0 * b2.nA

        # Assign drive current
        pools_el = self.pools_indices['el']
        pools_er = self.pools_indices['er']
        pools_il = self.pools_indices['il']
        pools_ir = self.pools_indices['ir']

        l_mult = 1.00
        r_mult = 1.00

        for seg in range(self.seg_axial + self.seg_limbs):
            self.pop.I1[ pools_el[seg] ] = self.drive_current * l_mult
            self.pop.I1[ pools_er[seg] ] = self.drive_current * r_mult
            self.pop.I1[ pools_il[seg] ] = self.drive_current * l_mult
            self.pop.I1[ pools_ir[seg] ] = self.drive_current * r_mult

        # Create network object
        self.network = b2.Network(self.pop, self.syn_ex, self.syn_in, self.statemon, self.spikemon)
        self.network.schedule = ['start', 'thresholds', 'resets', 'synapses', 'groups', 'end']

    ## CALLBACKS
    def step_function(self, _curtime):
        ''' Step function '''
        print(f'I am called at {_curtime}')

    def define_callback_function(self):
        '''
        Define the callback function
        When used (include_callback = True), called at every integration step.\n
        '''

        if not self.include_callback:
            return

        self.callback_clock = b2.Clock(
            dt   = self.callback_dt,
            name = 'callback_clock'
        )
        self.callback = b2.NetworkOperation(
            function = self.step_function,
            clock    = self.callback_clock,
            name     = 'callback',
        )
        self.network.add(self.callback)

    ## PLOTTING
    def plot_results(self):
        ''' Plot the results '''
        plotting.plot_results(
            self.statemon,
            self.spikemon,
            self.monitored_vars,
            self.wmat,
            self.seg_tot,
            self.n_ex_hemiseg,
            self.n_in_hemiseg,
            self.limbs_positions,
            self.pools_indices,
        )

    ## SIMULATION
    def simulate(self):
        ''' Network simulation '''
        logging.info('Running simulation')
        b2.defaultclock.dt = self.sim_dt
        self.network.run(self.duration)

if __name__ == '__main__':

    # Simulation parameters
    LOAD_CONNECTIVITY = False
    SAVE_CONNECTIVITY = False
    CONNECTIVITY_FILE = 'network_connectivity.npy'

    TIMESTEP = 1 * b2.ms
    DURATION = 10 * b2.second

    DRIVE_CURRENT = 4.0 * b2.nA

    N_EX_HEMISEG = 25
    N_IN_HEMISEG = 20

    SEG_AXIAL = 16          # 16
    LIMBS_POSITIONS = [0, 8]    # [0,8]

    STATE_MONITOR = False

    INCLUDE_CALLBACK = True
    CALLBACK_DT      = 50 * b2.ms

    # Logging
    logging.basicConfig(
        format  = '%(asctime)s - %(levelname)s - %(name)s : %(message)s',
        datefmt = '%d-%b-%y %H:%M:%S',
        level   = logging.INFO
    )

    # Define simulation
    simulation = SimulationSNN(
        load_connectivity = LOAD_CONNECTIVITY,
        save_connectivity = SAVE_CONNECTIVITY,
        connectivity_file = CONNECTIVITY_FILE,
        sim_dt            = TIMESTEP,
        duration          = DURATION,
        drive_current     = DRIVE_CURRENT,
        n_ex_hemiseg      = N_EX_HEMISEG,
        n_in_hemiseg      = N_IN_HEMISEG,
        seg_axial         = SEG_AXIAL,
        limbs_positions   = LIMBS_POSITIONS,
        state_monitor     = STATE_MONITOR,
        include_callback  = INCLUDE_CALLBACK,
        callback_dt       = CALLBACK_DT,
    )

    # Run simulation
    start_time = time.time()
    simulation.simulate()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    real_time_fraction = float( DURATION / (end_time - start_time) ) * 100
    print(f"Real time fraction: {real_time_fraction:.2f} %")


    # Plot results
    simulation.plot_results()




