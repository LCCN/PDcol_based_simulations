################ Simulation Block

# Configuration values
# Paths to files  **make sure to add a / after the path file name**
path_read_txt = "" # Directory where txts are located. In case you do not have this folder
                               # you can create it or set path_read_txt = "".
multimeters_path = "multimeter_out/" # Directory where multimeters files are located.
results_path = "results_folder/" # Directory where results are located.

# Seed
seed = 1
np.random.seed(seed)

# Simulation parameters
dt=1.0 # time resolution
simtime = 1000.0 #ms

# Nest Kernel
total_time_start = timer() 
nest.ResetKernel()
nest.SetKernelStatus({'rng_seed': seed})
nest.SetKernelStatus({"local_num_threads": 4})
nest.SetKernelStatus({'print_time':True}) # Show evolution of the simulation
nest.SetKernelStatus({"resolution": dt})
nest.SetKernelStatus({"overwrite_files": True})
                      #"data_path": "",
                      #"data_prefix": ""}) # I quit this values for my simulation only

# Neuron model
neuron_dict ={"I_e": 0.0, 
        "t_ref": 2.0, 
        "E_L":-65.0 , 
        "V_th":-50.0, 
        "tau_m": 10.0,
        "V_m": -60.0+10.0*np.random.rand()}
nest.SetDefaults("iaf_psc_delta", neuron_dict)

# Model definition
n_columns = 1
n_layers = 4
neuron_types = ["exc","inh"]
thalamic_input_layers = [2,4] # this indicates the index of the layer, but refers 
                              # to the L4 and L6 layers
                              
# Poisson generator parameters
noise_to_exc_rate = 8.0
noise_to_inh_rate = noise_to_exc_rate
thal_noise_rate = 0 # noise_to_exc_rate

# Weights 
w_gen_to_net= 0.08025 #0.085 this value lets the simulation work using the weights of the original paper
w_thal_driving= 0.0 # L4 and L6, for transient stimulation (15 Hz)
w_exc_to_exc_intra = 87.8 #87.8
w_exc_to_inh_intra = 87.8 #87.8
w_inh_to_inh_intra = -351.2 #-351.2 when balanced; -200 when unbalanced
w_inh_to_exc_intra = -351.2 #-351.2 when balanced; -200 when unbalanced
w_inter=1 
k_scaling_Ed_card = 0.01 # number at the divisor of the weight matrix  (nelle simulazioni a 100)

# Delays 
Ed_speed = 5.10

# Distribution for neurons
k_scaling_P_D_neurons = 1 # number at the divisor # 77
k_scaling_P_D_edges = 1 # number at the divisor # 0.125

# Read files
# total volume of nodes (can be useful to scale the groups)
N_card_vector = np.ones(n_columns) 
#subdivision of neurons in the pools. Order: L2/3e L2/3i L4e L4i L5e L5i L6e L6i
pop_card_vector = np.loadtxt(path_read_txt + 'P_D_cardinality.txt') / k_scaling_P_D_neurons
# connections between pools of the same node
conn_intra_matrix = np.transpose(np.loadtxt(path_read_txt + 'P_D_connectivity.txt'))/k_scaling_P_D_edges
conn_Ed_card_matrix = np.int_(np.loadtxt(path_read_txt + 'inter_weights.txt')/k_scaling_Ed_card)
conn_Ed_lgth_matrix = np.loadtxt(path_read_txt + 'inter_lengths.txt') / Ed_speed
P_D_input = np.loadtxt(path_read_txt + 'P_D_input.txt')


# Create nest devices
# Poisson generators (each one sends an independent train to the specified number of "outdegree" neurons of the target population)

# Generator to excitatory neurons
noise_to_exc = nest.Create("poisson_generator")
nest.SetStatus(noise_to_exc, {"rate": noise_to_exc_rate}) # rate [spikes/s], it will generate a unique 
                                                          # spike train for each of it’s targets

# Generator to inhibitory neurons
noise_to_inh = nest.Create("poisson_generator")
nest.SetStatus(noise_to_inh, {"rate": noise_to_inh_rate})

# Thalamic noise
thal_noise = nest.Create("poisson_generator")
nest.SetStatus(thal_noise, {"rate": thal_noise_rate})

# Detectors 
mult = nest.Create("multimeter",
                params={"interval": dt,
                        "record_from": ["V_m"],
                        #"withgid": True,
                        #"to_file": True,
                        #"to_memory": False,
                        "label": "my_multimeter"})

spikedet = nest.Create('spike_recorder',
                params={#"withgid": True, 
                        #"withtime": True,
                        #"to_file": False,
                        #"to_memory": True,
                        "label": "my_spike_recorder"})

mult.record_to = "ascii"
spikedet.record_to = "memory"


# Create the dictionary of the neuron populations in each pool for each node 
# Each key is defined like "neuron_pop_ + exc or inh (excitatory or inhibitory) + column number + layer number"
  
neuron_pop_dict = {} 
for col_num in range(1,n_columns + 1): 
    for layer_num in range(1,n_layers + 1): 
        for i,neuron_type in enumerate(neuron_types):
            layer_name = 'neuron_pop_' + neuron_type + str(col_num) + '_' + str(layer_num)
            population_size = int(N_card_vector[col_num-1]*pop_card_vector[2*(layer_num - 1)+i])
            neuron_pop_dict[layer_name] = nest.Create("iaf_psc_delta", population_size)
            

# Connections
connections_time_start = timer() 
for col_num in range(1,n_columns+1): 

    # Devices connections 
    for layer_num in range(1,n_layers+1): 
        for i,neuron_type in enumerate(neuron_types):
            connection_dict = {'rule': 'fixed_total_number', 
                                   'N': int(P_D_input[0, 2*(layer_num-1)+i] * pop_card_vector[2*(layer_num-1)+i])}
            pop_name = 'neuron_pop_' + neuron_type + str(col_num) + '_' + str(layer_num)
            nest.Connect(noise_to_exc, neuron_pop_dict[pop_name], 
                         conn_spec= connection_dict, 
                         syn_spec= {"weight": w_gen_to_net})
            

        if layer_num in thalamic_input_layers: #thalamic_gen_to columns 
            pop_name = 'neuron_pop_' + neuron_type + str(col_num) + '_' + str(layer_num)
            nest.Connect(thal_noise, neuron_pop_dict[pop_name], 
                         conn_spec=connection_dict, 
                         syn_spec= {"weight": w_thal_driving})

    
    # Intra-node Connections:  (order: L2/3e L2/3i L4e L4i L5e L5i L6e L6i)
    for layer_num in range(1,n_layers+1): 
        for i, target_neuron_type in enumerate(neuron_types):
            target_pop_name = 'neuron_pop_' + target_neuron_type + str(col_num) + '_' + str(layer_num)
            target_pop = neuron_pop_dict[target_pop_name]
        
            for source_layer in range(1,n_layers+1): 
                for j, source_neuron_type in enumerate(neuron_types):
                    if source_neuron_type == "exc" and target_neuron_type == "exc":
                        weights_dict = w_exc_to_exc_intra
                    elif source_neuron_type == "inh" and target_neuron_type == "exc":
                        weights_dict = w_inh_to_exc_intra
                    elif source_neuron_type == "exc" and target_neuron_type == "inh":
                        weights_dict = w_exc_to_inh_intra
                    elif source_neuron_type == "inh" and target_neuron_type == "inh":
                        weights_dict = w_inh_to_inh_intra
                    else: 
                        raise ValueError("Neurontype error in Intra-node Connections")
                
                    source_pop_name = 'neuron_pop_' + source_neuron_type + str(col_num) + '_' + str(source_layer)
                    source_pop = neuron_pop_dict[source_pop_name]
                    
                    nest.Connect(source_pop, target_pop, 
                                conn_spec={'rule': 'fixed_total_number', 
                                            'N':int(conn_intra_matrix[layer_num, source_layer] * k_scaling_P_D)}, 
                                syn_spec={"weight": weights_dict})  
    
    # Inter-node connections: exc_to_exc connections only
    for target_col in range(1,n_columns+1):
        if target_col != col_num: # we have already intra-node connections
            for source_layer_num in range(1,n_layers+1):        
                for target_layer_num in range(1,n_layers+1):     
                    source_pop_name = 'neuron_pop_exc' + str(col_num) + '_' + str(source_layer_num)
                    target_pop_name = 'neuron_pop_exc' + str(target_col) + '_' + str(target_layer_num)
                    
                    nest.Connect(neuron_pop_dict[source_pop_name], neuron_pop_dict[target_pop_name], 
                                conn_spec={'rule': 'fixed_total_number', 'N':conn_Ed_card_matrix[layer_num, target_col-1]}, 
                                syn_spec={"weight":w_inter, "delay":conn_Ed_lgth_matrix[layer_num, target_col-1]}) 


# Net to spike detector
for column in range(1,n_columns+1):
    for layer in range(1,n_layers+1): 
        for neuron_type in neuron_types:
            nest.Connect(neuron_pop_dict['neuron_pop_' + neuron_type + str(column)  + '_' + str(layer)], spikedet)



# Multimeter to net
for column in range(1,n_columns+1): 
    for layer in range(1,n_layers+1): 
        for neuron_type in neuron_types:
            nest.Connect(mult, neuron_pop_dict['neuron_pop_' + neuron_type + str(column) + '_' + str(layer)])


connections_time_end = timer() 

# Simulation
sim_time_start = timer() 
nest.Simulate(simtime)
sim_time_end = timer()


total_time_end = timer()  
print("\nConnections execution time: ", connections_time_end - connections_time_start, "s")
print("Simulation execution time: ", sim_time_end - sim_time_start, "s")
print("Total time spent: ",total_time_end - total_time_start,"s\n")  
