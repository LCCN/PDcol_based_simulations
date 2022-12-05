import nest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer 
import os
import shutil
import scipy.io
from glob import glob

####### SIMULATION

# Paths to files  **make sure to add a / after the path file name**
path_read_txt = "/Users/marinatenhave/Downloads/code/input_files/" # Directory where txts are located. In case you do not have this folder
                               # you can create it or set path_read_txt = "".
multimeters_path = "/Users/marinatenhave/Downloads/code/multimeters_folder/" # Directory where multimeters files are located.
results_path = "/Users/marinatenhave/Downloads/code/results_folder/" # Directory where results are located.

# Seed
seed = np.random.randint(859043)
np.random.seed(seed)

# Simulation parameters
dt = 1.0 # time resolution
simtime = 1.0 #ms

# Nest Kernel
nest.ResetKernel()
nest.SetKernelStatus({'rng_seed': seed})
nest.SetKernelStatus({"local_num_threads": 4})
nest.SetKernelStatus({'print_time':True}) # Show evolution of the simulation
nest.SetKernelStatus({"resolution": dt})
nest.SetKernelStatus({"overwrite_files": True})

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
noise_to_exc_rate = 8.0 # CHANGABLE -> to 7
noise_to_inh_rate = 8.0
thal_noise_rate = 0.0 # noise_to_exc_rate

# Weights 
w_gen_to_net= 0.08025 # CHANGABLE 0.085 this value lets the simulation work using the weights of the original paper
w_thal_driving= 0.0 # L4 and L6, for transient stimulation (15 Hz)
w_exc_to_exc_intra = 87.8 #87.8
w_exc_to_inh_intra = 87.8 #87.8
w_inh_to_inh_intra = -351.2 #-351.2 when balanced; -200 when unbalanced
w_inh_to_exc_intra = -351.2 #-351.2 when balanced; -200 when unbalanced
w_inter = 1 
k_scaling_Ed_card = 0.01 # (divider) for weight matrix 

# Delays 
Ed_speed = 5.10

# Distribution for neurons
k_scaling_P_D_neurons = 1 # 1 # (divider) number previously used: 77
k_scaling_P_D_edges = 0.000125 # CHANGABLE # (divider) number previously used: 0.125

# total volume of nodes (can be useful to scale the groups)
N_card_vector = np.ones(n_columns) 
#subdivision of neurons in the pools. Order: L2/3e L2/3i L4e L4i L5e L5i L6e L6i
pop_card_vector = np.loadtxt(path_read_txt + 'P_D_cardinality.txt') / k_scaling_P_D_neurons
# connections between pools of the same node
conn_intra_matrix = np.transpose(np.loadtxt(path_read_txt + 'P_D_connectivity.txt'))/k_scaling_P_D_edges
conn_Ed_card_matrix = np.int_(np.loadtxt(path_read_txt + 'inter_weights.txt')/k_scaling_Ed_card)
conn_Ed_lgth_matrix = np.loadtxt(path_read_txt + 'inter_lengths.txt') / Ed_speed
P_D_input = np.loadtxt(path_read_txt + 'P_D_input.txt')

# Poisson generators (each one sends an independent train to the specified number of "outdegree" neurons of the target population)

# Generator to excitatory neurons
noise_to_exc = nest.Create("poisson_generator")
nest.SetStatus(noise_to_exc, {"rate": noise_to_exc_rate}) # rate [spikes/s], it will generate a unique 
                                                          # spike train for each of it’s targets
# Generator to inhibitory neurons
noise_to_inh = nest.Create("poisson_generator")
nest.SetStatus(noise_to_inh, {"rate": noise_to_inh_rate})

poisson_generators = [noise_to_exc,noise_to_inh]

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

# CREATE POPULATION DICTIONARY

# Create the dictionary of the neuron populations in each pool for each node 
# Each key is defined like "neuron_pop_ + exc or inh (excitatory or inhibitory) + column number + layer number"
  
neuron_pop_dict = {} 
for col_num in range(1,n_columns + 1): 
    for layer_num in range(1,n_layers + 1): 
        for i,neuron_type in enumerate(neuron_types):
            layer_name = 'neuron_pop_' + neuron_type + str(col_num) + '_' + str(layer_num)
            population_size = int(N_card_vector[col_num-1]*pop_card_vector[2*(layer_num - 1)+i])
            neuron_pop_dict[layer_name] = nest.Create("iaf_psc_delta", population_size)

connections_time_start = timer() 

# CONNECTIONS

for col_num in range(1,n_columns+1): 

    # Devices connections 
    for layer_num in range(1,n_layers+1): 
        for i,(neuron_type,poiss_type) in enumerate(zip(neuron_types,poisson_generators)):
            connection_dict = {'rule': 'fixed_total_number', 
                                   'N': int(P_D_input[0, 2*(layer_num-1)+i] * pop_card_vector[2*(layer_num-1)+i])}
            pop_name = 'neuron_pop_' + neuron_type + str(col_num) + '_' + str(layer_num)
            nest.Connect(poiss_type, neuron_pop_dict[pop_name], 
                         conn_spec= connection_dict, 
                         syn_spec= {"weight": w_gen_to_net})
            

        if layer_num in thalamic_input_layers: #thalamic_gen_to columns 
            pop_name = 'neuron_pop_' + neuron_type + str(col_num) + '_' + str(layer_num)
            print('type of thalamic gen connection', neuron_type)
            nest.Connect(thal_noise, neuron_pop_dict[pop_name], 
                         conn_spec=connection_dict, 
                         syn_spec= {"weight": w_thal_driving})

    
    # Intra-node Connections:  (order: L2/3e L2/3i L4e L4i L5e L5i L6e L6i)
    for layer_num in range(1,n_layers+1): # loop through all the layers #TO
        for i, target_neuron_type in enumerate(neuron_types): # loop through inhibitory and exhibitory pools
            target_pop_name = 'neuron_pop_' + target_neuron_type + str(col_num) + '_' + str(layer_num)
            target_pop = neuron_pop_dict[target_pop_name]
        
            for source_layer in range(1,n_layers+1): # loop through all the layers (connected layer) #FROM
                for j, source_neuron_type in enumerate(neuron_types):
                    
                    source_pop_name = 'neuron_pop_' + source_neuron_type + str(col_num) + '_' + str(source_layer)
                    source_pop = neuron_pop_dict[source_pop_name]
                
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

                    nest.Connect(source_pop, target_pop, 
                                conn_spec={'rule': 'fixed_total_number', 
                                           'N':int(conn_intra_matrix[2*(layer_num-1)+i, 2*(source_layer-1)+j] * k_scaling_P_D_neurons)}, 
                                syn_spec={"weight": weights_dict})  
                    print(source_pop_name, target_pop_name)

    # Inter-node connections: exc_to_exc connections only
    for target_col in range(1,n_columns+1):
        if target_col != col_num: # we have already intra-node connections
            for source_layer_num in range(1,n_layers+1):        
                for target_layer_num in range(1,n_layers+1):     
                    source_pop_name = 'neuron_pop_exc' + str(col_num) + '_' + str(source_layer_num)
                    target_pop_name = 'neuron_pop_exc' + str(target_col) + '_' + str(target_layer_num)
                    nest.Connect(neuron_pop_dict[source_pop_name], neuron_pop_dict[target_pop_name], 
                                conn_spec={'rule': 'fixed_total_number', 
                                           'N':conn_Ed_card_matrix[layer_num, target_col-1]}, 
                                syn_spec={"weight":w_inter, 
                                          "delay":conn_Ed_lgth_matrix[layer_num, target_col-1]}) 

# connections to 3inh and 4inh - have to modify for multiple columns 

neuron_pops_col_1 = ['neuron_pop_exc1_1','neuron_pop_inh1_1','neuron_pop_exc1_2', 'neuron_pop_inh1_2', 'neuron_pop_exc1_3', 'neuron_pop_inh1_3', 'neuron_pop_exc1_4', 'neuron_pop_inh1_4']

for name in neuron_pops_col_1:

    if 'exc' in name: #exc to inh
        weights_dict = w_exc_to_inh_intra
    elif 'inh' in name:
        weights_dict = w_inh_to_inh_intra

    nest.Connect(neuron_pop_dict[name], neuron_pop_dict['neuron_pop_inh1_3'], 
                                            conn_spec={'rule': 'fixed_total_number', 
                                                    'N':int(conn_intra_matrix[2*(layer_num-1)+i, 2*(source_layer-1)+j] * k_scaling_P_D_neurons)}, 
                                            syn_spec={"weight": weights_dict}) 

    nest.Connect(neuron_pop_dict[name], neuron_pop_dict['neuron_pop_inh1_4'], 
                conn_spec={'rule': 'fixed_total_number', 
                            'N':int(conn_intra_matrix[2*(layer_num-1)+i, 2*(source_layer-1)+j] * k_scaling_P_D_neurons)}, 
                syn_spec={"weight": weights_dict}) 


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
print("\nConnections execution time: ", connections_time_end - connections_time_start, "s")

# SIMULATE

sim_time_start = timer() 
nest.Simulate(simtime)
sim_time_end = timer()

print("Simulation execution time: ", sim_time_end - sim_time_start, "s") 

#### RESULTS

def create_folder(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)

def remove_folder_contents(path_name):
    for filename in os.listdir(path_name):
        file_path = os.path.join(path_name, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            continue

def plot_events_times(times,events,name,path):
    fig = plt.figure()
    plt.clf() # Clear previous plot
    plt.plot(times,events,".", markersize=2)
    plt.xlabel("time (ms)")
    plt.ylabel("Spike events")
    plt.savefig(path + "/plot_time_events_" + name + ".png")
    plt.close(fig) # for not displaying the image
    return None

def plot_barchart(data_column,layer_names,y_label,title):
    y_pos = np.arange(len(layer_names))
    plt.bar(y_pos, df[data_column], align='center', alpha=0.5, 
        color=["dodgerblue","lightblue"])
    plt.xticks(y_pos, layer_names)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    return None

# Folder to save plots and results

create_folder(results_path)
remove_folder_contents(results_path)

# Get events and times from spike detectors

dSD = nest.GetStatus(spikedet,keys="events")[0]
events = dSD["senders"]
times = dSD["times"]


# _ = plot_events_times(times,events,"joint",results_path)

# The index of neurons are fixed by nest so we need to get them to count how many 
# spikes a layer has sent. 

count_spikes = []
irregularity = [] # definition of coefficient of variation: Coefficient of variation vs. mean interspike interval curves: What do they tell us about the brain? Chris Christodoulou, Guido Bugmann
synchrony = [] # Potjans 2012
for column in range(1,n_columns+1):
    for layer_num in range(1,n_layers+1):
        for neuron_type in neuron_types:
            pop_name = "neuron_pop_" + neuron_type + str(column) + "_" + str(layer_num)
            layer_bounds = neuron_pop_dict[pop_name].global_id[0],neuron_pop_dict[pop_name].global_id[-1]
            idx = np.where((layer_bounds[1]>=events) & (events>=layer_bounds[0]))
            
            count_spikes.append(len(events[idx]))
            
            interspike_intervals = np.diff(np.sort(times[idx]))/1000
            irregularity.append(np.std(interspike_intervals)/(np.mean(interspike_intervals)))
            
            counts, bins= np.histogram(events[idx], bins = 3)
            synchrony.append(np.var(counts)/np.mean(counts))


# Rename count_dict keys
layer_names = ["L23e","L23i","L4e","L4i","L5e","L5i","L6e","L6i"]
count_dict = dict(zip(layer_names,count_spikes)) 


# Get firing rates, irregularity and synchrony
firing_rates_dict = {}
irregularity_dict = {}
synchrony_dict = {}

# this tells us the average firing rate per neuron of each layer 
# (we need to scale it by the number of neurons in each layer described in pop_card_vector)
firing_rates_dict.update((x, count_dict[x]/y * 1000 / simtime ) for x, y in zip(count_dict.keys(),pop_card_vector))
irregularity_dict.update((x, y) for x, y in zip(layer_names,irregularity))
synchrony_dict.update((x, y) for x, y in zip(layer_names,synchrony))

df = pd.DataFrame([firing_rates_dict,irregularity_dict, synchrony_dict], 
                  index = ["mean firing rate","irregularity","synchrony"]).transpose()

# Average firing rate per neuron considering all neurons
FR_avg = len(events)/( int(np.sum(pop_card_vector))) * 1000 / simtime 

        
print("\nAverage FR :", np.around(FR_avg,4)) # print in console
for key, value in firing_rates_dict.items(): # print in console
    print (f"{key} FR:", np.around(value,4))

df

# ----- TESTING BLOCK ------

# TEST
# num_connections1 = len(connection1.get('target'))
# print(num_connections1)
# connection2 = nest.GetConnections(neuron_pop_dict['neuron_pop_exc1_1'], neuron_pop_dict['neuron_pop_inh1_1'])
# num_connections2 = len(connection2.get('target'))
# print(num_connections2)

# Print overall population dictionary
print('Neuron population dictionary:')
print(neuron_pop_dict)
print('-----------------------------')
print('Intra-connections and weights')
print('-----------------------------')
for column in range(1,n_columns+1):
    print('Column number: ' + str(column))
    
    # # Devices connections: 
    # print('----Devices connections----')
    # path_normdevices_conns = results_path + 'NormDevices_Col' + str(column)
    # normdevices_conns_file = open(path_normdevices_conns, "w") 
    # normdevices_conns_matrix = np.zeros((8,8))
    # path_thalamicdevices_conns = results_path + 'ThalamicDevices_Col' + str(column)
    # thalamicdevices_conns_file = open(path_normdevices_conns, "w") 
    # thalamicdevices_conns_matrix = np.zeros(8)
    # for layer_num in range(1,n_layers+1): 
    #     # for i,(neuron_type,poiss_type) in enumerate(zip(neuron_types,poisson_generators)):
    #     #     print('----Normal Devices----')
    #     #     target_pop_name = 'neuron_pop_' + neuron_type + str(col_num) + '_' + str(layer_num)
    #     #     print('hi')
    #     #     target_pop = neuron_pop_dict[source_pop_name]
    #     #     connection = nest.GetConnections(poiss_type, target_pop)
    #     #     print('hi')
    #     #     num_connections = len(connection.get('source'))
    #     #     normdevices_conns_matrix[2*(layer_num-1)+i] = num_connections
    #     #     print('hi')
    #     #     print(str(layer_num) + str(neuron_type) + ': ' + str(num_connections))

    #     # if layer_num in thalamic_input_layers: #thalamic_gen_to columns 
    #     #     print('----Thalamic Devices----')
    #     #     target_pop_name = 'neuron_pop_' + neuron_type + str(col_num) + '_' + str(layer_num)
    #     #     target_pop = neuron_pop_dict[source_pop_name]
    #     #     print('helloooouuu')
    #     #     connection = nest.GetConnections(thal_noise, target_pop)
    #     #     num_connections = len(connection.get('source'))
    #     #     normdevices_conns_matrix[2*(layer_num-1)+1] = num_connections
    #     #     print(str(layer_num) + str(neuron_type) + ': ' + str(num_connections))

    # Intra-layer connections: Print the number and weight of connections between all the pools 
    print('----Intra-layer connections----')
    path_weights = results_path + 'Weights_Col' + str(column)
    path_num_connections = results_path + 'Connections#_Col' + str(column)
    weights_file = open(path_weights, "w") 
    num_connections_file = open(path_num_connections, "w") 
    weights_matrix = np.zeros((8,8))
    num_connections_matrix = np.zeros((8,8))
    for source_layer in range(1,n_layers+1): # loop through all the layers #FROM
        for i, source_neuron_type in enumerate(neuron_types): # loop through inhibitory and exhibitory pools
            print('-------')
            print('Source layer: ' + str(source_layer) + str(source_neuron_type))
            source_pop_name = 'neuron_pop_' + source_neuron_type + str(col_num) + '_' + str(source_layer)
            source_pop = neuron_pop_dict[source_pop_name]
            for target_layer in range(1,n_layers+1): # loop through all the layers (connected layer) #TO
                for j, target_neuron_type in enumerate(neuron_types):
                    target_pop_name = 'neuron_pop_' + target_neuron_type + str(col_num) + '_' + str(target_layer)
                    target_pop = neuron_pop_dict[target_pop_name]
                    connection = nest.GetConnections(source_pop, target_pop)
                    num_connections = len(connection.get('source'))
                    weight = connection.get('weight')[0]
                    print('To ' + str(target_layer) + str(target_neuron_type) + ' : ' + 'Connection number: ' + str(num_connections) + ' Weight: ' + str(weight))
                    # TBD: use numpy and add these respective numbers to their appropriate coordinates
                    weights_matrix[(2*(source_layer-1)+i, 2*(target_layer-1)+j)] = weight # vertical/rows are source_pop, horizontal/columns are target_pop 
                    num_connections_matrix[(2*(source_layer-1)+i, 2*(target_layer-1)+j)] = num_connections
    # Save matricies
    np.savetxt(path_weights, weights_matrix, fmt="%f")
    np.savetxt(path_num_connections, num_connections_matrix, fmt="%f")
    # np.savetxt(path_normdevices_conns, normdevices_conns_matrix, fmt="%f")
    # np.savetxt(path_thalamicdevices_conns, thalamicdevices_conns_matrix, fmt="%f")

# Plot barchart for firing rates

plot_barchart("mean firing rate",layer_names,'firing rate [Hz]','Mean firing rates')

plot_barchart("irregularity",layer_names,'irregularity',"")

plot_barchart("synchrony",layer_names,'synchrony',"")

# Extract signals

