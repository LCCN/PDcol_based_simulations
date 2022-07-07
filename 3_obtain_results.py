################ Obtain_results

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
    plt.clf() # Clear previous plot
    plt.plot(times,events,".", markersize=2)
    plt.xlabel("time (ms)")
    plt.ylabel("Spike events")
    plt.savefig(path + "/plot_time_events_" + name + ".png")
    return None
    
        

# Create a folder where save plots and results
create_folder(results_path)
# If you do not want to erase the previous contents of the folder comment this line
#remove_folder_contents(results_path)

# Create a plot of time vs spikes
dSD = nest.GetStatus(spikedet,keys="events")[0]
events = dSD["senders"]
times = dSD["times"]
plot_events_times(times,events,"joint",results_path)

# We are interested in knowing the number of spikes in same types of layers
count_dict = {"layer_1_exc":0,
              "layer_1_inh":0,
              "layer_2_exc":0,
              "layer_2_inh":0,
              "layer_3_exc":0,
              "layer_3_inh":0,
              "layer_4_exc":0,
              "layer_4_inh":0,
              }

# The index of neurons are fixed by nest so we need to get them to count how many 
# spikes a layer has sent. Also we save an events vs times plot for each column.

for column in range(1,n_columns+1):
    bounds = []
    for layer_num in range(1,n_layers+1):
        for neuron_type in neuron_types:
            pop_name = "neuron_pop_" + neuron_type + str(column) + "_" + str(layer_num)
            ids = neuron_pop_dict[pop_name].global_id[0],neuron_pop_dict[pop_name].global_id[-1]
            count_dict["layer_" + str(layer_num) + "_" + neuron_type] += sum(map(lambda x : ids[1]>=x>=ids[0], events))
            bounds.append(ids)
            
    idx = np.where((events >= bounds[0][0]) & (events <= bounds[-1][1]))  
    # Take a plot for the activity of each column
    plot_events_times(times[idx],events[idx],"col_" + str(column),results_path)


# Rename count_dict keys
count_dict = dict(zip(["n23e","L23i","L4e","L4i","L5e","L5i","L6e","L6i"],list(count_dict.values()))) 


# Get firing rates
firing_rates_dict = {}
# this tells us the average fir_rate per neuron of each layer. 
# I divide by n_columns because we are counting all neurons of each layer in every column
firing_rates_dict.update((x, count_dict[x]/(y*n_columns*simtime/(1000))) for x, y in zip(count_dict.keys(),pop_card_vector))
# this tells us the average firing rate per neuron considering all neurons
FR_avg=len(events)/(int(sum(count_dict.values())*k_scaling_P_D)*simtime/1000) 

# Print firing rates
with open(results_path + '/firing_rates.txt', 'w') as f: # print in a file
    print("The average firing rate is", np.around(FR_avg,3), file=f)
    for key, value in firing_rates_dict.items():
        print (f"The {key} firing rate is", np.around(value,3),file = f)
        
# print("The average firing rate is", np.around(FR_avg,2)) # print in console
#for key, value in firing_rates_dict.items(): # print in console
    # print (f"The {key} firing rate is", np.around(value,2))
