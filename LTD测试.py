import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
import torch

# Simulation time.
time = 500

# Create the network.
network = Network()

# Create and add input, output layers.
source_layer = Input(n=1)
hidding_layer = LIFNodes(n=1)
target_layer = LIFNodes(n=1)

network.add_layer(
    layer=source_layer, name="input"
)

network.add_layer(
    layer=hidding_layer, name="hide"
)

network.add_layer(
    layer=target_layer, name="output"
)

# Create connection between input\hidding\output layers.
connection1 = Connection(
    source=source_layer,
    target=hidding_layer,
    w=10 + 5 * torch.randn(source_layer.n, hidding_layer.n),  # Normal(0.05, 0.01) weights.
)

connection2 = Connection(
    source=hidding_layer,
    target=target_layer,
    w=10 + 2 * torch.randn(hidding_layer.n, target_layer.n),  # Normal(0.05, 0.01) weights.
)

network.add_connection(
    connection=connection1, source="input", target="hide"
)

network.add_connection(
    connection=connection2, source="hide", target="output"
)


# Create and add input and output layer monitors.
source_monitor = Monitor(
    obj=source_layer,
    state_vars=("s",),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)
hidding_monitor = Monitor(
    obj=hidding_layer,
    state_vars=("s","v"),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)
target_monitor = Monitor(
    obj=target_layer,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)

network.add_monitor(monitor=source_monitor, name="A")
network.add_monitor(monitor=hidding_monitor, name="B")
network.add_monitor(monitor=target_monitor, name="C")


# Create input spike data, where each spike is distributed according to Bernoulli(0.1).
input_data = torch.bernoulli(0.1 * torch.ones(time, source_layer.n)).byte()
inputs = {"input": input_data}

# Simulate network on input data.
network.run(inputs=inputs, time=time)

# Retrieve and plot simulation spike, voltage data from monitors.
spikes = {
    "A": source_monitor.get("s"), "B": hidding_monitor.get("s"),"C": target_monitor.get("s") 
}
voltages = {"B": hidding_monitor.get("v"),"C": target_monitor.get("v") }

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()







