import networkx as nx
import matplotlib.pyplot as plt
from mpi4py import MPI
from enum import Enum
from random import sample
from sys import argv
import os
import math

class State(Enum):
    active = 1
    passive = 2
    leader = 3
    loser = 4

class Error(Enum):
    invalid_number_of_arguments = "invalid number of arguments \nformat of command: mpiexec -n {num_of_processes} python network.py [random | from_file {/path/to/file}]"
    invalid_number_of_communicators = "invalid number of communicators"
    invalid_mode = "invalid mode \nmode must be either random or from_file"
    file_not_found = "input file doesn't exist"
    ci_is_not_an_int = "%s is not an int"
    ci_duplication = "%s is already exists in the network"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
number_of_communicators = comm.Get_size() - 1

ALGORITHM_TAG = 1
DRAWING_TAG = 2
DELAY_TIME = 0.5

def generate_edges(vxs):
    edges = []
    
    for i in range(number_of_communicators - 1):
        edges.append((vxs[i], vxs[i + 1]))
    edges.append((vxs[number_of_communicators - 1], vxs[0]))
    
    return edges

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def finish(msg, *args):
    print(msg.value % args, flush=True)
    comm.Abort()
    exit(1)

def get_pos():
    pos = {}
    for i in range(len(vxs)):
        v = vxs[i]
        arc = 2 * math.pi / len(vxs) * i
        radius = 20
        x = math.sin(arc) * radius
        y = math.cos(arc) * radius
        pos[vxs[i]] = (x, y)
    return pos

def draw_network(vxs,pos):
    plt.clf()

    network = nx.DiGraph()
    network.add_edges_from(generate_edges(vxs))

    nx.draw_networkx_nodes(network, pos, node_color=node_colors, node_size = 500)
    nx.draw_networkx_labels(network, pos, labels=labels)
    nx.draw_networkx_edges(network, pos)

    plt.draw()
    plt.pause(DELAY_TIME)

def dst():
    if rank == number_of_communicators:
        return 1
    return rank + 1

def src():
    if rank == 1:
        return number_of_communicators
    return rank - 1

def send_data_to_drawer(data):
    comm.send(data, dest=0, tag=DRAWING_TAG)

def send(data):
    comm.send(data, dest=next, tag=ALGORITHM_TAG)
    send_data_to_drawer({'ci' : ci, 'state' : state})

if rank == 0:
    vxs = []

    if len(argv) < 2:
        finish(Error.invalid_number_of_arguments)

    if argv[1] == 'from_file':
        if len(argv) != 3:
            finish(Error.invalid_number_of_arguments)
        
        if not os.path.exists(argv[2]):
            finish(Error.file_not_found)

        for line in open(argv[2]):
            if not is_int(line):
                finish(Error.ci_is_not_an_int, line[:-1])

            ci = int(line)
            if ci in vxs:
                finish(Error.ci_duplication, ci)

            vxs.append(ci)

        if len(vxs) != number_of_communicators:
            finish(Error.invalid_number_of_communicators)
    elif argv[1] == 'random':
        if len(argv) != 2:
            finish(Error.invalid_number_of_arguments)

        vxs = sample(range(1, 100), number_of_communicators)
    else:
        finish(Error.invalid_mode)

    comm.bcast(vxs, root=0)
    
    passive_node_color = (0.9, 0, 0)
    active_node_color = (1, 1, 0.7)
    node_colors = [active_node_color for v in vxs]
    pos = get_pos()

    labels = {v: v for v in vxs}
    is_running = True

    while is_running:
        draw_network(vxs,pos)

        for i in range(number_of_communicators):
            j = i + 1
            data = comm.recv(source=j, tag=DRAWING_TAG)
            
            if 'need_stop' in data:
                is_running = False
            else:
                assert('state' in data)
                assert('ci' in data)

                if data['state'] == State.passive:
                    node_colors[i] = passive_node_color
                labels[vxs[i]] = data['ci']

    draw_network(vxs,pos)
    plt.show()

else:
    prev = src()
    next = dst()
    acn = -1
    win = -1
    state = State.active

    vxs = comm.bcast(None, root=0)
    ci = vxs[rank - 1]

    while win == -1:
        if state == State.active:
            send({'one' : ci})

            data = comm.recv(source=prev, tag=ALGORITHM_TAG)
            assert('one' in data)
            acn = data['one']
            
            if acn == ci:
                send({'small' : acn})
                
                win = acn

                data = comm.recv(source=prev, tag=ALGORITHM_TAG)
                assert('small' in data)
            else:
                send({'two' : acn})
                
                data = comm.recv(source=prev, tag=ALGORITHM_TAG)
                assert('two' in data)

                if acn < ci and acn < data['two']:
                    ci = acn
                else:
                    state = State.passive
        else:
             data = comm.recv(source=prev, tag=ALGORITHM_TAG)
             if 'small' in data:
                 win = data['small']
             send(data)

    if win == vxs[rank - 1]:
        state = State.leader
    else:
        state = State.loser

    send_data_to_drawer({'need_stop':True})
