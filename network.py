import math
import os
import sys
from enum import Enum
from itertools import tee
from random import sample

import matplotlib.pyplot as plt
import networkx as nx
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
number_of_communicators = comm.Get_size() - 1

ALGORITHM_TAG = 1
DRAWING_TAG = 2
DELAY_TIME = 0.5


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


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)

    return zip(a, b)


def generate_edges(vxs):
    edges = [(v1,v2) for v1, v2 in pairwise(vxs)]
    edges.append((vxs[number_of_communicators - 1], vxs[0]))
    
    return edges


def finish(msg, *args):
    print(msg.value % args, flush=True)
    comm.Abort()
    exit(1)


def get_pos(vxs):
    pos = {}
    radius = 20
    
    for i, v in enumerate(vxs):
        arc = 2 * math.pi * i / len(vxs)
        x = math.sin(arc) * radius
        y = math.cos(arc) * radius
        pos[v] = (x, y)

    return pos


def draw_network(vxs, pos, node_colors, labels):
    plt.clf()

    network = nx.DiGraph()
    network.add_edges_from(generate_edges(vxs))

    nx.draw_networkx_nodes(network, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(network, pos, labels=labels)
    nx.draw_networkx_edges(network, pos)

    plt.draw()
    plt.pause(DELAY_TIME)


def dst(rank, number_of_communicators):
    if rank == number_of_communicators:
        return 1

    return rank + 1


def src(rank, number_of_communicators):
    if rank == 1:
        return number_of_communicators

    return rank - 1


def send_data_to_drawer(data):
    comm.send(data, dest=0, tag=DRAWING_TAG)


def send(next_node, algo_data, state_data):
    comm.send(algo_data, dest=next_node, tag=ALGORITHM_TAG)
    send_data_to_drawer(state_data)


def get_vxs(argv):
    vxs = []

    if len(argv) < 1:
            finish(Error.invalid_number_of_arguments)
    
    if argv[0] == 'from_file':
        if len(argv) != 2:
            finish(Error.invalid_number_of_arguments)
        
        if not os.path.exists(argv[1]):
            finish(Error.file_not_found)
    
        for line in open(argv[1]):
            if not line[:-1].isdecimal():
                finish(Error.ci_is_not_an_int, line[:-1])
    
            ci = int(line)
            if ci in vxs:
                finish(Error.ci_duplication, ci)
    
            vxs.append(ci)
    
        if len(vxs) != number_of_communicators:
            finish(Error.invalid_number_of_communicators)
    elif argv[0] == 'random':
        if len(argv) != 1:
            finish(Error.invalid_number_of_arguments)
    
        vxs = sample(range(1, 100), number_of_communicators)
    else:
        finish(Error.invalid_mode)

    return vxs


def drawer_worker(argv):
    vxs = get_vxs(argv)
    comm.bcast(vxs, root=0)
    
    passive_node_color = (0.9, 0, 0)
    active_node_color = (1, 1, 0.7)
    node_colors = [active_node_color for v in vxs]
    pos = get_pos(vxs)
    
    labels = {v: v for v in vxs}
    is_running = number_of_communicators > 1
    
    while is_running:
        draw_network(vxs, pos, node_colors, labels)
    
        for i, v in enumerate(vxs):
            src_communicator = i + 1
            data = comm.recv(source=src_communicator, tag=DRAWING_TAG)
            
            if 'need_stop' in data:
                is_running = False
            else:
                assert 'state' in data
                assert 'ci' in data
    
                if data['state'] == State.passive:
                    node_colors[i] = passive_node_color
                labels[v] = data['ci']
    
    draw_network(vxs, pos, node_colors, labels)
    plt.show()
    

def communicator_worker():
    prev_node = src(rank, number_of_communicators)
    next_node = dst(rank, number_of_communicators)
    acn = -1

    vxs = comm.bcast(None, root=0)
    
    win = -1 if number_of_communicators > 1 else vxs[rank - 1]
    state_data = {'ci' : vxs[rank - 1], 'state' : State.active}
    
    while win == -1:
        if state_data['state'] == State.active:
            send(next_node, {'one' : state_data['ci']}, state_data)
    
            data = comm.recv(source=prev_node, tag=ALGORITHM_TAG)
            assert 'one' in data
            acn = data['one']
            
            if acn == state_data['ci']:
                send(next_node, {'small' : acn}, state_data)
                
                win = acn
    
                data = comm.recv(source=prev_node, tag=ALGORITHM_TAG)
                assert 'small' in data
            else:
                send(next_node, {'two' : acn}, state_data)
                
                data = comm.recv(source=prev_node, tag=ALGORITHM_TAG)
                assert 'two' in data
    
                if acn < state_data['ci'] and acn < data['two']:
                    state_data['ci'] = acn
                else:
                    state_data['state'] = State.passive
        else:
             data = comm.recv(source=prev_node, tag=ALGORITHM_TAG)
             win = data.get('small', win)
             send(next_node, data, state_data)
    
    if win == vxs[rank - 1]:
        state_data['state'] = State.leader
    else:
        state_data['state'] = State.loser
    
    send_data_to_drawer({'need_stop':True})


def simulate(argv):
    if rank == 0:
        drawer_worker(argv)
    else:
        communicator_worker()


if __name__ == '__main__':
    simulate(sys.argv[1:])
