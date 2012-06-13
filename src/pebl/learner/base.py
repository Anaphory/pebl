import numpy as N

from random import choice
from pebl import network, config, evaluator, data, prior
from pebl.taskcontroller.base import Task
import networkx as nx

#
# Exceptions
#
class CannotAlterNetworkException(Exception):
    pass

#
# Module parameters
#
_plearnertype = config.StringParameter(
    'learner.type',
    """Type of learner to use. 

    The following learners are included with pebl:
        * greedy.GreedyLearner
        * simanneal.SimulatedAnnealingLearner
        * exhaustive.ListLearner
    """,
    default = 'greedy.GreedyLearner'
)

_ptasks = config.IntParameter(
    'learner.numtasks',
    "Number of learner tasks to run.",
    config.atleast(0),
    default=1
)


class Learner(Task):
    def __init__(self, data_=None, prior_=None, whitelist=tuple(), blacklist=tuple(), **kw):
        self.data = data_ or data.fromconfig()
        self.prior = prior_ or prior.fromconfig()
        
        self.black_edges = kw.pop('blacklist', ())
        self.white_edges = kw.pop('whitelist', ())
        self.__dict__.update(kw)

        # parameters
        self.numtasks = config.get('learner.numtasks')

        # stats
        self.reverse = 0
        self.add = 0
        self.remove = 0
        
    def _alter_network_randomly_and_score(self):
        net = self.evaluator.network
        nodes = net.nodes()

        # continue making changes and undoing them till we get an acyclic network
        for i in xrange(len(nodes)**2):
            #node1, node2 = N.random.random_integers(0, n_nodes-1, 2)    
            node1 = choice(nodes)
            node2 = choice(nodes)
            
            if node1 == node2:
                continue
            
            if (node1, node2) in net.edges() or (node1, node2) in self.black_edges:
                # node1 -> node2 exists, so reverse it.    
                add,remove = [(node2, node1)], [(node1, node2)]
            elif (node2, node1) in net.edges() or (node2, node1) in self.black_edges:
                # node2 -> node1 exists, so remove it
                add,remove = [], [(node2, node1)]
            else:
                # node1 and node2 unconnected, so connect them
                add,remove =  [(node1, node2)], []
            
            try:
                score = self.evaluator.alter_network(add=add, remove=remove)
            except evaluator.CyclicNetworkError:
                continue # let's try again!
            else:
                if add and remove:
                    self.reverse += 1
                elif add:
                    self.add += 1
                else:
                    self.remove += 1
                return score

        # Could not find a valid network  
        raise CannotAlterNetworkException() 

    def _all_changes(self):
        net = self.evaluator.network
        changes = []

        # edge removals
        changes.extend((None, edge) for edge in net.edges())

        # edge reversals
        #reverse = lambda edge: (edge[1], edge[0])
        changes.extend((edge[::-1], edge) for edge in net.edges())

        # edge additions
        nz = N.nonzero(invert(nx.adjacency_matrix(net)))
        changes.extend( ((src,dest), None) for src,dest in zip(*nz) )

        return changes


