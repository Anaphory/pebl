"""Classes for representing networks and functions to create/modify them."""

import re
import tempfile
import os
from copy import copy, deepcopy
from itertools import chain
from bisect import insort
from collections import deque

import pydot
import numpy as N

from pebl.util import *

try:
    import networkx as nx
    _networkx = True
except ImportError:
    _networkx = False

class Network(nx.DiGraph):
    """A network is a set of nodes and directed edges between nodes"""
    
    #
    # Public methods
    #
    def __init__(self, nodes, edges=tuple(), score=None):
        """Creates a Network.

        nodes is a list of pebl.data.Variable instances.
        edges can be:

            * a list of edge tuples
            * an adjacency matrix (as boolean numpy.ndarray instance)
            * string representation (see Network.as_string() for format)

        """
        self.add_nodes_from(nodes)
        self.nodeids = range(len(nodes))
        
        self.add_edges_from(edges)
            
        #this store the network score.
        #If None, network is not scored, otherwise this is a float
        self.score = score

    def __hash__(self):
        return hash(self.edges)

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def __eq__(self, other):
        return self.score == other.score and hash(self.edges) == hash(other.edges)

    
    def is_acyclic(self, roots=None):
        """Uses a depth-first search (dfs) to detect cycles."""

        return nx.is_directed_acyclic_graph(self)

    def copy(self):
        """Returns a copy of this network."""
        return self.copy()    
       
    def layout(self, prog="dot", args=''): 
        """Determines network layout using Graphviz's dot algorithm.

        width and height are in pixels.
        dotpath is the path to the dot application.

        The resulting node positions are saved in network.node_positions.

        """

        self.node_positions = nx.graphviz_layout(self, prog=prog, args=args)

    def as_string(self):
        #DEPRECATE
        """Returns the sparse string representation of network.

        If network has edges (2,3) and (1,2), the sparse string representation
        is "2,3;1,2".

        """

        return ";".join([",".join([str(n) for n in edge]) for edge in list(self.edges)])
       
    
    def as_dotstring(self):
        """Returns network as a dot-formatted string"""

        return self.as_pydot().to_string()

    def as_dotfile(self, filename):
        """Saves network as a dot file."""

        nx.write_dot(self, filename)

    def as_pydot(self):
        """Returns a pydot instance for the network."""

        return nx.to_pydot(self)


    def as_image(self, filename, decorator=lambda x: x, prog='dot'):
        """Creates an image (PNG format) for the newtork.

        decorator is a function that accepts a pydot graph, modifies it and
        returns it.  decorators can be used to set visual appearance of
        networks (color, style, etc).

        prog is the Graphviz program to use (default: dot).

        """
        
        g = self.as_pydot()
        g = decorator(g)
        g.write_png(filename, prog=prog)


class EdgeSet(object):
    """
    Maintains a set of edges.

    Performance characteristics:
        - Edge insertion: O(1)
        - Edge retrieval: O(n)
    
    Uses adjacency lists but exposes an adjacency matrix interface via the
    adjacency_matrix property.

    """

    def __init__(self, num_nodes=0):
        self.edges = []

    def clear(self):
        """Clear the list of edges."""
        self.edges = [] 

    def add(self, edge):
        """Add an edge to the list."""
        u, v = edge
        self.add_edge(u, v)
        
    def add_many(self, edges):
        """Add multiple edges."""
        self.add_edges_from(edges)
         
    def remove(self, edge):
        """Remove edges from edgelist.
        
        If an edge to be removed does not exist, fail silently (no exceptions).

        """
        u, v = edge
        self.remove_edge(u, v)

    def remove_many(self, edges):
        """Remove multiple edges."""
        self.remove_edges_from(edges)

    def incoming(self, node):
        """Return list of nodes (as node indices) that have an edge to given node.
        
        The returned list is sorted.
        Method is also aliased as parents().
        
        """
        return self.in_edges(node)

    def outgoing(self, node):
        """Return list of nodes (as node indices) that have an edge from given node.
        
        The returned list is sorted.
        Method is also aliased as children().

        """
        return self.out_edges(node)

    parents = incoming
    children = outgoing

    def __iter__(self):
        """Iterate through the edges in this edgelist.

        Sample usage:
        for edge in edgelist: 
            print edge

        """
        return self.edges_iter()

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self.edges)
        
    def __copy__(self):
        other = EdgeSet(self.edges[:])
        return other

    def as_tuple(self):
        return tuple(tuple(s) for s in self._outgoing)

    @extended_property
    def adjacency_matrix():
        """Set or get edges as an adjacency matrix.

        The adjacency matrix is a boolean numpy.ndarray instance.

        """

        def fget(self):
            size = len(self._outgoing)
            adjmat = N.zeros((size, size), dtype=bool)
            selfedges = list(self)
            if selfedges:
                adjmat[unzip(selfedges)] = True
            return adjmat

        def fset(self, adjmat):
            self.clear()
            for edge in zip(*N.nonzero(adjmat)):
                self.add(edge)

        return locals()

    @extended_property
    def adjacency_lists():
        """Set or get edges as two adjacency lists.

        Property returns/accepts two adjacency lists for outgoing and incoming
        edges respectively. Each adjacency list if a list of sets.

        """

        def fget(self):
            return self._outgoing, self._incoming

        def fset(self, adjlists):
            if len(adjlists) is not 2:
                raise Exception("Specify both outgoing and incoming lists.")
           
            # adjlists could be any iterable. convert to list of lists
            _outgoing, _incoming = adjlists
            self._outgoing = [list(lst) for lst in _outgoing]
            self._incoming = [list(lst) for lst in _incoming]

        return locals()

    def __contains__(self, edge):
        """Check whether an edge exists in the edgelist.

        Sample usage:
        if (4,5) in edges: 
            print "edge exists!"

        """
        src, dest = edge

        try:
            return dest in self._outgoing[src]
        except IndexError:
            return False

    def __len__(self):
        return sum(len(out) for out in self._outgoing)

#        
# Factory functions
#
def fromdata(data_):
    """Creates a network from the variables in the dataset."""
    return Network(data_.variables)   

def random_network(nodes, required_edges=[], prohibited_edges=[]):
    """Creates a random network with the given set of nodes.

    Can specify required_edges and prohibited_edges to control the resulting
    random network.  
    
    """

    def _randomize(net, density=None):
        n_nodes = len(net.nodes)
        density = density or 1.0/n_nodes
        max_attempts = 50

        for attempt in xrange(max_attempts):
            # create an random adjacency matrix with given density
            adjmat = N.random.rand(n_nodes, n_nodes)
            adjmat[adjmat >= (1.0-density)] = 1
            adjmat[adjmat < 1] = 0
            
            # add required edges
            for src,dest in required_edges:
                adjmat[src][dest] = 1

            # remove prohibited edges
            for src,dest in prohibited_edges:
                adjmat[src][dest] = 0

            # remove self-loop edges (those along the diagonal)
            adjmat = N.invert(N.identity(n_nodes).astype(bool))*adjmat
            
            # set the adjaceny matrix and check for acyclicity
            net.edges.adjacency_matrix = adjmat.astype(bool)

            if net.is_acyclic():
                return net

        # got here without finding a single acyclic network.
        # so try with a less dense network
        return _randomize(density/2)

    # -----------------------

    net = Network(nodes)
    _randomize(net)
    return net

