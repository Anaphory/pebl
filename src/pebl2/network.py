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

from pebl2.util import *

import networkx as nx

class NodeException(Exception): pass

class Network(nx.DiGraph):
    """A network is a set of nodes and directed edges between nodes"""
    
    #
    # Public methods
    #
    def __init__(self, nodes=(), edges=tuple(), score=None):
        """Creates a Network.

        nodes is a list of pebl.data.Variable instances.
        edges can be:

            * a list of edge tuples
            * an adjacency matrix (as boolean numpy.ndarray instance)
            * string representation (see Network.as_string() for format)

        """
        
        super(Network, self).__init__()
        #nrange = xrange(len(nodes))
        self.add_nodes_from(nodes)
        
        for idf, node in enumerate(nodes):
            self._add_id(node, idf)
        
        if isinstance(edges, N.ndarray):
            #create edges using adj mat.
            rows, cols = edges.shape
            print "Edges Size: ",edges.shape
            print "Nodes Size: ",len(nodes)
            edg = [(nodes[j],nodes[k]) for k in xrange(cols) for j in xrange(rows) if edges[j,k]]
        elif isinstance(edges, str) and edges:
            edg = [map(int, x.split(",")) for x in edges.split(";")]
        else:
            edg = edges
            
        self.add_edges_from(edg)
        
        #this store the network score.
        #If None, network is not scored, otherwise this is a float
        self.score = score

    def __hash__(self):
        return hash(tuple(self.edges()))

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def __eq__(self, other):
        return self.score == other.score and hash(self.edges) == hash(other.edges)

    def _add_id(self, node, node_id):
        """Add a string representation to a node's dictionary"""
        
        self.node[node]['id'] = node_id
    
    def add_edges_from(self, edges, attr_dict=None, **attr):
        """Add edges from [edges] to the network
        
        Will fail if nodes being connected by edges don't exist"""
        
        for edge in edges:
            self.add_edge(edge[0], edge[1], attr_dict=attr_dict, **attr)
    
    def add_edge(self, u, v, attr_dict=None, **attr):
        """Add edge between nodes u and v.  u and v must exist otherwise exception is thrown"""
        
        u_exist = u in self.nodes()
        
        if u_exist and v in self.nodes():
            super(Network, self).add_edge(u, v, attr_dict=attr_dict, **attr)
        else:
            if u_exist:
                raise NodeException("Node {0} does not exist!".format(v))
            else:
                raise NodeException("Node {0} does not exist!".format(u))
            
    def add_nodes_from(self, nodes, **attr):
        print "Adding Nodes {0}".format(nodes)
        super(Network, self).add_nodes_from(nodes, **attr)
        
    def add_node(self, n, **attr):
        print "Adding Node {0}".format(n)
        super(Network, self).add_node(n, **attr)
        
    def get_id(self, node):
        """Get id of a node"""
        return self.nodes().index(node)
                
    def get_node_by_id(self, id):
        """Get a node by id"""
        
        return self.nodes()[id]
        
    def get_node_subset(self, node_ids):
        """Return a subset of nodes from node ids"""
        return [n for n in nodes() if n in node_ids]
        #return dict((k, self.nodeids[k]) for k in node_ids)
        
    def is_acyclic(self, roots=None):
        """Uses a depth-first search (dfs) to detect cycles."""

        return nx.is_directed_acyclic_graph(self)    
       
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
        n_nodes = len(net.nodes())
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
            net = nx.to_networkx_graph(adjmat, create_using=Network())

            if net.is_acyclic():
                return net

        # got here without finding a single acyclic network.
        # so try with a less dense network
        return _randomize(density/2)

    # -----------------------

    net = Network(nodes)
    _randomize(net)
    return net

