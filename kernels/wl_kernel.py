import numpy as np
from hashlib import blake2b
from collections import Counter
import itertools

class WLKernel:
    def __init__(self, edge_attr="labels", node_attr="labels", iterations=3):
        self.edge_attr = edge_attr
        self.node_attr = node_attr
        self.n_iter = iterations
    
    def hash(self, label, digest_size):
        """
        Hashing function to encode labels from aggregated neighborhood
        """
        return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()

    def agg_neighbors(self, G, node, node_labels):
        """
        Compute new labels for given node by aggregating
        the labels of each node's neighbors.
        """
        label_list = []
        for nbr in G.neighbors(node):
            prefix = "" if self.edge_attr is None else str(G[node][nbr][self.edge_attr])
            label_list.append(prefix + node_labels[nbr])
        return node_labels[node] + "".join(sorted(label_list))

    def wl_graphs(self, G, digest_size=16):
        def wl_iter(G, labels):
            """
            Apply neighborhood aggregation to each node
            in the graph.
            Computes a dictionary with labels for each node.
            """
            new_labels = {}
            for node in G.nodes():
                label = self.agg_neighbors(G, node, labels)
                new_labels[node] = self.hash(label, digest_size)
            return new_labels

        # set initial node labels
        node_labels = {u: str(dd[self.node_attr]) for u, dd in G.nodes(data=True)}

        subgraph_hash_counts = {}
        for it in range(self.n_iter):
            node_labels = wl_iter(G, node_labels)
            counter = Counter(node_labels.values())
            # normalize counter
            total = np.sum(list(counter.values()))
            for k in counter:
                counter[k] /= total

            # sort the counter, extend total counts
            subgraph_hash_counts[it] = sorted(counter.items(), key=lambda x: x[0])

        # return _hash_label(str(tuple(subgraph_hash_counts)), digest_size)
        return subgraph_hash_counts
    
    
    def compute_phi(self, Z):
        """
        Computes feature map of each graph
        """
        phi_list = []
        for g in Z:
            phi_list.append(self.wl_graphs(g))
        return phi_list
    
    def compute_base_kernel(self, wl1, wl2):
        """
        For two dictionnaries containing histograms of the hashed labels for each iteration
        of the WL algorithm, compute scalar product between histograms
        """
        k = 0
        for i in range(self.n_iter):
            dict1 = dict(wl1[i])
            dict2 = dict(wl2[i])
            # take scalar product only on common keys
            common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
            k += np.sum([dict1[c]*dict2[c] for c in common_keys])
        return k

    def compute_kernel_matrix(self, X, Y):
        """
        Computes kernel matrix
        """

        # Compute latent representation
        phi_X = self.compute_phi(X)
        if np.array_equal(X, Y):
            print("Not computing phi again as X=Y")
            phi_Y = phi_X.copy()
        else:
            phi_Y = self.compute_phi(Y)
        
        # initialize kernel and number of iterations
        ker = np.zeros((len(X), len(Y)))
        count_iter = 0

        if len(X) == len(Y): # check that X and Y have same size
            for i in range(len(X)):
                for j in range(i, len(Y)): # make use of kernel symetry
                    ker[i, j] = self.compute_base_kernel(phi_X[i], phi_Y[j])
                    ker[j, i] = ker[i,j]
                count_iter += 1
                if count_iter % 100 == 0:
                    print(f"Iteration {count_iter}")
        
        else: # Ker will be rectangular
            for (i,j) in itertools.product(range(len(X)), range(len(Y))):
                ker[i,j] = self.compute_base_kernel(phi_X[i], phi_Y[j])
        print("Kernel computed")
        return ker