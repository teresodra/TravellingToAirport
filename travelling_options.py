
from distributions import DistApprox, sum_distributions, min_distributions


class TravelOption:
    """Class representing travel options to the airport with uncertain travel times."""
    def __init__(self, initial_graph_weights, variance=10, final_node=12):
        self.initial_graph_weights = initial_graph_weights
        self.nodes = set([node for startend in initial_graph_weights.keys() for node in startend])
        self.final_node = final_node
        self.variance = variance
        self.distributions = dict() # Contains a distribution for each node on the time required to reach destination from it

    def find_distributions(self, node, timestep=0):
        """
        Recursively find the time distributions to reach the end for each node.
        """
        if (node, timestep) in self.distributions:
            return self.distributions[node, timestep]
        
        outgoing_edges = [(start, end) for (start, end) in self.initial_graph_weights.keys() if start == node]
        node_distributions = []
        for (start, end) in outgoing_edges:
            travel_time_expected = self.initial_graph_weights[(start, end)]
            travel_dist = DistApprox.from_normal(mean=travel_time_expected, std=self.variance**0.5*timestep, x_min=0, x_max=100, n_points=500)
            if end == self.final_node:
                node_distributions.append(travel_dist)
            else:
                node_distributions.append(sum_distributions(travel_dist, self.find_distributions(end, timestep + 1)))
        self.distributions[node, timestep] = min_distributions(*node_distributions)
        return self.distributions[node, timestep]


    def exploring_expected_costs(self):
        """
        Estimate expected costs from each possible starting node.
        """
        possible_nodes = [end for (start, end) in self.initial_graph_weights.keys() if 0 == start]
        expected_costs = dict()
        for node in possible_nodes:
            self.find_distributions(node)
            
            # Probability that travelling from node 1 to the airport takes less than 86 minutes
            probability86 = self.distributions[node, 0].prob_between(b=86 - self.initial_graph_weights[(0, node)])
            # Probability that travelling from node 1 to the airport takes less than 95 minutes
            probability95 = self.distributions[node, 0].prob_between(b=95 - self.initial_graph_weights[(0, node)])
            # print(f"Probability (86): {probability86:.4f}")
            # print(f"Probability (95): {probability95:.4f}")
            # Cost is 600 * (1 - probability95) + 80 * (probability95 - probability86)
            cost = 600 * (1 - probability95) + 80 * (probability95 - probability86)
            # print(f"Estimated cost from node {node}: {cost:.2f}")
            expected_costs[node] = cost
        return expected_costs


