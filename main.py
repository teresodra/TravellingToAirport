from way_to_airport import initial_weights
from travelling_options import TravelOption
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create costs for different variances to plot
    cost_curves = {}
    variance_values = np.linspace(0.01, 50, 100)
    for variance in variance_values:
        travel_option = TravelOption(initial_graph_weights=initial_weights, variance=variance, final_node=12)
        expected_costs = travel_option.exploring_expected_costs()
        # Store results: dict node → list of costs for each variance

        
        node_costs = travel_option.exploring_expected_costs()

        for node in node_costs:
            if node not in cost_curves:
                cost_curves[node] = []
            cost_curves[node].append(node_costs[node])

    # ----------------------
    #       PLOT
    # ----------------------
    plt.figure(figsize=(10, 6))

    for node in cost_curves:
        plt.plot(
            variance_values,
            cost_curves[node],
            label=f"Node {node}",
            linewidth=2
        )

    plt.xlabel("Variance")
    plt.xlim(0, 50)
    plt.ylabel("Expected Cost")
    plt.ylim(0, 170)
    plt.title("Expected Cost vs Variance for Each Starting Node")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("expected_costs_vs_variance.png")
    plt.show()