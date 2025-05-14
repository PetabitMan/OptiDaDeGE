import json
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define colors
COLORS = {
    "primary": {
        "midnight_blue": "#002147",
        "steel_blue": "#4682B4"
    },
    "secondary": {
        "light_blue": "#B0C4DE",
        "light_gray": "#D3D3D3"
    },
    "accent": {
        "dark_gray": "#A9A9A9",
        "white": "#FFFFFF"
    }
}

# Specify the path to your cases file
cases_file_path = 'sufficient_cases.json'  # Replace with your actual file path

# Specify the output path for the graph plot
output_graph_path = 'case_references_graph_top_nodes.png'

# Load the cases file
print("Loading cases file...")
with open(cases_file_path, 'r', encoding='utf-8') as f:
    cases = json.load(f)

# Initialize directed graph
print("Building case reference graph...")
G = nx.DiGraph()

# Add nodes and edges
main_case_ids = set()
for case in tqdm(cases, desc="Processing cases"):
    case_id = case.get('id')  # Replace 'id' with the actual field for case identifier
    main_case_ids.add(case_id)
    references = case.get('case_references', [])
    G.add_node(case_id)
    for ref in references:
        G.add_edge(case_id, ref)

# Graph Metrics
print("\nGraph Metrics:")
print(f"Number of nodes (cases): {G.number_of_nodes()}")
print(f"Number of edges (references): {G.number_of_edges()}")

# Select the top nodes by in-degree and out-degree
top_in_degree_nodes = [node for node, _ in sorted(G.in_degree, key=lambda x: x[1], reverse=True)[:300]]
top_out_degree_nodes = [node for node, _ in sorted(G.out_degree, key=lambda x: x[1], reverse=True)[:100]]

# Combine top nodes and create a subgraph
top_nodes = set(top_in_degree_nodes + top_out_degree_nodes)
subgraph = G.subgraph(top_nodes)

# Updated metrics for the subgraph
print("\nSubgraph Metrics:")
print(f"Number of nodes in subgraph: {subgraph.number_of_nodes()}")
print(f"Number of edges in subgraph: {subgraph.number_of_edges()}")

# Node sizes based on in-degree only
node_sizes = [1 + (subgraph.in_degree[node] * 2) for node in subgraph.nodes]  # Scale factor of 2 for better visualization

# Node colors: main cases = blue, referenced-only cases = steel blue
node_colors = [
    COLORS["primary"]["midnight_blue"] if node in main_case_ids else COLORS["primary"]["steel_blue"]
    for node in subgraph.nodes
]

# Plot the subgraph
print(f"\nSaving graph visualization to {output_graph_path}...")
plt.figure(figsize=(15, 15))

# Use a spring layout for better visualization
pos = nx.spring_layout(G, iterations=10)
# Draw nodes with sizes and colors
nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
nx.draw_networkx_edges(subgraph, pos, alpha=0.1, edge_color=COLORS["secondary"]["light_gray"])

# Add labels for the top referenced nodes in the subgraph
top_referenced = sorted(subgraph.in_degree, key=lambda x: x[1], reverse=True)[:20]
top_labels = {case: case for case, _ in top_referenced}
nx.draw_networkx_labels(subgraph, pos, labels=top_labels, font_size=10, font_color=COLORS["accent"]["dark_gray"])

# Finalize plot
plt.title("Case References Graph (Top In/Out-Degree Nodes)")
plt.axis('off')
plt.tight_layout()

# Save the graph
plt.savefig(output_graph_path)
plt.close()

print(f"Graph visualization saved successfully to {output_graph_path}")


