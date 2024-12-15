import torch
import random
from torch_geometric.data import Data
import os

save_dir = './generated_graph_datasets'
os.makedirs(save_dir, exist_ok=True)


# For plotting graphs
import networkx as nx
import matplotlib.pyplot as plt


for n in range(10): # We'll be generating 10 original random graphs
    for num_nodes in [10]: 
        # We generate the original graph
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.randint(0, 1) == 1: # coin flip to decide whether to add an edge
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.eye(num_nodes)
        original_graph = Data(x=x, edge_index=edge_index)
        for n_rem_edges in [1, 2, 5, 7, 10, 15]:
            graph_dataset = [] 
            graph_dataset.append(original_graph)
            # We now remove the given number of edges
            for s in range(999): # this will create a dataset of 1,000 graphs
                num_edges = original_graph.edge_index.shape[1]
                k = min(n_rem_edges, num_edges)
                remove_idx = random.sample(range(num_edges), k)
                new_edge_index = torch.cat([original_graph.edge_index[:, t].unsqueeze(1) for t in range(num_edges) if t not in remove_idx], 
                                           dim=1)
                new_graph = Data(x=original_graph.x, edge_index=new_edge_index)
                graph_dataset.append(new_graph)
            
            # We save the dataset
            save_path = os.path.join(save_dir, f'graph_{n}_{num_nodes}_nodes_{n_rem_edges}_removed_edges.pt')
            torch.save(graph_dataset, save_path)

            # Draw the original graph
            #G = nx.Graph()
            #for m in range(original_graph.edge_index.shape[1]):
                #u, v = original_graph.edge_index[0, m].item(), original_graph.edge_index[1, m].item()
                #G.add_edge(u, v)
            #plt.figure(figsize=(8, 8))
            #pos = nx.spring_layout(G, seed=42) 
            #nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', edge_color='gray')
            #original_graph_path = os.path.join(save_dir, f'original_graph_{n}_{num_nodes}_nodes_{n_rem_edges}_removed_edges.png')
            #plt.savefig(original_graph_path, dpi=300)
            #plt.close()
            
            #G_new = nx.Graph()
            #generated = graph_dataset[29]
            #for p in range(generated.edge_index.shape[1]):
                #u, v = generated.edge_index[0, p].item(), generated.edge_index[1, p].item()
                #G_new.add_edge(u, v)
            #plt.figure(figsize=(8, 8))
            #pos = nx.spring_layout(G_new, seed=42) 
            #nx.draw(G_new, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', edge_color='gray')
            #generated_graph_path = os.path.join(save_dir, f'generated_graph_{n}_{num_nodes}_nodes_{n_rem_edges}_removed_edges.png')
            #plt.savefig(generated_graph_path, dpi=300)
            #plt.close()



# 30 nodes
for n in range(10): # We'll be generating 10 original random graphs
    for num_nodes in [30]: 
        # We generate the original graph
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.randint(0, 1) == 1: # coin flip to decide whether to add an edge
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.eye(num_nodes)
        original_graph = Data(x=x, edge_index=edge_index)
        for n_rem_edges in [1, 10, 30, 50, 75, 100]:
            graph_dataset = [] 
            graph_dataset.append(original_graph)
            # We now remove the given number of edges
            for s in range(999): # this will create a dataset of 1,000 graphs
                num_edges = original_graph.edge_index.shape[1]
                k = min(n_rem_edges, num_edges)
                remove_idx = random.sample(range(num_edges), k)
                new_edge_index = torch.cat([original_graph.edge_index[:, t].unsqueeze(1) for t in range(num_edges) if t not in remove_idx], 
                                           dim=1)
                new_graph = Data(x=original_graph.x, edge_index=new_edge_index)
                graph_dataset.append(new_graph)
            
            # We save the dataset
            save_path = os.path.join(save_dir, f'graph_{n}_{num_nodes}_nodes_{n_rem_edges}_removed_edges.pt')
            torch.save(graph_dataset, save_path)




# 50 nodes
for n in range(10): # We'll be generating 10 original random graphs
    for num_nodes in [50]: 
        # We generate the original graph
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.randint(0, 1) == 1: # coin flip to decide whether to add an edge
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.eye(num_nodes)
        original_graph = Data(x=x, edge_index=edge_index)
        for n_rem_edges in [1, 20, 50, 100, 200, 300]:
            graph_dataset = [] 
            graph_dataset.append(original_graph)
            # We now remove the given number of edges
            for s in range(999): # this will create a dataset of 1,000 graphs
                num_edges = original_graph.edge_index.shape[1]
                k = min(n_rem_edges, num_edges)
                remove_idx = random.sample(range(num_edges), k)
                new_edge_index = torch.cat([original_graph.edge_index[:, t].unsqueeze(1) for t in range(num_edges) if t not in remove_idx], 
                                           dim=1)
                new_graph = Data(x=original_graph.x, edge_index=new_edge_index)
                graph_dataset.append(new_graph)
            
            # We save the dataset
            save_path = os.path.join(save_dir, f'graph_{n}_{num_nodes}_nodes_{n_rem_edges}_removed_edges.pt')
            torch.save(graph_dataset, save_path)

            

