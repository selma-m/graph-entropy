# Measuring Entropy Over Graph Datasets Using Geometric Scattering and Diffusion Geometry

<a href="senior_thesis.pdf">Senior Thesis<\a>

In this study, we propose a novel method to compute entropy over a set of graphs, providing a statistic that quantifies the variability of a graph dataset. Our approach is based on Diffusion Spectral Entropy (DSE), designed for the analysis of point clouds. Graph Neural Networks (GNNs) have traditionally been considered as a method to embed graphs into point clouds. However, we demonstrate the limitations of this approach and instead propose the use of geometric scattering, a transform that applies wavelet convolutions and non-linear activations to extract hierarchical, invariant features from data. Previous studies have shown that geometric scattering, an untrained embedding method, outperforms vanilla GNNs and produces embeddings that more effectively preserve the geometric and topological structure of the data. We demonstrate that our method effectively captures the variance of a graph dataset through experiments on toy datasets generated by increasingly perturbing a graph. Additionally, we apply our method to a real molecular dataset. The results highlight the potential of using diffusion spectral entropy combined with geometric scattering as a powerful tool for analyzing graph variability. 
