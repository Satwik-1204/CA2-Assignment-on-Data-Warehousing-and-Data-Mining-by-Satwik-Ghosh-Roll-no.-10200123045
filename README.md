# CA2-Assignment-on-Data-Warehousing-and-Data-Mining-by-Satwik-Ghosh-Roll-no.-10200123045
# Abstract
Traditional unsupervised clustering algorithms, such as K-Means, often fail to capture semantically meaningful structures in complex datasets because they rely solely on geometric similarity measures, ignoring valuable domain expertise. This report addresses the limitation of "blind" clustering by evaluating a semi-supervised framework that incorporates user-provided "Must-Link" and "Cannot-Link" constraints. To overcome the inefficiency of random constraint generation, an "Active Constraint Selection" strategy is proposed to enhance the COP-KMEANS algorithm. By iteratively identifying ambiguous data points with low confidence scores on cluster boundaries, the algorithm actively queries users for constraints on these specific pairs. This optimizes the trade-off between the cost of labeling and the purity of the final clusters.
# Introduction
Data clustering is fundamentally tasked with partitioning data points into groups such that intra-group similarity is maximized and inter-group similarity is minimized. However, purely geometric distance metrics (e.g., Euclidean distance) struggle to identify contextually correct clusters in complex, overlapping real-world data. 
To bridge the gap between mathematical compactness and contextual accuracy, semi-supervised clustering introduces background knowledge into the partitioning process. This background knowledge is provided via pairwise constraints:
•	Must-Link (ML): Specifies that two instances must be assigned to the same cluster.
•	Cannot-Link (CL): Specifies that two instances cannot be assigned to the same cluster.
A critical operational challenge in constraint-based clustering is the acquisition of these links. Constraints are often provided randomly or statically, which leads to redundancy and wasted user effort. Consequently, developing an active learning mechanism to intelligently select the most informative constraints is essential for maximizing algorithmic efficiency.
# Review of Previous Work
Significant advancements in semi-supervised clustering have established frameworks to incorporate pairwise constraints effectively. The paper Active Semi-Supervision for Pairwise Constrained Clustering by Sugato Basu, Arindam Banerjee, and Raymond J. Mooney presents a robust active semi-supervised clustering framework based on pairwise ML and CL constraints.
The authors introduce Pairwise Constrained Clustering (PCC) alongside a practical algorithm, PCK-Means. This algorithm extends traditional K-Means by jointly minimizing intra-cluster distance and the penalty cost associated with violating the provided constraints. The mathematical objective corresponds to Maximum a Posteriori (MAP) inference within a Hidden Markov Random Field, ensuring strong theoretical grounding and convergence to a local optimum.
To maximize the utility of limited supervision, the authors propose an active learning strategy divided into two sequential phases:
•	Explore: This phase utilizes farthest-first traversal to rapidly identify one representative from each true cluster, forming a reliable cluster skeleton. The algorithm continuously picks the farthest point from existing neighbourhoods until a predefined number of disjoint neighbourhoods (λ ≤ k) is established.
# Methodology
While previous methods focus heavily on initial cluster skeleton formation via farthest-first traversal, the proposed contribution in this project explores an alternate "Active Constraint Selection" strategy focused on decision boundaries.
The primary methodology relies on identifying the most "ambiguous" data points—specifically those lying on the boundaries of current clusters or exhibiting low confidence scores. Instead of querying the user randomly, the algorithm computes an ambiguity margin for each data point xi relative to the current estimated centroids. Let µ1 and µ2 be the closest and second-closest cluster centroids to xi, respectively. The ambiguity margin M(xi) is formulated as:
M(xi) = || xi - µ2 ||2 - || xi - µ1 ||2
Data points where M(xi) approaches zero are considered highly ambiguous, as they sit almost equidistant between two clusters. The algorithm actively queries the user oracle for ML or CL constraints exclusively on pairs containing these ambiguous points.
By focusing human supervision only on the most difficult cases (the decision boundaries), this approach aims to resolve the highest sources of algorithmic error. Once constraints are actively acquired, a constrained assignment logic (derived from COP-K-Means) is applied. This logic evaluates distance metrics while simultaneously ensuring that no assignment violates a Cannot-Link constraint.
5.	COMMENTS AND CONCLUSION

Execution Output Summary:
<===================================================> 
Dataset: Breast Cancer Wisconsin (30 Features) 
Samples: 569, Features: 30, Clusters: 2
--- Clustering Statistics --- 
Queries made: 25 data points 
Constraints generated: 146 Must-Link, 154 Cannot-Link
[Adjusted Rand Index (ARI)] - Higher is better (matches ground truth) 
Standard K-Means : 67.65% 
Active Constrained : 67.14% 
Improvement : -0.76%
[Normalized Mutual Info (NMI)] - Higher is better 
Standard K-Means : 56.20% 
Active Constrained : 57.55% 
<===================================================> 
Dataset: Handwritten Digits (64 Features) 
Samples: 1797, Features: 64, Clusters: 10
--- Clustering Statistics --- 
Queries made: 50 data points 
Constraints generated: 142 Must-Link, 1083 Cannot-Link
[Adjusted Rand Index (ARI)] - Higher is better (matches ground truth) 
Standard K-Means : 53.05% 
Active Constrained : 47.49% 
Improvement : -10.49%
[Normalized Mutual Info (NMI)] - Higher is better 
Standard K-Means : 67.20% 
Active Constrained : 62.73%
The empirical implementation of the proposed active constraint selection strategy yielded critical insights into the behaviour of hard constraints in high-dimensional spaces. As shown in the detailed output below, actively querying ambiguous points resulted in a slight increase in NMI for the Breast Cancer dataset (from 56.20% to 57.55%), indicating a slight improvement in information theoretic grouping. However, the Adjusted Rand Index (ARI) experienced a decline across both datasets, dropping by 0.76% in the Breast Cancer dataset and by a significant 10.49% in the Handwritten Digits dataset.
Scientifically, these negative results are highly informative. They demonstrate a known limitation of strict "hard" constraints in greedy clustering algorithms (often referred to as the "dead-end" problem). When a greedy algorithm strictly enforces Cannot-Link constraints on highly ambiguous boundary points in a dense 64-dimensional space, it can forcefully push points into entirely incorrect sub-optimal clusters to satisfy the rules, thereby degrading the overall geometric coherence.
In conclusion, while targeting ambiguous points successfully identifies difficult cases, enforcing strict constraint satisfaction on these boundaries is detrimental in complex datasets. Future work must transition from the implemented hard-constraint logic to a soft-constraint formulation—such as PCK-Means—where constraint violations incur a mathematical penalty rather than an absolute blockage.
•	Consolidate: This phase efficiently expands the established neighbourhoods by estimating centroids for each neighbourhood and sorting unassigned points based on increasing distances to these centroids. Additional points are assigned using a minimal number of user queries.
