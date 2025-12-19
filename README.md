# Social Media Bot Detection using GNNs
This project implements a **Graph Neural Network (GNN)** to identify coordinated bot accounts on social media using the **Aditya Goyal Twitter Dataset**.

## Project Highlights
* **Algorithm:** Built a Graph Convolutional Network (GCN) using `torch_geometric`.
* **Dataset:** Real-world Twitter metadata including follower counts and retweet patterns.
* **Intervention Examples:** This model can be used for **Shadow-banning** and **Coordinated Takedown** of identified bot clusters.

## Results
![Detection Results](results.png)
The model successfully identifies patterns of automated behavior by analyzing the interaction graph.