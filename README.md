# Federated Learning on Graph Neural Network
 
Graph neural networks (GNN) have recently been receiving a lot of attention due to their ability to represent high-dimensional node relationships and attributes in a convenient way that utilizes the progress achieved in deep learning. However, GNNs face privacy concerns since node edges and features may be considered private in many applications. In the euclidean data domain, federated learning is an emerging setting for learning a machine learning model in a decentralized and collaborative way across multiple clients. Specifically, model coefficients are communicated from the server to the clients which perform model updates based on their data and feed these updates back to the server. In this setting, clients share model updates while their data is kept private. Thus, federate learning naturally embraces data privacy. Therefore, it is an interesting area of research to consider merging GNNs with FL. In this work, we aim at developing an FL setting for GNNs where we maintain data privacy. Then, we evaluate the applicability of this setting in different federation types and graph-related tasks. This is done both numerically in terms of performance quality metrics, and mathematically in terms of statistical analysis. 
 
## I. Introduction

Unlike euclidean data such as images and videos, graphs contain features and links between their nodes. Furthermore, graphs tend to have huge numbers of nodes with complicated connections that make standard machine learning methods unable to cope up with. In recent years, Graph Neural Networks (GNN)s have been receiving an increasing amount of attention as a key framework for representing graph data. This awes to their proven capability in representing both features and connections at high dimensions for graph data[^20]. GNN has a wide range of appreciations on various graph types including; computer vision[^19], fraud detection[^12], text classification[^21], and social recommendation[^4].

Despite the success of GNNs in representing graph data, and forming a framework for a variety of graph-level, node-level, and edge-level tasks, there is an inherent concern about their tendency to reveal private data. Specifically, The features and connections of graph data may potentially be private and not subject to sharing or exposure to the public. However, just like all other machine learning models, GNNs are vulnerable to adversarial privacy attacks where an adversary aims at exposing private data from publicly available data. A typical example in this context is inferring user attributes or connections in a social network platform based on generally published data. 

As a form of distributed learning, federated learning (FL)[^15] aggregates user-end machine learning (ML) model contributions to obtain a common ML model. This learning paradigm has recently emerged and received a lot of attention as it inherently solves two main issues with distributed learning[^9]. First, is user privacy concerns; as user data is kept at the user end. Second, is the need for communicating huge amounts of data over the network; since only the ML coefficients (or their updates) are communicated. FL relies on two main pillars; first, is local training on client's data, and second is model's parameters aggregation. In each round of FL training, each client receives the model's parameters from the server; trains a model based on its (private) data, and sends the trained model's parameters back to the server.  After receiving the parameters from the clients, the server aggregates model contributions into a global model. This model is then sent to clients, and so on[^22]. Therefore, in the FL setting, the data of each user is never shared with other parties which naturally preserves user data and preserver privacy. The general concept of FL is illustrated in Figure 1.

<p align="center">
<img width="500" src="https://github.com/khang4dang/Federated-Learning-on-Graph-Neural-Network/blob/main/images/Figure_1.png">
</p>
<p align="center"><b>
Figure 1: The Federated Learning Concept
</b></p>

FL has a wide range applications in euclidean data domain: healthcare recommendation[^16], google keyboard recommendation[^1]. However, it is still in its first steps to be applied to the graph domain. The majority of GNN works concern the efficiency and scalability aspects of GNNs[^11]. A minority of works consider the privacy concerns in GNNs. 

Since its inception, FL has been known to yield two main advantages. First, is protecting user's privacy, and second is alleviating the need for data communication as model coefficients are alternatively communicated[^8]. To this end, FL in the a GNN context inherently reaps those advantages. This can be detailed as follows.

- **Privacy preservation:** Using federated learning, wireless APs at different geolocations collaborate without breaking the data privacy of their individual users. This allows for applying FL at a large scale (e.g., cross-company or cross country).
- **Reducing communicating demands:** Rather than sharing their data points (observations), users in an FL context only share their model coefficients (or coefficient updates). This substantially reduces the burden on communicating resources and saves such (scarce and valuable) resources for communicating other useful information.

The heterogeneity of data across clients poses a major challenge against the usefulness of the FL setting in practice. Graph data is not an expectation; such data is typically heterogeneous between clients, as conceptually illustrated by Figure 2.

Motivated by the above discussion, in this work, we plan to devise an FL setting for GNNs. Also, we plan to investigate the usefulness of this setting on two main tasks: node classification and graph classification. These are summarized as follows:

- **Node Classification:** This task is to predict the labels of individual nodes in graphs. It is more important in node-level FL, such as predicting the active research fields of an author based on his/her $k$-hop collaborators or habits of a user based on his/her $k$-hop friends. 
- **Graph Classification:** This task is to categorize different types of graphs based on their structure and overall information. Unlike other tasks, this requires to characterize the property of the entire input graph. This task is naturally important in graph-level FL, with real examples such as molecule property prediction, protein function prediction, and social community classification.

<p align="center">
<img width="700" src="https://github.com/khang4dang/Federated-Learning-on-Graph-Neural-Network/blob/main/images/Figure_2.png">
</p>
<p align="center"><b>
Figure 2: An example of heterogeneity of graph data between different clients in federated scenarios. Client A/B/C has 9/5/4 nodes, 9/5/3 edges, and 4/2/1 labels. The three clients vary in the number of nodes, edges, and labels, and this difference becomes more obvious on large-scale graphs in the real world. This is so-called the heterogeneity.
</b></p>

We first formulate graph FL to provide a unified framework for federated GNNs (Section 3). Under this formulation, we introduce the various graph datasets with synthesized partitions according to real-world application scenarios (Section 4). Extensive empirical analysis demonstrates the utility and efficiency of our system and indicates the need for further research in graph FL (Section 4). Finally, we summarize our work as well as future directions (Section 5).

## II. Problem Description

FL setting is a distributed learning method that addresses data privacy concerns. In FL, training is an act of involving multiple clients in collaboration without requiring centralized local data. Despite its successful application in many domains, FL has yet to be widely adopted in the domain of machine learning on graph data. There are multiple reasons for this:
- There is a lack of formulation over the various graph FL settings and tasks in literature review, making it difficult for scientists who focus on federated optimization algorithms to comprehend challenges in federated GNNs;
- Existing FL libraries do not support diverse graph datasets and learning tasks to different models and training algorithms. Given the complexity of graph data, the dynamics of training GNNs in an FL setting can be vary between several models. A fair and easy-to-use setting with different datasets and reference implementations is essential to the development of new graph FL models and algorithms;
- The simulation-oriented federated training system is inefficient and insecure for federated GNNs research on large-scale and private graph datasets. Disruptive research ideas may be constrained by the lack of a modularized federated training system tailored for diverse GNN models and FL algorithms.

To address the above problems, we conduct statistical and empirical analyses on the context that FL can be applied to GNN and also the effects of FL on the model performance and convergence rate. Our empirical analysis showcases the utility of our FL setting while exposing significant challenges in graph FL.

## III. Methodology

<p align="center">
<img width="600" src="https://github.com/khang4dang/Federated-Learning-on-Graph-Neural-Network/blob/main/images/Figure_3.png">
</p>
<p align="center"><b>
Figure 3: Architecture of Federated Learning Setting on Graph Neural Network.
</b></p>

We present an FL setting for Graph Neural Networks (GNN)s, which contains a variety of graph datasets from different domains and eases the training and evaluation of GNN models and FL algorithms. In our architecture of FL setting on GNN (Figure 3), we assume that there are $K$ clients. Each client has its own dataset $d_k$ with feature sets ${x_k}$ $\in$ $R^{d_k}$ and ${z_k}$ $\in$ $R^{d_k}$. They also have their own GNN model to learn graph presentations and make predictions. Multiple clients collaborate through a server to improve their GNN model without revealing their graph dataset. The algorithm 2 illustrate our proposed method. In each federated round, the clients update their local models and send the model's parameters back to the server. Then the server average the models' parameters to create the global model.

<p align="center">
<img width="800" src="https://github.com/khang4dang/Federated-Learning-on-Graph-Neural-Network/blob/main/images/Algo_1.png">
</p>

<p align="center">
<img width="800" src="https://github.com/khang4dang/Federated-Learning-on-Graph-Neural-Network/blob/main/images/Algo_2.png">
</p>

As illustrated by Figure 3, clients can have data of different types and dimensions. Therefore, their specific GGN architectures may not be compatible in terms of their coefficients and structures and hence can not be aggregated directly. To tackle this problem, we propose a setting where we share model updates on the fully-connected level (FCN), as depicted by Figure 3. In this setting, each client can have its own dimensional while the FL can take palace. It is note-worthy to mention that the server's centralized GNN model will be updated and shared at the FCN level, thereby assuring the dimensional compatibility between client models. 

## IV. Experiments
 
### 1. Experimental Setting
In this section, we investigate the performance of the graph convolutional network in the FL setting. Since the data of each client is not from the same distribution, the performance and also the convergence of the training process are interesting to investigate. We conduct our experiments on both node classification task and graph classification task which are described in the following sub-sections.

We use Python run on Google's Colab as a simulation platform. We utilize Spektral[^6], a Python library for graph deep learning, based on the Keras API and TensorFlow 2. Spektral is rich of tools for generating, simulating, and attesting graph neural networks. We refer the interested reader to Spektral's website for a collection of datasets, implementations, and examples on GNNs in multiple applications.

#### a. Node Classification Task

In this task, we investigate two scenarios of downstream tasks of each client: homogeneous and heterogeneous graph tasks. In homogeneous graph tasks, the clients' graph data share the same intrinsic properties (e.g Facebook and Twitter have different distributions but they're both social networks, Cora[^13] and citeseer[^5] have different label sets, but they are both citation networks) while the heterogeneous graph data has different intrinsic properties. The architectures of the GNN models used for these three datasets are presnted in the Appendix.

**Homogeneous graph tasks:** We experiment on three data sets of citation networks: Cora, Citeseer, Pubmed[^17]. The records contain bag-of-words feature vectors for each document and a list of inter-document citation links. We treat the citation links as edges (undirected). Each local model combines a Dense layer to map to the same feature space, the common 2-layer graph convolutional network (whose parameters will be sent to the server), and one Fully Connected Layer (FCN) for the downstream task. Each client trains the local models in 20 ep using Adam optimizers with a learning rate equal to 0.01. 

**Heterogeneous graph task:** Beside the datasets that mentioned above, we also test on Reddit dataset[^7] which is a social network community detection task. The Reddit dataset is a graph dataset from Reddit posts made in the month of September, 2014. The node label in this case is the community, or “subreddit”, that a post belongs to. 50 large communities have been sampled to build a post-to-post graph, connecting posts if the same user comments on both. The local model is the same as above. 

#### b. Graph Classification Task

In this task, different types of graphs based on their structure and overall information are categorized. Unlike the node classification task, this requires an entire input graph to characterize its property. It is important in graph-level FL.

For this part, we use the TUDataset[^14], which is a collection of benchmark datasets for graph classification and regression. In particular, we choose the dataset PROTEINS from the TUDataset website. PROTEINS contains data for 1113 protein graphs collected from [^2] and [^3]. In this setting, the average node per graph of 39.06. Besides, the average number of edges per graph is 72.82. This set is a labeled set with 2 classes. Thus, we consider binary classification. For each client, we draw training, testing, and validation data randomly from the PROTEINS set. As another dataset, we also consider the DD set [^18] from the same source. More specifically, this set is composed of 1178 with 284.32 and 715.66 average nodes and edges per graph, respectively. In these sets, each protein is represented by a
graph, in which the nodes are amino acids and two nodes are connected by an edge if they are less
than 6 Angstroms apart. The prediction task is to classify the protein structures into enzymes and non-enzymes. Note that nodes are labeled in all data sets.

### 2. Results

In this section, we present the results of the aforementioned experiments, discuss, and draw conclusive remarks and findings.

#### a. Node Classification Task

**Homogeneous graph tasks:**

The model performance in training process of the homogeneous graph task setting is presented in Figure 4 and the testing accuracy for each local model is presented in the Table 1. In federated learning setting, the model performance of each local task suffer from a minor drop in model performance. Comparing with the work of [^10], the accuracy in inference time for cora dataset drops 5.2%, for citeseer dataset drops 9.8% and for pubmed dataset drops 5.5%. This is expected since the graphs of the three datasets has different distributions (non-i.i.d) which reduce the performance of the downstream tasks. However, the drop is reasonable for real-world applications.

<p align="center">
<img width="1000" src="https://github.com/khang4dang/Federated-Learning-on-Graph-Neural-Network/blob/main/images/Figure_4.png">
</p>
<p align="center"><b>
Figure 4: Results of Node Classification Implementation for Homogeneous Graph Task
</b></p>

As in Figure 4, Federated learning still assures convergence in the learning process of each local models. It took around 40 - 50 rounds of federated learning to reach the limit points of each models. This results is significantly improved for centralized setting[^10], which usually take around 200 epochs for each centrallized model to reach convergence points.

**Table 1: Testing Accuracy of Node Classification - Homogeneous Graph Tasks**

| Dataset   | GCN[^10] | FL setting |
| :---:     | :---:    | :---:      |
| Cora      | 81.5     | 76.3       |
| Citeseer  | 70.3     | 60.5       |
| Pubmed    | 79.0     | 73.5       |

**Heterogeneous graph tasks:**

The model performance in training process of the heterogeneous graph task setting is presented in Figure 5 and the testing accuracy for each local model is presented in the Table 2. In federated learning setting, the model performance of each local task suffer from a minor drop in model performance. Comparing with the work of [^10], the accuracy in inference time for cora dataset drops 5.6%, for citeseer dataset drops 9.6% and for pubmed dataset drops 4.3%. This is expected since the graphs of the three datasets has different distributions (non-i.i.d). However, comparing with the homogeneous task, by enhancing the structure of reddit dataset, the performance of the citation networks is improved.


<p align="center">
<img width="1000" src="https://github.com/khang4dang/Federated-Learning-on-Graph-Neural-Network/blob/main/images/Figure_5.png">
</p>
<p align="center"><b>
Figure 5: Results of Node Classification Implementation for Heterogeneous Graph Task
</b></p>

As in Figure 5, Federated learning still assures convergence in the learning process of each local models. It took around 10 rounds of federated learning to reach the limit points of each models. This results is significantly improved for centralized setting and also the homogeneous tasks. This is easily understandable since the reddit datasets has significantly large number of nodes comparing to the citation networks which give the global model the ability to capture the better structural information in a faster rate.

**Table 2: Testing Accuracy of Node Classification - Heterogeneous Graph Tasks**

| Dataset   | GCN[^10] | FL setting |
| :---:     | :---:    | :---:      |
| Cora      | 81.5     | 75.9       |
| Citeseer  | 70.3     | 60.7       |
| Pubmed    | 79.0     | 74.7       |
| Reddit[^7]| 90.1     | 94.0       |

#### b. Graph Classification Task

A set of preliminary results for graph classification are presented in Figures 6 and 7. Several observations can be drawn from these figures. First, it is seen that the federated setting works on the graph classification level. This is evident due to the general improvement of the learning observed at each of the three clients. Another observation is that the rate of convergence of learning differs from one client to another. This is due to the inherent data heterogeneity exhibited across different clients, as opposed to centralized learning. In particular, heterogeneity is strongly visible in the sense that the validation loss of the first client is much less than these of the latter two clients.

<p align="center">
<img width="1000" src="https://github.com/khang4dang/Federated-Learning-on-Graph-Neural-Network/blob/main/images/Figure_6.png">
</p>
<p align="center"><b>
Figure 6: Results of Graph Classification Implementation on the PROTEINS Dataset
</b></p>

Similar can be made in view of Figure 7 which shows the results of the same experiment conducted over the DD dataset. 

<p align="center">
<img width="1000" src="https://github.com/khang4dang/Federated-Learning-on-Graph-Neural-Network/blob/main/images/Figure_7.png">
</p>
<p align="center"><b>
Figure 7: Results of Graph Classification Implementation on the DD Dataset
</b></p>

## V. Conclusion and Future Work

In this work, we develop a privacy-preserving FL setting for learning GNNs on node classification task and graph classification task. We conduct comprehensive evaluations of the setting's performance and convergence rate. We emphasize the possibility of doing FL on multiple GNN clients to unify the embedding space dimension to bridge the gap in the feature dimensionality. We successfully evaluate the performance of the tasks in terms of accuracy and loss measures.

Some future improvements and research directions based on our FL setting are summarized here: 
- Supporting more datasets considering their heterogeneity and non-iid-ness as well as various GNN models for diverse applications
- Optimizing the setting to increase the training speed applied on larger graphs
- Searching for personalized GNN models by designing advanced graph FL algorithms that improve the accuracy on datasets with non-iid-ness
- Investigating and mitigating more challenges in the security and privacy under our FL setting
- Proposing ethics and societal impacts to avoid unwanted negative effects considering the sub-graph scenario

## Appendix

Model implementation of datasets for Node Classification Task: Cora dataset (Figure \ref{cora_model}), Citeseer dataset (Figure \ref{citeseer_model}), and Pubmed dataset (Figure \ref{pubmed_model}).

\begin{figure}[!htb]
    \centering
    \resizebox{0.8\columnwidth}{!}{
    \includegraphics[width=\textwidth]{cora_model.png}}
    \caption{Model Implementation of Cora dataset}
    \label{cora_model}
\end{figure}

\begin{figure}[!htb]
    \centering
    \resizebox{0.8\columnwidth}{!}{
    \includegraphics[width=\textwidth]{citeseer_model.png}}
    \caption{Model Implementation of Citeseer dataset}
    \label{citeseer_model}
\end{figure}

\begin{figure}[!htb]
    \centering
    \resizebox{0.8\columnwidth}{!}{
    \includegraphics[width=\textwidth]{pubmed_model.png}}
    \caption{Model Implementation of Pubmed dataset}
    \label{pubmed_model}
\end{figure}

\end{document}

---
<p align="center"><b>
Khang Dang, Mahmoud Nazzal & Khang Tran
</b></p>

[^1]: K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, and K. Seth. Practical secure aggregation for privacy-preserving machine learning. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, CCS ’17, page 1175–1191, New York, NY, USA, 2017. Association for Computing Machinery.

[^2]: K. M. Borgwardt, C. S. Ong, S. Schönauer, S. Vishwanathan, A. J. Smola, and H.-P. Kriegel. Protein function prediction via graph kernels. Bioinformatics, 21(suppl_1):i47–i56, 2005.

[^3]: P. D. Dobson and A. J. Doig. Distinguishing enzyme structures from non-enzymes without alignments. Journal of molecular biology, 330(4):771–783, 2003.

[^4]: W. Fan, Y. Ma, Q. Li, Y. He, Y. E. Zhao, J. Tang, and D. Yin. Graph neural networks for social recommendation. CoRR, abs/1902.07243, 2019.

[^5]: C. L. Giles, K. D. Bollacker, and S. Lawrence. Citeseer: An automatic citation indexing system. In Proceedings of the Third ACM Conference on Digital Libraries, DL ’98, page 89–98, New York, NY, USA, 1998. Association for Computing Machinery.

[^6]: D. Grattarola and C. Alippi. Graph neural networks in tensorflow and keras with spektral [application notes]. IEEE Computational Intelligence Magazine, 16(1):99–106, 2021.

[^7]: W. L. Hamilton, R. Ying, and J. Leskovec. Inductive representation learning on large graphs. CoRR, abs/1706.02216, 2017.

[^8]: P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, et al. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977, 2019.

[^9]: P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. A. Bonawitz, Z. Charles, G. Cormode, R. Cummings, R. G. L. D’Oliveira, S. E. Rouayheb, D. Evans, J. Gardner, Z. Garrett, A. Gascón, B. Ghazi, P. B. Gibbons, M. Gruteser, Z. Harchaoui, C. He, L. He, Z. Huo, B. Hutchinson, J. Hsu, M. Jaggi, T. Javidi, G. Joshi, M. Khodak, J. Konecný, A. Korolova, F. Koushanfar, S. Koyejo, T. Lepoint, Y. Liu, P. Mittal, M. Mohri, R. Nock, A. Özgür, R. Pagh, M. Raykova, H. Qi, D. Ramage, R. Raskar, D. Song, W. Song, S. U. Stich, Z. Sun, A. T. Suresh, F. Tramèr, P. Vepakomma, J. Wang, L. Xiong, Z. Xu, Q. Yang, F. X. Yu, H. Yu, and S. Zhao. Advances and open problems in federated learning. CoRR, abs/1912.04977, 2019.

[^10]: T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. CoRR, abs/1609.02907, 2016.

[^11]: H. Li, Y. Liu, Y. Li, B. Huang, P. Zhang, G. Zhang, X. Zeng, K. Deng, W. Chen, and C. He. Graphtheta: A distributed graph neural network learning system with flexible training strategy. CoRR, abs/2104.10569, 2021.

[^12]: Z. Liu, C. Chen, X. Yang, J. Zhou, X. Li, and L. Song. Heterogeneous graph neural networks for malicious account detection. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, CIKM ’18, page 2077–2085, New York, NY, USA, 2018. Association for Computing Machinery.

[^13]: A. K. McCallum, K. Nigam, J. Rennie, and K. Seymore. Automating the construction of internet portals with machine learning. Information Retrieval, 3(2):127–163, 2000.

[^14]: C. Morris, N. M. Kriege, F. Bause, K. Kersting, P. Mutzel, and M. Neumann. Tudataset: A collection of benchmark datasets for learning with graphs. In ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020), 2020.

[^15]: V. Mothukuri, R. M. Parizi, S. Pouriyeh, Y. Huang, A. Dehghantanha, and G. Srivastava. A survey on security and privacy of federated learning. Future Generation Computer Systems, 115:619–640, 2021.

[^16]: N. Rieke, J. Hancox, W. Li, F. Milletari, H. Roth, S. Albarqouni, S. Bakas, M. N. Galtier, B. A. Landman, K. H. Maier-Hein, S. Ourselin, M. J. Sheller, R. M. Summers, A. Trask, D. Xu, M. Baust, and M. J. Cardoso. The future of digital health with federated learning. CoRR, abs/2003.08119, 2020.268

[^17]: P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Galligher, and T. Eliassi-Rad. Collective classification in network data. AI Magazine, 29(3):93, Sep. 2008.

[^18]: N. Shervashidze, P. Schweitzer, E. J. Van Leeuwen, K. Mehlhorn, and K. M. Borgwardt. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12(9), 2011.

[^19]: Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon. Dynamic graph CNN for learning on point clouds. CoRR, abs/1801.07829, 2018.

[^20]: Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and P. S. Yu. A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems, 32(1):4–24, 2021.

[^21]: L. Yao, C. Mao, and Y. Luo. Graph convolutional networks for text classification. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):7370–7377, Jul. 2019.

[^22]: C. Zhang, Y. Xie, H. Bai, B. Yu, W. Li, and Y. Gao. A survey on federated learning. Knowledge-Based Systems, 216:106775, 2021.
