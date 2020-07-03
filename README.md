# Ultrahyperbolic Representation Learning

This is the code related to the paper "Ultrahyperbolic Representation Learning" available at https://arxiv.org/abs/2007.00211

## Introduction

We created one directory for each dataset: Zachary's karate club dataset and NIPS co-authorship dataset as described in the paper.

### Prerequisites

The training code was tested with Python 3.6.7 and PyTorch 1.0.1.post2.
The evaluation scripts were tested on Matlab_R2016b. 

## Zachary's karate club dataset

### Training 

To learn ultrahyperbolic representations, run the following code:
```
python zachary_experiments.py
```

- To train in the weighted graph setup (i.e. edge weights are not only 1), set the variable **weighted_version** to True. By default, the code considers the unweighted graph setup. 
If **weighted_version** is set to True, the algorithm stops after 10000 iterations.

- To train with the optimizer introduced in Section 4.1, set the variable **apply_standard_sgd** to True. By default, the code optimizes the pseudo-Riemannian optimizer introduced in Section 4.2.

- To use the pseudo-Riemannian gradient (see Eq. (11) of the paper) as search direction, set the variable **use_pseudoRiemannian_gradient** to True. By default, the code uses the proposed descent direction introduced in Eq. (14) of the paper, except in the Riemannian case where the negative of the Riemannian gradient can be used as descent direction. 
When the metric tensor is not positive definite, the optimizer should not converge when the pseudo-Riemannian gradient is used as search direction as explained in the paper.

- The variables **space_dimensions** and **time_dimensions** can be set to different values. Please note that the variable **q** in the code corresponds to (q+1) in the paper.

By default, the code runs on CPU. It converges relatively fast (less than 10 minutes) on recent CPUs since the dataset is small. 

### Evaluation

The MATLAB evaluation script is provided in the file "zachary_evaluate_representations.m". It opens the distance matrix files saved for 5 different random splits and evaluates different metrics reported in the paper.

- To evaluate Euclidean representations optimized with the squared Euclidean distance, set the variable **evaluate_euclidean_representations** to True. By default, the script considers pseudo-hyperbolic cases. 

- Set the variable **time_dimensions** to the appropriate number of time dimensions you want to evaluate 4-dimensional pseudo-hyperboloids with. For example, the directory "d_5_q_1" corresponds to the case with hyperbolic case (i.e. with 1 time dimension and 5-dimensional ambien space).


## NIPS co-authorship dataset

### Training 

To learn ultrahyperbolic representations, run the following code:
```
python nips_dataset_experiments.py
```

- We only considered the weighted graph setup for this experiment, so please keep the variable **weighted_version** to True except if you want to consider the unweighted graph setup. 

- To train with the optimizer introduced in Section 4.1, set the variable **apply_standard_sgd** to True. By default, the code optimizes the pseudo-Riemannian optimizer introduced in Section 4.2.

- To use the pseudo-Riemannian gradient (see Eq. (11) of the paper) as search direction, set the variable **use_pseudoRiemannian_gradient** to True. By default, the code uses the proposed descent direction introduced in Eq. (14) of the paper, except in the Riemannian case where the negative of the Riemannian gradient can be used as descent direction. 
When the metric tensor is not positive definite, the optimizer should not converge when the pseudo-Riemannian gradient is used as search direction as explained in the paper.

- Since the number of weaker pairs of nodes is too large to be stored in the memory of a GPU, the script randomly select **negative_batch_size** pairs of nodes that are not connected. 
We only considered **negative_batch_size = 42000** in our experiments because larger numbers could not fit into memory.

- The variables **space_dimensions** and **time_dimensions** can be set to different values. Please note that the variable **q** in the code corresponds to (q+1) in the paper.

- Depending on the dimensionality of the ambient space, the algorithm stops after **nb_max_iterations** iterations. We ran our experiments on a 12 GB NVIDIA TITAN V GPU. The algorithm takes about 1 hour to perform 1000 iterations.

### Evaluation

The MATLAB evaluation script is provided in the file "nips_evaluate_representation.m". It opens the embedding matrix files saved for 1 split and evaluates different metrics reported in the appendix.

- To evaluate Euclidean representations optimized with the squared Euclidean distance, set the variable **evaluate_euclidean_representations** to True. By default, the script considers pseudo-hyperbolic cases. Also choose the variable **euclidean_dimension** appropriately.

- Set the variables **time_dimensions** and **dimensionality_of_ambient_space ** to the appropriate number of time dimensions and dimensionality of the ambient space. For example, the directory "d_7_q_1" corresponds to the hyperbolic case (i.e. with 1 time dimension and 7-dimensional ambien space).



## Authors

* **Marc T. Law** - *NVIDIA* - http://www.cs.toronto.edu/~law/
* **Jos Stam** - *NVIDIA* - https://www.josstam.com/

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
