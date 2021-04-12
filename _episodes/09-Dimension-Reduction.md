---
title: "Dimension Reduction"
teaching: 20
exercises: 0
questions:
- "What happen when there are lots of covariates?"
objectives:
- "Learn how to apply PCA in ML model"
keypoints:
- "PCA"
---

## Principal Component Analysis
- Handy with large data
- Where many variables correlate with one another, they will all contribute strongly to the same principal component
- Each principal component sums up a certain percentage of the total variation in the dataset
- More Principal Components, more summarization of the original data sets

### Explanation
- For example, we have 3 data sets: `X, Y, Z`
- We need to compute the covariance matrix **M** for the 3 data set:
![image](https://user-images.githubusercontent.com/43855029/114459677-d67c0980-9bae-11eb-85b2-758a98f0cd29.png)

in which, the covariance value between 2 data sets can be computed as:
![image](https://user-images.githubusercontent.com/43855029/114459740-ea277000-9bae-11eb-9259-8ef1b233c0fa.png)

- For the Covariance matrix **M**, we will find **m** eigenvectors and **m** eigenvalues

```
- Given mxm matrix, we can find m eigenvectors and m eigenvalues
- Eigenvectors can only be found for square matrix.
- Not every square matrix has eigenvectors
- A square matrix A and its transpose have the same eigenvalues but different eigenvectors
- The eigenvalues of a diagonal or triangular matrix are its diagonal elements.
- Eigenvectors of a matrix A with distinct eigenvalues are linearly independent.
```

**Eigenvector with the largest eigenvalue forms the first principal component of the data set
… and so on …***
