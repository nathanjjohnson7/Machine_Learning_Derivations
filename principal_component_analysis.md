# Principal Component Analysis (PCA)

Data matrix: 

$$X \in \mathbb{R}^{F \times N}$$

Sample Mean:

$$\bar{x} = \frac{1}{N}\sum_{i=1}^{N}x^{(i)}$$

Sample Variance:

$$cov(X)\ =\frac{1}{N-1}\sum_{i=1}^{N} \left(x^{(i)}-\bar{x}\right)\left(x^{(i)}-\bar{x}\right)^T$$

where $$x^{(i)}$$ refers to the i-th sample of shape Fx1, N = number of samples

Succinct Sample Mean Notation:

$$\bar{x} = \frac{1}{N}X\mathbf{1}_N$$

We want to find a matrix $$W\in\mathbb{R}^{F\times K}$$, where K < F,

$$Y = W^TX$$

$$Y\in\mathbb{R}^{K\times N}$$

Although we are reducing the number of dimensions, we want maximize the covariance of Y so the relationships between variables in X are preserved.

$$cov(Y) = cov(W^TX)$$
$$cov(Y) = \frac{1}{N-1}\sum_{i=1}^{N} \left(W^Tx^{(i)}-\frac{1}{N}W^TX\mathbf{1}_N\right)\left(W^Tx^{(i)}-\frac{1}{N}W^TX\mathbf{1}_N\right)^T$$

$$cov(Y) = \frac{1}{N-1}\sum_{i=1}^{N} W^T\left(x^{(i)}-\frac{1}{N}X\mathbf{1}_N\right)\left(x^{(i)}-\frac{1}{N}X\mathbf{1}_N\right)^TW$$

$$cov(Y) = W^T\left[\frac{1}{N-1}\sum_{i=1}^{N} \left(x^{(i)}-\frac{1}{N}X\mathbf{1}_N\right)\left(x^{(i)}-\frac{1}{N}X\mathbf{1}_N\right)^T\right]W$$

$$cov(Y) = W^Tcov(X)W$$

We want to find the W that maximizes the covariance of Y

$$\arg\max_{w} W^Tcov(X)W$$

We need a constraint to ensure that $$W^T$$ doesn't grow to infinity as we seek the maxmize the expression.

$$\arg\max_{w} W^Tcov(X)W \qquad s.t. \ W^TW=I$$

Lagrangian Formulation for the i-th column of W:

Note: This is equivalent to finding the optimal W where K = 1 (W is of size Fx1), which would reduce the Y matrix to 1 feature (size 1xN):

$$\text{Constraint: } w^{{(i)}^T}w^{(i)}-1=0$$

$$L(w^{(i)}, \lambda) = w^{{(i)}^T}cov(X)w^{(i)} - \lambda (w^{{(i)}^T}w^{(i)}-1)$$

Where $$w^{(i)}$$ is the i-th column of W and $$\lambda$$ is a langrangian multiplier

We set the partial derivative of the langrangian w.r.t $$w^{(i)}$$ to 0:

$$\frac{\partial L(w^{(i)}, \lambda)}{\partial w^{(i)}} = 2cov(X)w^{(i)} - 2\lambda w^{(i)} = 0$$

$$cov(X)w^{(i)} = \lambda w^{(i)}$$

This equation is satisified when $$w^{(i)}$$ is the i-th eigenvector of cov(X) and $$\lambda$$ is the i-th eigenvalue of cov(X).

Combining all the column vectors of W:

$$cov(X)W = W\Lambda$$

This equation is satisified when the columns of W are the eigenvectors of cov(X) and the diagonal of $$\Lambda$$ holds the corresponding eigenvalues of cov(X).

Since the covariance matrix is always symmetric (symmetric matrices are always diagonalizable) and if the columns of $$W_F$$ hold all F eigenvectors of cov(x) and the diagonal of $$\Lambda_F$$ holds all F eigenvalues of cov(x), cov(x) can be factored as:

$$cov(X) = W_F \Lambda_F W_F^{-1}$$


Since W holds the eigenvectors of a symmetric matrix, W is orthogonal. 
Hence, 

$$W_F^{-1} = W_F^T$$
$$cov(X) = W_F \Lambda W_F^T$$

Therefore,

$$cov(Y) = W_F^Tcov(X) W_F = W_F^T(W_F \Lambda_F W_F^T)W_F = (W_F^TW_F) \Lambda_F (W_F^TW_F) = I\Lambda_F I = \Lambda_F$$

The maximum covariance of Y is the diagonal matrix of all the eigenvalues of the cov(X) and is achieved when X is projected onto all eigenvectors of cov(X).

However, we are trying to reduce the dimensions of Y, so we don't use all the eigenvectors but the top-k. 

$$W\in\mathbb{R}^{F\times K}, \Lambda \in \mathbb{R}^{K \times K} \ is \ diagonal$$
$$cov(X) \approx W \Lambda W^T$$
$$cov(Y) \approx W^Tcov(X) W = W^T(W \Lambda W^T)W = (W^TW) \Lambda (W^TW) = I\Lambda I = \Lambda$$

The maximum covariance for Y, of size KxN, is equal to the diagonal matrix of the top-K eigenvalues of cov(X), and is obtained when X is projected onto the top-K eigenvectors of cov(X).

$$W \in \mathbb{R}^{F \times K} \text{ contains the top k eigenvectors of cov(X)}$$

Side Note: top-k refers to the k eigenvectors with the highest eigenvalues. The eigenvalue of an eigenvector is the covariance of Y if projected exclusively onto that singular eigenvector.

Going back to the case where K=1, and W is of size Fx1 and Y is of size 1xN:

$$cov(X)w^{(i)} = \lambda w^{(i)}$$
$$cov(Y) = w^{{(i)}^T}cov(X) w^{(i)} = w^{{(i)}^T}\lambda w^{(i)} = \lambda (w^{{(i)}^T}w^{(i)}) = \lambda 1 = \lambda$$

This shows that the eigenvalue is the covariance, so to get the highest covariance with K > 1 we have to use the eigenvectors with the highest eigenvalues.

IMPORTANT NOTE: Before being projected onto W, X must be mean-centered by subtracting the sample mean. W captures the most variance of X . This variance has been computed relative to the mean. Thus, X must be mean-centered to obtain accurate results.
