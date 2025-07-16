# Deriving the Update Rule for a Linear Classfier + Softmax Activation + Cross-Entropy Loss

The equation of a line:

$$\hat{y} = wx + b$$

$$x \in \mathbb{R}^{F}, \hat{y} \in \mathbb{R}^{C}, w \in \mathbb{R}^{C \times F}, b \in \mathbb{R}^{C}$$
$$\text{where } F =\text{num features }, C =\text{num classes }, x = \text{inputs}, \hat{y} = \text{predictions}, w = \text{weight matrix}, \ b = \text{bias} $$

The output should be a probability distribution over the number of classes. So we apply a softmax function.

$$z = wx + b$$
$$s = \text{softmax}\left(z\right)$$
$$\text{where} \ s_i = \frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}} $$

We calculate the error in the classifier's prediction using Cross Entropy Loss:

$$L = - \sum_{n=1}^{C}y_n\log s_n$$
$$ \text{where} \ y = \text{correct probability distribution}$$

$$ \text{We must: } \min_{w, \ b} L$$

To do this, we find the gradient of the loss function, L, w.r.t. to the parameters, w and b. The gradients tell us the slope of the loss function for a given w and b. We follow this slope downwards until we reach the lowest point of the loss function, the minimum.

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial s} \frac{\partial s}{\partial z} \frac{\partial z}{\partial w}$$
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial s} \frac{\partial s}{\partial z} \frac{\partial z}{\partial b}$$

Using the definition of the loss given above, the partial derivative w.r.t. s:

$$\frac{\partial L}{\partial s} = \left(\frac{-y}{s}\right)^T $$

We transpose because we are using numerator layout notation, where the derivative of a scalar with respect to a vector is a scalar (row vector).

The partial derivative of s w.r.t. z will be a matrix. For an s of length 3:

$$\frac{\partial s}{\partial z} = \begin{bmatrix} \frac{\partial s_1}{\partial z_1} & \frac{\partial s_1}{\partial z_2} & \frac{\partial s_1}{\partial z_3} \\\ \frac{\partial s_2}{\partial z_1} & \frac{\partial s_2}{\partial z_2} & \frac{\partial s_2}{\partial z_3}  \\\ \frac{\partial s_3}{\partial z_1} & \frac{\partial s_3}{\partial z_2} & \frac{\partial s_3}{\partial z_3} \end{bmatrix}$$

Using the definition of $$s_i$$ given above:

$$ \text{where } i = j, \ \frac{\partial s_i}{\partial z_j} = \frac{\partial s_i}{\partial z_i} = \frac{e^{z_i}\left(\sum_{j=1}^{C}e^{z_j}\right) \ - (e^{z_i})^2}{\left(\sum_{j=1}^{C}e^{z_j}\right)^2}$$
$$= \frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}\left(\frac{\left(\sum_{j=1}^{C}e^{z_j}\right) \ -e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}\right)$$
$$= \frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}\left(1 \ -\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}\right)$$

$$\text{The astute reader will notice: }$$

$$= s_i(1 - s_i) $$

$$ \text{where } i \neq j, \ \frac{\partial s_i}{\partial z_j} = \frac{-e^{z_i}e^{z_j}}{\left(\sum_{j=1}^{C}e^{z_j}\right)^2}$$

$$\text{Again, the astute reader will notice: }$$

$$= -s_is_j $$

$$\text{If } s \text{ is a vector of length 3:}$$
$$\frac{\partial s}{\partial z} = \begin{bmatrix} s_1(1-s_1) & -s_1s_2 & -s_1s_3 \\\ -s_2s_1 & s_2(1-s_2) & -s_2s_3  \\\ -s_3s_1 & -s_3s_2 & s_3(1-s_3)\end{bmatrix}$$
$$\frac{\partial s}{\partial z} = Diag(s) - ss^T$$


$$\text{We can now find the derivative of the loss w.r.t. z}$$
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial s} \frac{\partial s}{\partial z}$$
$$=\frac{-y^T}{s^T}(Diag(s) - ss^T)$$
$$=-y^T(I_{|s|} - 1_{|s|}s^T)$$
$$= y^T1_{|s|}s^T - y^TI_{|s|}$$

Since y is a one-hot encoding vector (all zeros and a singular 1)

$$\frac{\partial L}{\partial z}=s^T\ -y^T = s\ -y$$

We find the derivative of z w.r.t. w and b:

$$\frac{\partial z}{\partial w}= x$$
$$\frac{\partial z}{\partial b}= 1$$

We find the derivative of the loss w.r.t. w and b:

$$\frac{\partial L}{\partial w}= \frac{\partial L}{\partial z}\frac{\partial z}{\partial w} = (s-y)x^T$$
$$\frac{\partial L}{\partial b}= \frac{\partial L}{\partial z}\frac{\partial z}{\partial b} = s-y$$

If we add an L2 regularization term to the loss:

$$L = - \sum_{n=1}^{C}y_n\log s_n + \frac{\lambda}{2}\sum_{i}^{}\sum_{j}^{} w_{ij}^2$$
$$\frac{\partial L}{\partial w} = \frac{\partial \left[- \sum_{n=1}^{C}y_n\log s_n\right]}{\partial w} + \frac{\partial \left[\frac{\lambda}{2}\sum_{i}^{}\sum_{j}^{} w_{ij}^2\right]}{\partial w}$$
$$=(s-y)x^T \ + \lambda w$$

The derivative of the loss w.r.t. b is unchanged.

Update rules:

$$w = w - \alpha\left((s-y)x^T \ + \lambda w\right)$$
$$b = b - \alpha\left(s-y\right)$$

where $$\alpha$$ is the learning rate.
