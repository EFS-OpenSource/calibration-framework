# Multivariate Regression Calibration Example

In this package, we demonstrate the usage of the regression calibration methods with additional correlation estimation and recalibration.
The file *main.py* holds an artificial example that samples 2D data around a sine and cosine function with a certain mean and a variance.
Furthermore, the variance on the data is correlated to a certain degree.
The calibration methods can be used to identify and capture the correlations within the input data.
Thus, it is possible to model parametric multivariate distributions based on independently learned/provided distributions for each dimension,

Basically, each calibration method can be used to recalibrate multiple (independent) dimensions.
However, only the *GPNormal* is able to model a joint multivariate Gaussian with correlations as calibration output.
See the code example below:

```python
mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions

from netcal.regression import GPNormal

gpnormal = GPNormal(
    n_inducing_points=12,    # number of inducing points
    n_random_samples=256,    # random samples used for likelihood
    n_epochs=256,            # optimization epochs
    use_cuda=False,          # can also use CUDA for computations
    correlations=True,       # enable correlation capturing
    name_prefix="gpnormal",  # internal name prefix required for Pyro
)

# fit GP-Normal
gpnormal.fit((mean, stddev), ground_truth)

# transform distributions to obtain covariance matrices
cov = gpnormal.transform((mean, stddev))  # NumPy array with covariances - has shape (n, d, d)
```
