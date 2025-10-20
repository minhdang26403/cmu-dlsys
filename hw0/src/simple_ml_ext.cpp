#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;

void matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t p) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * p + j];
      }
      C[i * p + j] = sum;
    }
  }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t num_iterations = (m + batch - 1) / batch;
    for (size_t iter = 0; iter < num_iterations; iter++) {
      // The size of the last batch may be less than previous batches
      size_t cur_batch = std::min(batch, m - iter * batch);
      
      const float *x_batch = &X[iter * batch * n];
      const unsigned char *y_batch = &y[iter * batch];
      std::vector<float> Z(cur_batch * k);

      // 1. Compute logits: Z = x_batch @ theta
      for (size_t i = 0; i < cur_batch; i++) {
        for (size_t j = 0; j < k; j++) {
          float sum = 0;
          for (size_t l = 0; l < n; l++) {
            sum += x_batch[i * n + l] * theta[l * k + j];
          }
          Z[i * k + j] = sum;
        }
      }

      // 2. Compute softmax probabilities: Z = normalize(exp(Z))
      for (size_t i = 0; i < cur_batch; i++) {
        float row_sum = 0;
        for (size_t j = 0; j < k; j++) {
          Z[i * k + j] = exp(Z[i * k + j]);
          row_sum += Z[i * k + j];
        }
        for (size_t j = 0; j < k; j++) {
          Z[i * k + j] /= row_sum;
        }
      }

      // 3. Subtract one-hot encoding: Z = Z - I_y
      for (size_t i = 0; i < cur_batch; i++) {
        Z[i * k + y_batch[i]] -= 1;
      }

      // 4. Compute gradient without explicit transpose
      // grad = (1/m) * x_batch_T @ Z
      std::vector<float> grad(n * k);
      for (size_t i = 0; i < cur_batch; i++) {
        for (size_t j = 0; j < n; j++) {
          for (size_t l = 0; l < k; l++) {
            grad[j * k + l] += x_batch[i * n + j] * Z[i * k + l];
          }
        }
      }

      // 5. Update theta
      for (size_t i = 0; i < n * k; i++) {
        theta[i] -= (lr / cur_batch) * grad[i];
      }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
