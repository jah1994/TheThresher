# Function for quickly computing the modified Bessel function of the first kind
# as implemented in Salahat et al. 2013 for approximating the
import torch

def i1_vectorised(argument, coeffs):

    z0 = argument[argument >= 0]
    z0 = z0[z0 <= 11.5]
    z0 = z0.reshape(len(z0), 1)

    z1 = argument[argument > 11.5]
    z1 = z1[z1 <= 20]
    z1 = z1.reshape(len(z1), 1)

    z2 = argument[argument > 20]
    z2 = z2[z2 <= 37.25]
    z2 = z2.reshape(len(z2), 1)

    z3 = argument[argument > 37.25]
    z3 = z3.reshape(len(z3), 1)

    a0, a1, a2, a3 = coeffs[:,0], coeffs[:,2], coeffs[:, 4], coeffs[:,6]
    b0, b1, b2, b3 = coeffs[:,1], coeffs[:,3], coeffs[:, 5], coeffs[:,7]

    def linalg(z, a, b):
        expz = torch.exp(z @ b.reshape(1, len(b)))
        ab = a*b
        out = expz @ ab
        return out

    out0, out1, out2, out3 = linalg(z0, a0, b0), linalg(z1, a1, b1), linalg(z2, a2, b2), linalg(z3, a3, b3)

    out = torch.cat((out0, out1, out2, out3))

    return out

# fit coefficients from Salahat et al. 2013
def return_i1_coefficients():
    coeffs = torch.Tensor([[0.1682, 0.7536, 0.2667, 0.4710, 0.1121, 0.9807, 2.41e-9, 1.144],
                       [0.1472, 0.9739, 0.4916, -163.40, 0.1055, 0.8672, 0.06745, 0.995],
                       [0.4450, -0.715, 0.1110, 0.9852, -0.00018, 1.0795, 0.05471, 0.5686],
                       [0.2382, 0.2343, 0.1304, 0.8554, 0.00326, 1.0385, 0.07869, 0.946]])
    return coeffs



### Optional - Estimate parameter uncertanties via Fisher Information
def compute_full_hessian(grads):

  #Note the use of .detach(). In general, computations involving
  #variables that require gradients will keep history.

  grads = torch.cat((grads[0].flatten(), grads[1].flatten()))
  grad = grads.reshape(-1)
  d = len(grad)
  H = torch.zeros((d, d))

  t = time.time()
  print('Looping...')
  for i,dl_dthetai in enumerate(grads):
    H_rowi = torch.autograd.grad(dl_dthetai, model.parameters(), retain_graph=True)
    H_rowi = torch.cat((H_rowi[0].flatten(), H_rowi[1].flatten()))
    H[i] = H_rowi.detach()
  print('Looping took %s seconds' % (time.time() - t))
  return H


def compute_hessian_blockdiags(grads, params):
  H = []
  t = time.time()
  print('Looping...')
  for i, (grad, p) in enumerate(zip(grads, params)):
      grad = grad.reshape(-1)
      d = len(grad)
      dg = torch.zeros((d, d))

      for j, g in enumerate(grad):
          g2 = torch.autograd.grad(g, p, create_graph=True)[0].view(-1)
          dg[j] = g2.detach()

      H.append(dg)
  print('Looping took %s seconds' % (time.time() - t))
  return H


'''
We're minimizing an approximation to the negative log-likelihood
So the returned Hessian is equivalent to the observed Fisher
Information Matrix (i.e. FIM evaluated at MLE)
'''

def det_test(matrix):
    sign, logdet = torch.slogdet(matrix)
    if sign.item() <= 0.:
      print('Covariance matrix not positive definite! Sign of determinant:', sign.item())
    elif sign.item() > 0.:
      pass

def eigenvals_test(matrix):
    eigenvals = torch.eig(matrix)[0]
    if any(eigval <= 0. for eigval in eigenvals[:,0]):
      print('Covariance matrix not positive definite!. Non-positive eigenvalues.1')


def get_stderror(obs_fisher_matrix):
    '''
    Is the estiamted covariance matrix valid?
    A valid covariance matrix has to be positive definite
    Test 1:
    check if det cov_matrix <= 0., cov_matrix is not valid
    Test 2:
    diagnoalise cov_matrix to determine eigenvalues.
    If any of these are <= 0., cov_matrix is not valid
    '''
    cov_matrix = torch.inverse(obs_fisher_matrix)
    #print('Covariance Matrix:', cov_matrix)
    det_test(cov_matrix) # Test 1
    eigenvals_test(cov_matrix) # Test 2
    cov_matrix_diagonals = torch.diag(cov_matrix)

    return cov_matrix_diagonals


# Adam update rule
def Adam(t, g, m, v, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon = 1e-8):

    t += 1

    # update biased 1st and 2nd moment vectors
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)

    # compute bias-corrected 1st and 2nd moment vectors
    m_ = m / (1 - beta1 ** t)
    v_ = v / (1 - beta2 ** t)

    update = alpha * (m_ / torch.sqrt(v_) + epsilon)

    return update, m, v
