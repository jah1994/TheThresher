# Function for quickly computing the modified Bessel function of the first kind
# as implemented in Salahat et al. 2013 for approximating the
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i1
from scipy import stats
import config


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

### NQR ####
sigma, f, gain, c, q = config.EMCCD_params

class PGN_gen(stats.rv_continuous):
    "PGN distribution"
    def _pdf(self, x, n):

        g = f*x

        #### this bit deals with the PDF evaluation (arrays)
        try:
            g_pos = g[g>=0]
            pdf_readout = np.exp(-n) * (1./np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*(g/sigma)**2)
            pdf_EM = np.exp(-n - (g_pos/gain)) * np.sqrt(n/(g_pos*gain)) * i1(2*np.sqrt((n*g_pos)/gain))
            pdf_pos = pdf_readout[g >= 0] + pdf_EM
            pdf_neg = pdf_readout[g < 0]
            pdf = f*np.concatenate((pdf_neg, pdf_pos)) # convert to 1/ADU = (e-_EM/ADU) * (1/e-_EM)

        #### ... and this bit with the CDF evaluation (floats)
        except TypeError:
            pdf_readout = np.exp(-n) * (1./np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*(g/sigma)**2)
            if g > 0:
                pdf_EM = np.exp(-n - (g/gain)) * np.sqrt(n/(g*gain)) * i1(2*np.sqrt((n*g)/gain))
                pdf = f * (pdf_EM + pdf_readout)
            else:
                pdf = f * pdf_readout


        return pdf

def compute_nqr(data, model):

    # evaluate cdf at MLE for n
    PGN = PGN_gen(name='PGN')
    ms = model.detach().cpu().numpy().flatten()
    ys = data.detach().cpu().numpy().flatten()
    cdf_vals = []
    for m, y_pix in zip(ms, ys):
        n = (f/gain) * m
        cdf_val = PGN.cdf(y_pix, n)
        cdf_vals.append(cdf_val)

    cdf = np.array(cdf_vals)
    r = stats.norm.ppf(cdf)

    image = ys.reshape(data.size())
    model = ms.reshape(data.size())
    residuals = r.reshape(data.size())

    r = r[~np.isinf(r)] # hack away any infs

    ks_stat, pvalue = stats.kstest(r, 'norm') # quantify with a KS test

    plt.figure(figsize=(10, 10))
    grid = np.linspace(-5, 5, 100)
    plt.hist(r, density=True)[2]
    plt.plot(grid, stats.norm.pdf(grid, 0, 1))
    plt.xlabel('NQR')
    plt.ylabel('Probability')
    plt.grid()
    plt.title('KS test: statistic=%.4f, pvalue=%.4f' % (ks_stat, pvalue))
    plt.savefig(os.path.join(config.out_path, 'plots', 'NQR.png'), bbox_inches='tight');


######################
