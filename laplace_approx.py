def laplace_linear_posterior(X, y, prior_var=np.inf):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y

    # Prior precision Lambda
    if np.isinf(prior_var):
        Lambda_0 = np.zeros((p, p)) # assuming a flat prior, Lambda_0 = 0
    else:
        Lambda_0 = np.eye(p) / float(prior_var) # otherwise, Isotropic Gaussian prior (Bishop, 2006)

    # Initial OLS estimate for sigma^2
    beta_ols = np.linalg.pinv(XtX) @ Xty
    resid = y - X @ beta_ols
    df = n - p
    sigma2_hat = float(resid @ resid / df)

    # MAP estimate under Gaussian prior (reduces to OLS when Lambda=0)
    Lambda_N = XtX / sigma2_hat + Lambda_0
    Sigma_N = np.linalg.pinv(Lambda_N) # posterior covariance = inverse Hessian at the mode
    beta_hat = Sigma_N @ (Xty / sigma2_hat)
    return beta_hat, Sigma_N