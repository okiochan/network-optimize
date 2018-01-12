from numpy.linalg import norm

#righest that equals zero
def bins(f, l, r, eps=1e-7):
    d = r-l
    if f(r) < 0: return r
    while d > eps:
        m = l+d/2
        if f(m) < 0: l += d/2
        d /= 2
    return l + d/2

def optimize(f, g, x0, maxiter=2000, gtol=1e-6, verbose=True, printEvery=50):
    x = x0.copy()
    n = x0.size

    for iter in range(maxiter):
        grad = g(x)
        gradnorm = norm(grad)

        if iter == 0:
            d = -grad
        else: # conjugate direction
            den = d_1.dot(grad-grad_1) # Hestenes-Stiefel formula ( grad . (grad-grad_1) / d_1 . (grad-grad_1) )
            if abs(den)<gtol:
                d = -grad
            else:
                num = grad.dot(grad-grad_1)
                beta=num/den
                d = -grad + beta * d_1

        # ensure descent direction
        if d.dot(-grad) <= 0:
            d = -grad

        phi = lambda a: f(x + a*d)
        dphi = lambda a: g(x + a*d).dot(d)
        
        low = 0
        high = 0.01
        phi_low = phi(low)
        phi_high = phi(high)
        while phi_high < phi_low:
            low, phi_low = high, phi_high
            high *= 2.5
            phi_high = phi(high)
        alpha = bins(dphi, low, high, gtol)

        x_1 = x
        d_1 = d
        grad_1 = grad

        x = x + alpha * d
        
        if verbose and iter % printEvery == 0:
            print(iter, gradnorm, f(x))

    return x