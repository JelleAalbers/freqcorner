Freqcorner
===========

Jealous of Bayesian [corner plots](https://corner.readthedocs.io/en/latest/index.html), but can't choose a prior? You came to the right place.

```python
import freqcorner

def log_likelihood(a, b):
    # Uncorrelated bivariate normal
    return -0.5 * ( ((a - 1)/2)**2 + (b - 3)**2 )

freqcorner.corner(log_likelihood, guess=dict(a=1, b=1))
```


<img src="https://raw.githubusercontent.com/JelleAalbers/freqcorner/main/demo_image.png" width="500">


This displays a frequentist version of a corner plot, with prior-free asymptotic confidence intervals and contours, rather than marginals of a posterior.

In more detail:
  * The diagonal plots show:
    * In gray, the [confidence distributions](https://en.wikipedia.org/wiki/Confidence_distribution) derived from the [profile likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Profile_likelihood). Horizontal slices through this are confidence intervals of different levels; the 68% and 90% confidence intervals are plotted by default.
    * Faintly, the confidence distribution if all other parameters are held fixed / assumed known at the best fit, rather than profiled over. If the parameters are uncorrelated, these are the same as the regular distribution.
  * Off-diagonal plots show:
      * 68% and 90% confidence regions derived from the the profile likelihood.
      * Fainly, the same confidence regions if all other parameters are held fixed (rather than profiled over). If the parameters are uncorrelated, or you only have two parameters, these are identical to the regular contours.


 This package uses [iminuit](https://github.com/scikit-hep/iminuit) to minimize the likelihood; in particular, the [minos](https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.Minuit.minos) and[mncontour](https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.Minuit.mncontour) methods are used to find confidence intervals/contours.