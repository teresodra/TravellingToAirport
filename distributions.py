import numpy as np

class DistApprox:
    """
    1D distribution approximated on a uniform grid.
    Attributes:
        grid : np.ndarray of x values (shape (N,))
        pdf  : np.ndarray of f(x) values, sum(pdf*dx) ≈ 1
    """
    def __init__(self, grid, pdf):
        self.grid = np.asarray(grid, dtype=float)
        self.pdf  = np.asarray(pdf, dtype=float)

        assert self.grid.ndim == 1
        assert self.pdf.ndim  == 1
        assert len(self.grid) == len(self.pdf)
        assert len(self.grid) >= 2

        self.dx = float(self.grid[1] - self.grid[0])

    # ---------- basic helpers ----------

    def cdf(self):
        """Cumulative distribution F(x) on the same grid."""
        return np.cumsum(self.pdf) * self.dx

    def resample(self, new_grid):
        """
        Resample this distribution onto a new grid using CDF interpolation.
        This lets us combine distributions defined on slightly different grids.
        """
        new_grid = np.asarray(new_grid, dtype=float)
        dx_new   = float(new_grid[1] - new_grid[0])

        F = self.cdf()
        # Interpolate CDF; outside original range assume 0/1.
        F_new = np.interp(new_grid, self.grid, F, left=0.0, right=1.0)

        pdf_new = np.diff(F_new, prepend=0.0) / dx_new
        pdf_new = np.maximum(pdf_new, 0.0)    # remove tiny negatives
        pdf_new /= (pdf_new.sum() * dx_new)   # renormalise

        return DistApprox(new_grid, pdf_new)

    # ---------- constructors ----------

    @staticmethod
    def from_normal(mean=0.0, std=1.0, x_min=None, x_max=None, n_points=2001):
        """
        Approximate N(mean, std^2) on a uniform grid.

        Choose x_min/x_max so that most mass lies inside, e.g.
        x_min = mean - 6*std, x_max = mean + 6*std.
        For combining many dists, use the same x_min/x_max/n_points everywhere.
        """
        mean = float(mean)
        std  = float(std)
        if std == 0:
            return ConstDist(mean)

        if x_min is None:
            x_min = mean - 6 * std
        if x_max is None:
            x_max = mean + 6 * std

        grid = np.linspace(x_min, x_max, n_points)
        dx   = grid[1] - grid[0]

        pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * \
              np.exp(-0.5 * ((grid - mean) / std) ** 2)

        pdf /= (pdf.sum() * dx)  # normalise numerically

        return DistApprox(grid, pdf)

    # ---------- operations: sum and min ----------

    def add(self, other):
        """
        Distribution of Z = X + Y for independent X,Y.

        Uses discrete convolution and then resamples back onto self.grid.
        """
        if isinstance(other, (int, float, ConstDist)):
            # Shift distribution by a constant
            shifted_grid = self.grid + float(other)
            return DistApprox(shifted_grid, self.pdf.copy())
        # Make sure we have the same spacing; resample if needed.
        if not np.isclose(self.dx, other.dx):
            other = other.resample(self.grid)

        dx = self.dx
        # Convolution of PDFs approximates sum distribution.
        pdf_conv = np.convolve(self.pdf, other.pdf) * dx
        g0       = self.grid[0] + other.grid[0]
        grid_conv = g0 + dx * np.arange(len(pdf_conv))

        conv_dist = DistApprox(grid_conv, pdf_conv)
        # Resample back onto the original grid range
        return conv_dist.resample(self.grid)

    def min_with(self, other):
        """
        Distribution of Z = min(X, Y) for independent X,Y.
        Uses F_Z = 1 - (1-F_X)(1-F_Y) on a common grid.
        """
        # Put both distributions on the same grid
        if not np.allclose(self.grid, other.grid):
            other = other.resample(self.grid)

        F1 = self.cdf()
        F2 = other.cdf()
        dx = self.dx

        F_min = 1.0 - (1.0 - F1) * (1.0 - F2)
        pdf_min = np.diff(F_min, prepend=0.0) / dx
        pdf_min = np.maximum(pdf_min, 0.0)
        pdf_min /= (pdf_min.sum() * dx)

        return DistApprox(self.grid.copy(), pdf_min)
    
    def prob_between(self, a=None, b=None):
        """
        Compute:
          - P(a <= X <= b)       if both a and b are given
          - P(X >= a)            if only a is given
          - P(X <= b)            if only b is given

        Uses linear interpolation of the CDF.
        """
        F = self.cdf()

        # ---- Case 1: only a given -> P(X >= a) = 1 - F(a)
        if a is not None and b is None:
            Fa = np.interp(a, self.grid, F, left=0.0, right=1.0)
            return max(1.0 - Fa, 0.0)

        # ---- Case 2: only b given -> P(X <= b) = F(b)
        if a is None and b is not None:
            Fb = np.interp(b, self.grid, F, left=0.0, right=1.0)
            return max(Fb, 0.0)

        # ---- Case 3: a and b both given -> P(a <= X <= b)
        if a is not None and b is not None:
            if b < a:
                return 0.0

            Fa = np.interp(a, self.grid, F, left=0.0, right=1.0)
            Fb = np.interp(b, self.grid, F, left=0.0, right=1.0)
            return max(Fb - Fa, 0.0)

        # ---- Case 4: neither argument provided -> not meaningful
        raise ValueError("Provide at least one bound (a or b).")
    

class ConstDist:
    """Distribution that is a constant value."""
    def __init__(self, value):
        self.value = float(value)

    def __float__(self):
        return self.value

    def prob_between(self, a=None, b=None):
        if a is not None and b is not None:
            return 1.0 if a <= self.value <= b else 0.0
        elif a is not None:
            return 1.0 if self.value >= a else 0.0
        elif b is not None:
            return 1.0 if self.value <= b else 0.0
        else:
            raise ValueError("Provide at least one bound (a or b).")


def sum_distributions(*args):
    """Given any number of distributions, these are added up."""
    if len(args) == 0:
        raise ValueError("At least one distribution is required.")
    
    distributions = []
    
    constant_sum = 0
    for arg in args:
        if isinstance(arg, (int, float, ConstDist)):
            constant_sum += float(arg)
        else:
            distributions.append(arg)
    
    if not distributions:
        return ConstDist(constant_sum)
    
    if len(distributions) >= 1:
        result = distributions[0]
        for dist in distributions[1:]:
            result = result.add(dist)
        if constant_sum != 0:
            result = result.add(constant_sum)
    else:
        result = constant_sum

    return result

def min_distributions(*args):
    """Given any number of distributions, returns the distribution of their minimum."""
    if len(args) == 0:
        raise ValueError("At least one distribution is required.")
    result = args[0]
    for dist in args[1:]:
        result = result.min_with(dist)
    return result

