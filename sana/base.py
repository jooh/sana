"""code that does not explicitly depend on numpy or tensorflow. For convenience these
are aliased inside the tf and np modules as well."""


def n2npairs(n):
    """return number of pairs for n conditions. see also scipy.special.binom."""
    return (n - 1) * n / 2


def npairs2n(npairs):
    """return number of conditions for npairs. see also scipy.special.binom."""
    return ((8. * npairs + 1) ** .5 - 1) / 2 + 1
