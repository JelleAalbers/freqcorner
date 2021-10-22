import itertools

import iminuit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats


DEFAULT_FAINT_ALPHA = 0.1
DEFAULT_TRUTH_COLOR = "darkorange"
DEFAULT_CLS = (
    stats.norm.cdf(1) - stats.norm.cdf(-1),
    0.9)
DEFAULT_CONTOUR_COLORS = ("fuchsia", "purple", "crimson")
DEFAULT_AX_SIZE = 2.5


def corner(
    log_likelihood,
    guess=None,
    truth=None,
    cls=DEFAULT_CLS,
    joint_coverage=False,
    limits=None,
    progress=True,
    scales=None,
    ranges=None,
    scaled_ranges=None,
    domain_n_sigma=5,
    n_points=30,
    ax_size=DEFAULT_AX_SIZE,
    labels=None,
):
    if guess is None:
        assert truth is not None, "Provide guess or truth"
        guess = truth
    params = list(guess.keys())

    results = corner_data(
        log_likelihood,
        guess=guess,
        cls=cls,
        joint_coverage=joint_coverage,
        limits=limits,
        progress=progress,
        scales=scales,
        ranges=ranges,
        scaled_ranges=scaled_ranges,
        domain_n_sigma=domain_n_sigma,
        n_points=n_points,
    )

    return plot_corner(
        params, off_diag, diag, truth=truth, labels=labels, ax_size=ax_size, **results
    )


def corner_data(
    log_likelihood,
    guess,
    cls=DEFAULT_CLS,
    joint_coverage=False,
    limits=None,
    progress=True,
    scales=None,
    ranges=None,
    scaled_ranges=None,
    domain_n_sigma=5,
    n_points=30,
):
    params = list(guess.keys())
    n_par = len(params)
    if limits is None:
        limits = dict()
    if scales is None:
        scales = dict()
    scaled_ranges = get_scaled_ranges(
        params, scales=scales, ranges=ranges, scaled_ranges=scaled_ranges
    )

    # Setup Minuit objective
    def objective(param_values):
        return -log_likelihood(**dict(zip(params, param_values)))

    m = iminuit.Minuit(objective, np.array(list(guess.values())), name=params)
    m.errordef = m.LIKELIHOOD
    for pname, lims in limits.items():
        m.limits[pname] = lims

    # Find bestfit
    migrad_output = m.migrad()
    bestfit = dict(zip(params, m.values))

    # Find MINOS errors - profile likelihood confidence intervals
    minos_errors = dict()
    for cl in cls:
        if joint_coverage:
            _minuit_cl = minuit_hardcoded_dof_workaround(
                cl=cl, real_dof=n_par, hardcoded_dof=1
            )
        else:
            _minuit_cl = cl
        m.minos(cl=_minuit_cl)
        minos_errors[cl] = {k: (v.lower, v.upper) for k, v in m.merrors.items()}

    # Determine unspecified parameter ranges using these errors
    for pname in params:
        if scaled_ranges[pname] == (None, None):
            min_, max_ = limits.get(pname, (-float("inf"), float("inf")))
            low, high = (
                bestfit[pname] - domain_n_sigma * m.errors[pname],
                bestfit[pname] + domain_n_sigma * m.errors[pname],
            )
            # If one of the values has to be clipped, extend the other.
            # (If both are out of range, both will be clipped)
            if low < min_:
                high += min_ - low
                low = min_
            if high > max_:
                low -= high - max_
                high = max_
            scaled_ranges[pname] = (
                max(min_, low) * scales.get(pname, 1),
                min(max_, high) * scales.get(pname, 1),
            )

    # Compute likelihoods on param grids
    results = dict()
    for par_i, px in enumerate((tqdm if progress else lambda x: x)(params)):
        for par_j, py in enumerate(params):
            if par_i > par_j:
                results[(px, py)] = results[(py, px)]
            elif px == py:

                # Simple log likelihood scan, all other params fixed
                x = np.linspace(*scaled_ranges[px], num=n_points)
                z = np.asarray([-log_likelihood(**{**bestfit, **{px: _x}}) for _x in x])

                # Profile likelihood scan, all other params optimized
                mnprofile = dict(
                    zip(
                        "locations fvals status".split(),
                        m.mnprofile(px, size=n_points, bound=scaled_ranges[px]),
                    )
                )

                results[(px, px)] = dict(mnprofile=mnprofile, x=x, z=z)

            else:
                # Simple log likelihood scan, all parameters but two fixed
                x, y, z = m.contour(
                    px, py, bound=(scaled_ranges[px], scaled_ranges[py]), size=n_points
                )
                z *= -1

                # Profile likelihood contour, all other params optimized
                mncontours = dict()
                for cl in cls:
                    if joint_coverage:
                        _minuit_cl = minuit_hardcoded_dof_workaround(
                            cl=cl, real_dof=n_par, hardcoded_dof=2
                        )
                    else:
                        _minuit_cl = cl
                    mncontours[cl] = m.mncontour(px, py, size=n_points, cl=_minuit_cl)

                results[(px, py)] = dict(x=x, y=y, z=z, mncontours=mncontours)

    return dict(
        results_for_params=results,
        bestfit=bestfit,
        cls=cls,
        guess=guess,
        joint_coverage=joint_coverage,
        minos_errors=minos_errors,
        migrad_output=migrad_output,
        scaled_ranges=scaled_ranges,
        minuit=m,
    )


def diag(
    *,
    ax,
    bestfit,
    pname,
    scale,
    truth,
    guess,
    label,
    joint_coverage,
    minos_errors,
    cls,
    results_for_params,
    contour_colors=DEFAULT_CONTOUR_COLORS,
    truth_color=DEFAULT_TRUTH_COLOR,
    title_cl=None,
    title_formats=None,
    tick_conf_levels=None,
    faint_alpha=DEFAULT_FAINT_ALPHA,
    **kwargs,
):
    r = results_for_params[pname, pname]
    params = list(guess.keys())
    n_par = len(params)
    if title_formats is None:
        title_formats = dict()
    if title_cl is None:
        title_cl = cls[0]
    if tick_conf_levels is None:
        tick_conf_levels = list(sorted(list(set(list(cls) + [0.25, 0.5]))))

    # Truth and bestfit lines
    lines_to_plot = [("k", bestfit[pname])]
    if truth:
        lines_to_plot.append((truth_color, truth[pname]))
    for color, x in lines_to_plot:
        plt.axvline(x * scale, linewidth=1, color=color, alpha=0.5)

    # Profile likelihood valley
    prof = r["mnprofile"]
    deviance = 2 * (prof["fvals"] - prof["fvals"].min())
    p = stats.chi2(n_par if joint_coverage else 1).cdf(deviance)
    plt.fill_between(prof["locations"] * scale, 0, p, color="gray")

    # Confidence intervals
    for cl, color in zip(cls, contour_colors):
        merr = minos_errors[cl][pname]
        mid = bestfit[pname] * scale
        elow = -merr[0] * scale
        ehigh = merr[1] * scale
        plt.errorbar(
            [mid],
            y=[cl],
            xerr=([elow], [ehigh]),
            c=color,
            capsize=3,
        )
        if cl == title_cl:
            # From https://github.com/dfm/corner.py/blob/main/src/corner/core.py#L220
            title_fmt = title_formats.get(pname, ".2f")
            fmt = "{{0:{0}}}".format(title_fmt).format
            title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            title = title.format(fmt(mid), fmt(elow), fmt(ehigh))
            plt.title(label + " = " + title)

    # Simple 1d log likelihood ratio scan, all other params fixed.
    # Very faint.
    deviance = 2 * (r["z"] - np.min(r["z"]))
    p = stats.chi2(1).cdf(deviance)
    plt.plot(r["x"] * scale, p, linewidth=1, linestyle="-", c="k", alpha=faint_alpha)

    # Ticks for confidence levels
    plt.ylim(0, 1)
    plt.yticks(
        tick_conf_levels,
        [f"{round(100 * x):d}%" for x in tick_conf_levels],
        alpha=faint_alpha,
    )
    plt.grid(axis="y", alpha=faint_alpha, c="k", linewidth=0.5)
    ax.yaxis.tick_right()

    # Brighten up ticks for confidence levels that we draw
    for cl, color in zip(cls, contour_colors):
        our_tick = ax.get_yticklabels()[list(tick_conf_levels).index(cl)]
        our_tick.set_color(color)
        our_tick.set_alpha(1)


def off_diag(
    *,
    px,
    sx,
    py,
    sy,
    bestfit,
    truth,
    cls,
    results_for_params,
    faint_alpha=DEFAULT_FAINT_ALPHA,
    contour_colors=DEFAULT_CONTOUR_COLORS,
    truth_color=DEFAULT_TRUTH_COLOR,
    **kwargs,
):
    r = results_for_params[px, py]

    # Truth / bestfit lines
    for color, q in [("gray", bestfit), (truth_color, truth)]:
        if q is None:
            continue
        style = dict(linewidth=1, color=color, alpha=faint_alpha)
        plt.axvline(q[px] * sx, **style)
        plt.axhline(q[py] * sy, **style)

    # Profile likelihood contour / shadow of n-d confidence blob
    # See mncontour docstring about appending final point
    for cl, color in zip(cls, contour_colors):
        mncontour = r["mncontours"][cl]
        if not len(mncontour):
            continue
        xp, yp = mncontour[:, 0], mncontour[:, 1]
        xp = np.concatenate([xp, [xp[0]]])
        yp = np.concatenate([yp, [yp[0]]])
        plt.plot(xp * sx, yp * sy, c=color)

    # Simple 2d scan, very faint
    z = stats.chi2(2).cdf((2 * (r["z"].max() - r["z"])))
    for cl, color in zip(cls, contour_colors):
        plt.contour(
            r["x"] * sx,
            r["y"] * sy,
            z.T,
            linewidths=1,
            linestyles="-",
            alpha=faint_alpha,
            levels=[cl],
            colors=[color],
        )

    if truth:
        plt.scatter(
            truth[px] * sx, truth[py] * sy, marker="+", color="darkorange", zorder=5
        )
    plt.scatter(bestfit[px] * sx, bestfit[py] * sy, marker="o", color="k", zorder=5)


def plot_corner(
    params,
    off_diag=off_diag,
    diag=diag,
    labels=None,
    scales=None,
    ranges=None,
    truth=None,
    scaled_ranges=None,
    ax_size=DEFAULT_AX_SIZE,
    **kwargs,
):
    if scales is None:
        scales = dict()
    if labels is None:
        labels = dict()
    scaled_ranges = get_scaled_ranges(
        params, scales=scales, ranges=ranges, scaled_ranges=scaled_ranges
    )
    n_par = len(params)

    fig, axes = plt.subplots(
        n_par,
        n_par,
        sharex=False,
        sharey=False,
        figsize=(ax_size * len(params), ax_size * len(params)),
    )

    for par_i, px in enumerate(params):
        for par_j, py in enumerate(params):
            ax = axes[par_j, par_i]

            if par_j < par_i or (par_i == par_j and not diag):
                fig.delaxes(ax)
                continue

            plt.sca(ax)

            sx = scales.get(px, 1)
            plt.xlim(*scaled_ranges[px])
            xlabel = labels.get(px, px)
            if sx != 1:
                xlabel += r" $\times " + str(sx) + "$"

            sy = scales.get(py, 1)
            plt.ylim(*scaled_ranges[py])
            ylabel = labels.get(py, py)
            if sy != 1:
                ylabel += r" $\times " + str(sy) + "$"

            if par_j == n_par - 1:
                plt.xlabel(xlabel, fontsize=12)
            else:
                plt.xticks([])
            if par_i == 0 and par_j != 0:
                plt.ylabel(ylabel, fontsize=12)
            else:
                plt.yticks([])

            if par_i == par_j:
                diag(pname=px, scale=sx, ax=ax, label=xlabel, truth=truth, **kwargs)
            else:
                off_diag(
                    px=px,
                    py=py,
                    sx=sx,
                    sy=sy,
                    ax=ax,
                    truth=truth,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    **kwargs,
                )

    plt.subplots_adjust(wspace=0, hspace=0)


def get_scaled_ranges(params, scales=None, ranges=None, scaled_ranges=None):
    if scales is None:
        scales = dict()
    if scaled_ranges is None:
        scaled_ranges = dict()
    if ranges is None:
        ranges = dict()
    result = dict()
    for pname in params:
        if pname in scaled_ranges:
            result[pname] = scaled_ranges[pname]
        elif pname in ranges:
            result[pname] = np.asarray(ranges[pname]) * scales.get(pname, 1)
        else:
            result[pname] = (None, None)
    return result


def minuit_hardcoded_dof_workaround(cl, real_dof, hardcoded_dof):
    """
    Return 'cl' for use in a function assuming hardcoded_dof
    when the real dof is real_dof
    """
    factor = stats.chi2(real_dof).ppf(cl)
    return stats.chi2(hardcoded_dof).cdf(factor)
