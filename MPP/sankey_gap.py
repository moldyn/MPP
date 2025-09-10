# -*- coding: utf-8 -*-
r"""
Produces simple Sankey Diagrams with matplotlib.
@author: Anneya Golob & marcomanz & pierre-sassoulas & jorwoods & vgalisson
                      .-.
                 .--.(   ).--.
      <-.  .-.-.(.->          )_  .--.
       `-`(     )-'             `)    )
         (o  o  )                `)`-'
        (      )                ,)
        ( ()  )                 )
         `---"\    ,    ,    ,/`
               `--' `--' `--'
                |  |   |   |
                |  |   |   |
                '  |   '   |
"""

# fmt: off
import warnings
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# fmt: on

LOGGER = logging.getLogger(__name__)


class PySankeyException(Exception):
    """Generic PySankey Exception."""


class NullsInFrame(PySankeyException):
    pass


class LabelMismatch(PySankeyException):
    pass


def check_data_matches_labels(labels, data, side):
    """Check whether or not data matches labels.

    Raise a LabelMismatch Exception if not."""
    if len(labels) > 0:
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise LabelMismatch(
                "{0} labels and data do not match.{1}".format(side, msg)
            )


def sankey(
    left,
    right,
    leftWeight=None,
    rightWeight=None,
    colorDict=None,
    leftLabels=None,
    rightLabels=None,
    aspect=4,
    rightColor=False,
    fontsize="medium",
    figureName=None,
    closePlot=False,
    figSize=None,
    ax=None,
):
    """
    Make Sankey Diagram showing flow from left-->right

    Inputs:
        left = NumPy array of object labels on the left of the diagram
        right = NumPy array of corresponding labels on the right of the diagram
            len(right) == len(left)
        leftWeight = NumPy array of weights for each strip starting from the
            left of the diagram, if not specified 1 is assigned
        rightWeight = NumPy array of weights for each strip starting from the
            right of the diagram, if not specified the corresponding leftWeight
            is assigned
        colorDict = Dictionary of colors to use for each label
            {'label':'color'}
        leftLabels = order of the left labels in the diagram
        rightLabels = order of the right labels in the diagram
        aspect = vertical extent of the diagram in units of horizontal extent
        rightColor = If true, each strip in the diagram will be be colored
                    according to its left label
        figSize = tuple setting the width and height of the sankey diagram.
            Defaults to current figure size
        ax = optional, matplotlib axes to plot on, otherwise uses current axes.
    Output:
        ax : matplotlib Axes
    """
    warn = []
    if figureName is not None:
        msg = "use of figureName in sankey() is deprecated"
        warnings.warn(msg, DeprecationWarning)
        warn.append(msg[7:-14])
    if closePlot is not False:
        msg = "use of closePlot in sankey() is deprecated"
        warnings.warn(msg, DeprecationWarning)
        warn.append(msg[7:-14])
    if figSize is not None:
        msg = "use of figSize in sankey() is deprecated"
        warnings.warn(msg, DeprecationWarning)
        warn.append(msg[7:-14])

    if warn:
        LOGGER.warning(
            " The following arguments are deprecated and should be removed: %s",
            ", ".join(warn),
        )

    if ax is None:
        ax = plt.gca()

    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))

    if len(rightWeight) == 0:
        rightWeight = leftWeight

    # plt.rc("text", usetex=False)
    # plt.rc("font", family="serif")

    # Create Dataframe
    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    dataFrame = pd.DataFrame(
        {
            "left": left,
            "right": right,
            "leftWeight": leftWeight,
            "rightWeight": rightWeight,
        },
        index=range(len(left)),
    )

    if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
        raise NullsInFrame("Sankey graph does not support null values.")

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(
        np.r_[dataFrame.left.unique(), dataFrame.right.unique()]
    ).unique()
    LOGGER.debug("Labels to handle : %s", allLabels)

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(dataFrame.left.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame["left"], "left")

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(dataFrame.right.unique()).unique()
    else:
        check_data_matches_labels(rightLabels, dataFrame["right"], "right")

    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        palette = "hls"
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
    LOGGER.debug("The colordict value are : %s", colorDict)

    # Determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
            ].leftWeight.sum()
            rightDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
            ].rightWeight.sum()
        ns_l[leftLabel] = leftDict
        ns_r[leftLabel] = rightDict

    # Determine positions of left label patches and total widths
    leftWidths, topEdge = _get_positions_and_total_widths(dataFrame, leftLabels, "left")

    # Determine positions of right label patches and total widths
    rightWidths, topEdge = _get_positions_and_total_widths(
        dataFrame, rightLabels, "right"
    )

    # Total vertical extent of diagram
    xMax = topEdge / aspect

    previousleftlabel = ""
    # Draw vertical bars on left and right of each  label's section & print label
    for vall, leftLabel in enumerate(leftLabels):
        if vall != 0:
            if _draw_label(leftWidths[leftLabel], leftWidths[previousleftlabel]):
                continue
        ax.text(
            -0.05 * xMax,
            leftWidths[leftLabel]["bottom"] + 0.5 * leftWidths[leftLabel]["left"],
            rf"\textbf{{{leftLabel}}}",
            {"ha": "right", "va": "center"},
            fontsize=fontsize,
            zorder=2,
        )
        previousleftlabel = leftLabel
    previousrightlabel = ""
    for valr, rightLabel in enumerate(rightLabels):
        if valr != 0:
            if _draw_label(rightWidths[rightLabel], rightWidths[previousrightlabel]):
                continue
        ax.text(
            1.05 * xMax,
            rightWidths[rightLabel]["bottom"] + 0.5 * rightWidths[rightLabel]["right"],
            rf"\textbf{{{rightLabel}}}",
            {"ha": "left", "va": "center"},
            fontsize=fontsize,
            zorder=2,
        )
        previousrightlabel = rightLabel

    ymin, ymax = None, None
    # Plot strips
    for vall, leftLabel in enumerate(leftLabels):
        for valr, rightLabel in enumerate(rightLabels):
            labelColor = leftLabel
            if rightColor:
                labelColor = rightLabel
            if (
                len(
                    dataFrame[
                        (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
                    ]
                )
                > 0
            ):
                # Create array of y values for each strip, half at left value,
                # half at right, convolve
                ys_d = np.array(
                    50 * [leftWidths[leftLabel]["bottom"]]
                    + 50 * [rightWidths[rightLabel]["bottom"]]
                )
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
                ys_u = np.array(
                    50 * [leftWidths[leftLabel]["bottom"] + ns_l[leftLabel][rightLabel]]
                    + 50
                    * [rightWidths[rightLabel]["bottom"] + ns_r[leftLabel][rightLabel]]
                )
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")

                yrange = np.subtract(*ax.get_ylim()[::-1])
                relative_width = np.mean(ys_u - ys_d) / yrange

                # Update bottom edges at each label so next strip starts at the right place
                leftWidths[leftLabel]["bottom"] += ns_l[leftLabel][rightLabel]
                rightWidths[rightLabel]["bottom"] += ns_r[leftLabel][rightLabel]
                ax.fill_between(
                    np.linspace(-0.013, 1.013, len(ys_d)) * xMax,
                    ys_d,
                    ys_u,
                    alpha=1.0,
                    zorder=1,
                    facecolor=colorDict[labelColor],
                )

                if ymin is None:
                    ymin = ys_d.min()
                ymin = np.min([ys_d.min(), ys_u.min(), ymin])

                if ymax is None:
                    ymax = ys_d.max()
                ymax = np.max([ys_d.max(), ys_u.max(), ymax])

    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    if figSize is not None:
        plt.gcf().set_size_inches(figSize)

    if figureName is not None:
        fileName = "{}.png".format(figureName)
        plt.savefig(fileName, bbox_inches="tight", dpi=150)
        LOGGER.info("Sankey diagram generated in '%s'", fileName)
    if closePlot:
        plt.close()

    return ax


def _get_positions_and_total_widths(df, labels, side):
    """Determine positions of label patches and total widths"""
    # add gap
    gap = 50000
    # print(f'gap : {gap}')
    widths = defaultdict()
    for i, label in enumerate(labels):
        labelWidths = {}
        labelWidths[side] = df[df[side] == label][side + "Weight"].sum()
        if i == 0:
            labelWidths["bottom"] = 0
            labelWidths["top"] = labelWidths[side]
        else:
            bottomWidth = widths[labels[i - 1]]["top"] + gap
            weightedSum = 0.02 * df[side + "Weight"].sum()
            labelWidths["bottom"] = bottomWidth + weightedSum
            labelWidths["top"] = labelWidths["bottom"] + labelWidths[side]
            topEdge = labelWidths["top"]
        widths[label] = labelWidths
        LOGGER.debug("%s position of '%s' : %s", side, label, labelWidths)

    return widths, topEdge


def _draw_label(widths1, widths2, minDistanceOfLabels=150000):
    return (
        np.abs(
            (widths1["top"] - 0.5 * (widths1["top"] - widths1["bottom"]))
            - (widths2["top"] - 0.5 * (widths2["top"] - widths2["bottom"]))
        )
        < minDistanceOfLabels
    )
