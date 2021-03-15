import json
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_conditions(
    nBlocks=2,
    nFiller=20,
    alphas=[0.7, 0.9, 1.1],
    gammas=[1],
    m_min=0.1,
    m_max=10,
    U=1,
    e_sd=0.05,
    p_ranges={"a": [0.2, 0.4], "b": [0.45, 0.6], "c": [0.75, 0.9]},
    nPresentationsEach=2,
    targetDuration=1500,
    otherDuration=1000,
    fillerDuration=1250,
    pairs=[("a", "b"), ("b", "c"), ("a", "c")],
    seed=None,
):
    """Main function to generate the core conditions DataFrame.

    Additional variables are added using additional functions below.

    Args:
        pairs (list, optional): [description]. Defaults to [("a", "b"), ("b", "c"), ("a", "c")].

    Returns:
        [type]: [description]
    """
    if seed is not None:
        np.random.seed(seed)

    conditions = []

    # Experimental trials
    i = 0
    experimental = []
    for alt0, alt1 in pairs:
        # 2 each alternative targeted once
        for target in [0, 1]:
            other = 1 - target
            # 2 target first, target second
            for targetFirst in [True, False]:
                if targetFirst:
                    durations = [targetDuration, otherDuration] * nPresentationsEach
                else:
                    durations = [otherDuration, targetDuration] * nPresentationsEach

                # 2 alternative-wise or attribute-wise presentation
                for sequenceKind in ["alternatives", "attributes"]:
                    if sequenceKind == "alternatives":
                        attributes = ["all"] * nPresentationsEach * 2
                        targetId = target
                        otherId = other
                        if targetFirst:
                            alternatives = [target, other] * nPresentationsEach
                        else:
                            alternatives = [other, target] * nPresentationsEach
                    else:
                        alternatives = ["all"] * nPresentationsEach * 2
                        targetId = ["p", "m"][target]
                        otherId = ["p", "m"][other]
                        if targetFirst:
                            attributes = [targetId, otherId] * nPresentationsEach
                        else:
                            attributes = [otherId, targetId] * nPresentationsEach
                    sequence = dict(
                        durations=durations,
                        alternatives=alternatives,
                        attributes=attributes,
                    )

                    # Repetitions
                    for alpha in alphas:
                        for gamma in gammas:
                            for block in range(nBlocks):
                                p0 = np.random.uniform(*p_ranges[alt0], size=1)[0]
                                w0 = W1(p0, gamma)
                                p1 = np.random.uniform(*p_ranges[alt1], size=1)[0]
                                w1 = W1(p1, gamma)
                                e0 = np.random.normal(scale=e_sd)
                                e1 = np.random.normal(scale=e_sd)
                                m0 = (
                                    np.clip((1.0 / w0) ** (U / alpha), m_min, m_max)
                                    + e0
                                )
                                m1 = (
                                    np.clip((1.0 / w1) ** (U / alpha), m_min, m_max)
                                    + e1
                                )
                                condition = pd.DataFrame(
                                    dict(
                                        block=block,
                                        condition="exp_{}".format(i),
                                        alt0=alt0,
                                        alt1=alt1,
                                        p0=np.round(p0, 2),
                                        p1=np.round(p1, 2),
                                        m0=np.round(m0, 2),
                                        m1=np.round(m1, 2),
                                        target=targetId,
                                        other=otherId,
                                        targetFirst=targetFirst,
                                        presentation=sequenceKind,
                                        sequence=json.dumps(sequence),
                                        alpha=alpha,
                                        gamma=gamma,
                                    ),
                                    index=np.ones(1) * i,
                                )
                                experimental.append(condition)
                                i += 1

    experimental = pd.concat(experimental)
    experimental["phase"] = "experimental"
    conditions.append(experimental)

    # Filler trials
    conditions.append(
        pd.DataFrame(
            dict(
                block=np.repeat(range(nBlocks), nFiller // nBlocks),
                condition=["catch_{}".format(i) for i in range(nFiller)],
                alt0="dominating",
                alt1="dominated",
                p0=np.random.uniform(0.5, 0.80, size=nFiller).round(2),
                p1=np.random.uniform(0.10, 0.5, size=nFiller).round(2),
                m0=m_max * np.random.uniform(0.35, 0.7, size=nFiller).round(2),
                m1=m_max * np.random.uniform(m_min, 0.35, size=nFiller).round(2),
                target=np.nan,
                other=np.nan,
                targetFirst=np.nan,
                presentation=(nFiller // 2) * ["alternatives", "attributes"],
                sequence=nFiller
                // 2
                * [
                    # Alternative-wise sequence
                    json.dumps(
                        dict(
                            durations=[fillerDuration] * 2 * nPresentationsEach,
                            attributes=["all"] * 2 * nPresentationsEach,
                            alternatives=[0, 1] * nPresentationsEach,
                        )
                    ),
                    # Attribute-wise sequece
                    json.dumps(
                        dict(
                            durations=[fillerDuration] * 2 * nPresentationsEach,
                            attributes=["p", "m"] * nPresentationsEach,
                            alternatives=["all"] * 2 * nPresentationsEach,
                        )
                    ),
                ],
                phase="experimental",
            )
        )
    )
    conditions = pd.concat(conditions)
    conditions.reset_index(drop=True, inplace=True)
    return conditions


def W1(p, gamma):
    """
    One parameter probability weighting function.
    Reference: Fehr & Glimcher (Eds.) Neuroeconomics, Appendix: Prospect Theory, pp. 545ff
    """
    return p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))


def computeDurations(sequence, alternative=None, attribute=None):
    """Computes the relative presentation duration of alternatives, attributes, or combinations of both for a given sequence.

    Args:
        sequence (dict): Sequence dictionary with keys "attributes", "alternatives" and "durations", each containing a list.
        alternative (int, optional): Index of alternative for which overall relative duration should be computed. Defaults to None.
        attribute (str, optional): Attribute for which overall relative duration should be computed. For example "p" or "m". Defaults to None.

    Returns:
        float: Relative duration measure.
    """
    if alternative is not None:
        alt_mask = np.array(
            [alt in [alternative, "all"] for alt in sequence["alternatives"]]
        )
    else:
        alt_mask = np.ones(len(sequence["alternatives"])).astype(bool)

    if attribute is not None:
        att_mask = np.array(
            [att in [attribute, "all"] for att in sequence["attributes"]]
        )
    else:
        att_mask = np.ones(len(sequence["attributes"])).astype(bool)

    g = np.sum(np.array(sequence["durations"])[alt_mask & att_mask]) / np.sum(
        np.array(sequence["durations"])
    )
    return g


def addDurationVariables(df):
    """Adds variables for relative durations towards alernatives and attributes.

    Args:
        df (pandas.DataFrame): Dataframe with `sequence` variable containing the presentation sequence.

    Returns:
        pandas.DataFrame: The DataFrame with added variables.
    """
    for alt in [0, 1]:
        df[f"g{alt}r"] = df.apply(
            lambda x: computeDurations(json.loads(x["sequence"]), alternative=alt),
            axis=1,
        )
    for att in ["p", "m"]:
        df[f"g{att}r"] = df.apply(
            lambda x: computeDurations(json.loads(x["sequence"]), attribute=att), axis=1
        )

    # Normalize durations to 1 in each trial
    df["g0"] = df["g0r"] / df[["g0r", "g1r"]].sum(axis=1)
    df["g1"] = df["g1r"] / df[["g0r", "g1r"]].sum(axis=1)
    df["gm"] = df["gmr"] / df[["gmr", "gpr"]].sum(axis=1)
    df["gp"] = df["gpr"] / df[["gmr", "gpr"]].sum(axis=1)

    return df.drop(["g0r", "g1r", "gmr", "gpr"], axis=1)


def addLastFavoursVariable(df):
    """Adds variable that describes which alternative is favoured by the last presentation step in the sequence.

    Args:
        df (pandas.DataFrame): DataFrame with conditions. Must contain columns `presentation`, `targetFirst`, `target`, `other`, `p0`, `p1`, `m0`, `m1`.

    Returns:
        pandas.DataFrame: The DataFrame with added `lastFavours` column.
    """
    df["lastFavours"] = np.where(
        df["presentation"] == "alternatives",
        df["sequence"].apply(lambda x: json.loads(x)["alternatives"][-1]),
        np.where(
            df["presentation"] == "attributes",
            np.where(
                df["sequence"].apply(lambda x: json.loads(x)["attributes"][-1] == "p"),
                df["higherP"],
                df["higherM"],
            ),
            np.nan,
        ),
    ).astype(float)
    return df


if __name__ == "__main__":

    conditions = build_conditions(
        nBlocks=2,
        nFiller=20,
        alphas=[0.7, 0.9, 1.1],
        gammas=[1],
        m_min=0.1,
        m_max=10,
        U=1,
        e_sd=0.05,
        p_ranges={"a": [0.2, 0.4], "b": [0.45, 0.6], "c": [0.75, 0.9]},
        nPresentationsEach=2,
        targetDuration=1500,
        otherDuration=1000,
        fillerDuration=1250,
        seed=44,
    )

    # Identify options with higher P and higher M in each trial
    conditions["higherP"] = (
        conditions[["p0", "p1"]].idxmax(axis=1).apply(lambda x: int(x[-1]))
    )
    conditions["higherM"] = (
        conditions[["m0", "m1"]].idxmax(axis=1).apply(lambda x: int(x[-1]))
    )

    conditions = addDurationVariables(conditions)
    conditions = addLastFavoursVariable(conditions)

    # Add numerical `presentation` variable for pyDDM
    conditions["presentation01"] = np.where(
        conditions["presentation"] == "alternatives",
        0,
        np.where(conditions["presentation"] == "attributes", 1, np.nan),
    )

    conditions.to_json(join("stimuli", "conditions.json"), orient="records")