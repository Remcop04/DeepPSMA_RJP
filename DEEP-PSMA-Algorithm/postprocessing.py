import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, distance_transform_edt, binary_dilation

def _median_ignore_zeros(prob_array: np.ndarray, mask: np.ndarray, default: float = 0.0) -> float:
    """
    Geef de mediaan van de waarden in prob_array binnen mask, waarbij 0 wordt genegeerd.
    Als er na filtering geen waarden overblijven, retourneer 'default'.
    """
    vals = prob_array[mask]
    if vals.size == 0:
        return float(default)
    # Filter NaN/inf en nullen
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0.0]
    if vals.size == 0:
        return float(default)
    return float(np.median(vals))


def get_core_mask_top_fraction(prob_fdg, lesion_mask, top_fraction=0.4, small_lesion_size=5):
    """
    Bepaal kernmasker van een laesie op basis van de hoogste fractie probabilities.
    - top_fraction: fractie van voxels die als 'kern' wordt genomen (bijv. 0.2 = hoogste 20%)
    - small_lesion_size: als laesie klein is, neem alleen de voxel met max probability
    """
    lesion_probs = prob_fdg[lesion_mask]
    n = len(lesion_probs)
    if n == 0:
        return np.zeros_like(lesion_mask, dtype=bool)

    if n <= small_lesion_size:
        # Kleine laesies: enkel de hoogste voxel meenemen
        max_idx = np.argmax(lesion_probs)
        core_mask = np.zeros_like(lesion_mask, dtype=bool)
        flat_indices = np.flatnonzero(lesion_mask)
        core_mask.flat[flat_indices[max_idx]] = True
        return core_mask

    # cutoff voor top-fractie
    cutoff_index = int(n * (1 - top_fraction))
    cutoff_index = max(0, min(cutoff_index, n - 1))
    cutoff = np.partition(lesion_probs, cutoff_index)[cutoff_index]
    core_mask = (prob_fdg >= cutoff) & lesion_mask
    return core_mask


def filter_fdg_by_overlap_with_probabilities(
    psma_pred_sitk,
    fdg_pred_sitk,
    prob_sitk_psma,
    prob_sitk_fdg,
    min_relative_overlap=0.9,
    psma_prob_base=1.0,
    psma_prob_slope=0.5,
    fdg_prob_thresh=0.80,
    print_reason=True,
):
    """
    Filter FDG-laesies o.b.v. PSMA overlap/nabijheid én probabiliteiten.
    - Neemt ALLE relevante PSMA-laesies mee; acceptatie zodra ÉÉN PSMA-laesie voldoet.
    - Mediaanprobabilities worden berekend met uitsluiting van nulwaarden.
    """
    psma = sitk.GetArrayFromImage(psma_pred_sitk).astype(np.uint8)
    fdg = sitk.GetArrayFromImage(fdg_pred_sitk).astype(np.uint8)
    prob_psma = sitk.GetArrayFromImage(prob_sitk_psma).astype(np.float32)
    prob_fdg = sitk.GetArrayFromImage(prob_sitk_fdg).astype(np.float32)

    labeled_fdg, num_fdg = label(fdg)
    labeled_psma, _ = label(psma)

    fdg_filtered = np.zeros_like(fdg, dtype=np.uint8)
    if print_reason:
        print(f"Number of lesions in FDG (before filtering): {num_fdg}")

    for lesion_label in range(1, num_fdg + 1):
        lesion_mask = (labeled_fdg == lesion_label)
        lesion_volume = int(np.sum(lesion_mask))
        if lesion_volume < 1:
            continue

        # Nabijheid via SignedMaurerDistanceMap (mm)
        radius_mm = 8
        lesion_mask_sitk = sitk.GetImageFromArray(lesion_mask.astype(np.uint8))
        lesion_mask_sitk.CopyInformation(fdg_pred_sitk)
        dist = sitk.SignedMaurerDistanceMap(
            lesion_mask_sitk, useImageSpacing=True, squaredDistance=False
        )
        dilated = dist <= radius_mm
        dilated_mask = sitk.GetArrayFromImage(dilated).astype(bool)

        # Core FDG-masker en mediaan zonder nullen
        core_mask_fdg = get_core_mask_top_fraction(prob_fdg, lesion_mask, top_fraction=0.5)
        median_prob_fdg = _median_ignore_zeros(prob_fdg, core_mask_fdg, default=0.0)

        # Zoek alle overlappende/aanliggende PSMA-laesies
        overlapping_labels = np.unique(labeled_psma[dilated_mask])
        overlapping_labels = overlapping_labels[overlapping_labels > 0]

        accept = False
        best_reason = None

        if overlapping_labels.size > 0:
            for psma_label in overlapping_labels:
                psma_mask = (labeled_psma == psma_label)
                if np.any(psma_mask):
                    core_mask_psma = get_core_mask_top_fraction(prob_psma, psma_mask, top_fraction=0.5)
                    median_prob_psma = _median_ignore_zeros(prob_psma, core_mask_psma, default=0.0)
                    relative_psma_volume = float(np.sum(psma_mask)) / float(lesion_volume)
                else:
                    median_prob_psma = 0.0
                    relative_psma_volume = 0.0

                # Adaptieve PSMA-drempel o.b.v. FDG-zekerheid
                required_psma_prob = psma_prob_base - psma_prob_slope * median_prob_fdg

                # Acceptatie op basis van PSMA-bewijs
                if relative_psma_volume >= min_relative_overlap and median_prob_psma >= required_psma_prob:
                    best_reason = (f"PSMA support (PSMA_label={psma_label}, "
                                   #f"v~={relative_psma_volume:.1%}, "
                                   f"median p={median_prob_psma:.2f} ≥ req {required_psma_prob:.2f}. "
                                   f"FDG_vol {lesion_volume})")
                    accept = True
                    break  # één ondersteunende PSMA-laesie is voldoende

        # Valt terug op FDG-confidence als PSMA niet overtuigt
        if not accept and median_prob_fdg >= fdg_prob_thresh:
            best_reason = (f"FDG high confidence (median p(no-zero)={median_prob_fdg:.2f} ≥ {fdg_prob_thresh}. "
                           f"FDG_vol {lesion_volume})")
            accept = True

        if not accept:
            # Meld expliciet dat nullen genegeerd zijn
            best_reason = (f"Rejected: missing PSMA-support; "
                           f"p_fdg={median_prob_fdg:.2f} < {fdg_prob_thresh}. "
                           f"FDG_vol {lesion_volume})")

        if accept:
            fdg_filtered[lesion_mask] = 1

        if print_reason:
            print(f"Lesion {lesion_label:2d}: {best_reason}")

    fdg_filtered_sitk = sitk.GetImageFromArray(fdg_filtered)
    fdg_filtered_sitk.CopyInformation(fdg_pred_sitk)
    return fdg_filtered_sitk