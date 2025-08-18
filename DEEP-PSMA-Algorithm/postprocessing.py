import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, distance_transform_edt, binary_dilation

def filter_fdg_by_overlap_with_probabilities(
    psma_pred_sitk,
    fdg_pred_sitk,
    prob_sitk_psma,
    prob_sitk_fdg,
    distance_threshold=10,
    dilation_radius=5,
    min_relative_overlap=0.10,
    psma_prob_base=1.0,
    psma_prob_slope=0.6,
    min_psma_prob=0.5,
    fdg_prob_thresh=0.90,
    print_reason=True
):
    """
    Filter FDG-laesies op basis van PSMA overlap/nabijheid én probabiliteiten.
    """

    # SimpleITK naar NumPy
    psma = sitk.GetArrayFromImage(psma_pred_sitk).astype(np.uint8)
    fdg = sitk.GetArrayFromImage(fdg_pred_sitk).astype(np.uint8)
    prob_psma = sitk.GetArrayFromImage(prob_sitk_psma).astype(np.float32)
    prob_fdg = sitk.GetArrayFromImage(prob_sitk_fdg).astype(np.float32)

    labeled_fdg, num_fdg = label(fdg)
    print(f"Aantal laesies in FDG (voor filtering): {num_fdg}")

    fdg_filtered = np.zeros_like(fdg, dtype=np.uint8)

    for lesion_label in range(1, num_fdg + 1):
        lesion_mask = (labeled_fdg == lesion_label)
        lesion_coords = np.where(lesion_mask)

        lesion_volume = np.sum(lesion_mask)
        median_prob_fdg = np.median(prob_fdg[lesion_coords])

        # PSMA-overlap met FDG-lesie
        overlap_mask = lesion_mask & (psma > 0)
        overlap_volume = np.sum(overlap_mask)
        relative_overlap = overlap_volume / lesion_volume

        # Nabijheid via dilatie van FDG-lesie
        dilated_mask = binary_dilation(lesion_mask, iterations=dilation_radius)
        near_psma_mask = dilated_mask & (psma > 0)
        psma_near_coords = np.where(near_psma_mask)

        if len(psma_near_coords[0]) > 0:
            median_prob_psma_near = np.median(prob_psma[psma_near_coords])
            relative_psma_volume_near = np.sum(near_psma_mask) / lesion_volume
        else:
            median_prob_psma_near = 0.0
            relative_psma_volume_near = 0.0

        # Adaptieve PSMA-drempel op basis van FDG-zekerheid
        required_psma_prob = psma_prob_base - psma_prob_slope * median_prob_fdg

        # Beslislogica
        accept = False
        if relative_psma_volume_near >= min_relative_overlap and median_prob_psma_near >= required_psma_prob:
            reason = f"PSMA support near lesion (v~={relative_psma_volume_near:.1%}, median p={median_prob_psma_near:.2f} ≥ required {required_psma_prob:.2f})"
            accept = True
        elif median_prob_fdg >= fdg_prob_thresh:
            reason = f"FDG high confidence (median p={median_prob_fdg:.2f} ≥ {fdg_prob_thresh})"
            accept = True
        else:
            reason = f"Rejected: p_psma={median_prob_psma_near:.2f} < required {required_psma_prob:.2f}, p_fdg={median_prob_fdg:.2f} < required {fdg_prob_thresh}"

        if accept:
            fdg_filtered[lesion_mask] = 1
        if print_reason:
            print(f"Lesie {lesion_label:2d}: {reason}")

    labeled_fdg_filtered, num_fdg_filtered = label(fdg_filtered)
    print(f"Aantal laesies in FDG (na filtering): {num_fdg_filtered}")

    fdg_filtered_sitk = sitk.GetImageFromArray(fdg_filtered)
    fdg_filtered_sitk.CopyInformation(fdg_pred_sitk)
    return fdg_filtered_sitk