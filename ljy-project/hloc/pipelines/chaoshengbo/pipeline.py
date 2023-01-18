from pathlib import Path
import argparse

from .utils import (
    create_query_list_with_intrinsics, scale_sfm_images, evaluate)
from ... import extract_features, match_features, pairs_from_covisibility
from ... import triangulation, localize_sfm, pairs_from_retrieval, logger







if __name__ == '__main__':

    dataset = Path('/home/ljy/outputs/sfm/sfm_superpoint+superglue/')

    output = Path('/home/ljy/outputs/csb/')

    gt_dirs = dataset

    results = output / 'results.txt'

    run_loc(dataset,
            gt_dirs,
            output,
            results,
            num_covis,
            num_loc)

    all_results = {}
    all_results = results

    evaluate(
        gt_dirs / 'empty', all_results,
        gt_dirs / 'list_query.txt', ext='.txt'
    )

