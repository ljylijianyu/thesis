from pathlib import Path
import argparse

from utils import (
    create_query_list_with_intrinsics, scale_sfm_images, evaluate)
from hloc import extract_features, match_features, pairs_from_covisibility,match_dense
from hloc import triangulation, localize_sfm, pairs_from_retrieval, logger


def run_loc(images, gt_dir, outputs, results, num_covis, num_loc):
    ref_sfm_sift = gt_dir / 'model_train'
    test_list = gt_dir / 'list_query.txt'

    outputs.mkdir(exist_ok=True, parents=True)
    ref_sfm = outputs / 'sfm_sift+adalam'
    ref_sfm_scaled = outputs / 'sfm_sift_scaled'
    query_list = outputs / 'query_list_with_intrinsics.txt'
    sfm_pairs = outputs / f'pairs-db-covis{num_covis}.txt'
    loc_pairs = outputs / f'pairs-query-netvlad{num_loc}.txt'

    feature_conf = extract_features.confs['superpoint_aachen']

    dense_conf = match_dense.confs['loftr']

    retrieval_conf = extract_features.confs['netvlad']

 ###################   #
    create_query_list_with_intrinsics(
        gt_dir / 'empty_all', query_list, test_list,
        ext='.txt', image_dir=images)
    with open(test_list, 'r') as f:
        query_seqs = {q.split('/')[0] for q in f.read().rstrip().split('\n')}

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)

    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, num_loc,
        db_model=ref_sfm_sift, query_prefix=query_seqs)
######################

#    features_sp = extract_features.main(feature_conf, images)

#    features, matches = match_dense.main(dense_conf, pairs, images,
#                           export_dir=outputs,
#                          features_ref=features_sp)

    pairs_from_covisibility.main(
            ref_sfm_sift, sfm_pairs, num_matched=num_covis)

    sfm_matches = match_dense.main(
            dense_conf, sfm_pairs, images, export_dir=outputs)

    scale_sfm_images(ref_sfm_sift, ref_sfm_scaled, images)














if __name__ == '__main__':
    dataset = Path('/home/ljy/colmap_model/jiguangqi/jgq/dense/images/')
    outputs = Path('/home/ljy/outputs/jgq-loftr/')
    gt_dirs = Path('/home/ljy/colmap_model/jiguangqi/gt-dir/')
    num_loc = 5
    num_covis = 5

    results = outputs / 'results.txt'
    all_results = {}

    run_loc(
        dataset,
        gt_dirs,
        outputs,
        results,
        num_covis,
        num_loc
    )

    print("开始评估")
    evaluate(
        gt_dirs / 'empty_all', results,
        gt_dirs / 'list_query.txt', ext='.txt')

