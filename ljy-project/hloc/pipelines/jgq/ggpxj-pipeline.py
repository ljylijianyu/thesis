from pathlib import Path
import argparse

from utils import (
    create_query_list_with_intrinsics, scale_sfm_images, evaluate)
from hloc import extract_features, match_features, pairs_from_covisibility
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

    # feature_conf = {
    #     'output': 'feats-superpoint-n4096-r1024',
    #     'model': {
    #         'name': 'superpoint',
    #         'nms_radius': 3,
    #         'max_keypoints': 4096,
    #     },
    #     'preprocessing': {
    #         'grayscale': True,
    #         'resize_max': 1024,
    #     },
    # }
    feature_conf = extract_features.confs['sift']

    matcher_conf = match_features.confs['NN-mutual']
    retrieval_conf = extract_features.confs['netvlad']

    create_query_list_with_intrinsics(
            gt_dir / 'empty_all', query_list, test_list,
            ext='.txt', image_dir=images)
    with open(test_list, 'r') as f:
        query_seqs = {q.split('/')[0] for q in f.read().rstrip().split('\n')}

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, num_loc,
        db_model=ref_sfm_sift, query_prefix=query_seqs)

    features = extract_features.main(
            feature_conf, images, outputs, as_half=True)
    pairs_from_covisibility.main(
            ref_sfm_sift, sfm_pairs, num_matched=num_covis)
    sfm_matches = match_features.main(
            matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    scale_sfm_images(ref_sfm_sift, ref_sfm_scaled, images)
    triangulation.main(
        ref_sfm, ref_sfm_scaled,
        images,
        sfm_pairs,
        features,
        sfm_matches)

    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf['output'], outputs)

    localize_sfm.main(
        ref_sfm,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
        prepend_camera_name=False)





if __name__ == '__main__':
    dataset = Path('/home/ljy/colmap_model/jiguangqi/jgq/dense/images/')
    outputs = Path('/home/ljy/outputs/jiguangqi_dbg/')
    gt_dirs = Path('/home/ljy/colmap_model/jiguangqi/gt-dir/')
    num_loc = 10
    num_covis = 20

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

