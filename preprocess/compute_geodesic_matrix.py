import numpy as np
import igl
from joblib import Parallel, delayed
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from glob import glob
import h5py


def compute_pairwise_geodesic(vs, faces, n_jobs):
    faces = faces.astype(np.int64)
    empty_arr = np.array([]).astype(np.int64)
    all_faces_idxs = np.arange(len(faces))

    def dist_from_one(src_face_idx):
        return igl.exact_geodesic(v=vs,
                                  f=faces,
                                  vs=empty_arr,
                                  fs=np.array([src_face_idx], dtype=np.int64),
                                  vt=empty_arr,
                                  ft=all_faces_idxs)

    dists = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(dist_from_one)(face_idx) for face_idx in all_faces_idxs)

    return np.array(dists)


def worker(vs, faces, item_idx, output_dir, n_jobs):
    split, idx = item_idx
    dists = compute_pairwise_geodesic(vs, faces, n_jobs)
    np.savez(f'{output_dir}/{split}|{idx}.npz', data=dists)


def update_dataset(dataset_path, output_dir):
    with h5py.File(dataset_path, 'a') as dataset:
        for file in glob(f'{output_dir}/*.npz'):
            data = np.load(file)['data']
            item_idx = file.split('/')[-1]
            split, idx = item_idx[:-4].split('|')
            dataset[split][idx]['mesh'].create_dataset('geodesic_dists', data=data,
                                                       dtype='f4', compression='gzip')


def parse_arguments():
    parser = ArgumentParser(description='Compute pairwise geodesic distances between faces\' centers')

    parser.add_argument('--input_file',
                        action='store',
                        type=str,
                        help='path to h5 file with dataset')

    parser.add_argument('--n_jobs_outer',
                        action='store',
                        type=int,
                        help='the number of process for processing dataset')

    parser.add_argument('--n_jobs_inner',
                        action='store',
                        type=int,
                        help='the number of process for processing one shape')

    parser.add_argument('--output_dir',
                        action='store',
                        type=str,
                        help='path to output_dir')

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_arguments()
    dataset = h5py.File(opt.input_file, 'r')
    tmp_dir = Path(opt.output_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)

    worker_fn = partial(worker,
                        output_dir=str(tmp_dir),
                        n_jobs=opt.n_jobs_inner)

    items_idx = []

    for split in dataset.keys():
        for sample_idx in dataset[split].keys():
            items_idx.append((split, sample_idx))

    Parallel(n_jobs=opt.n_jobs_outer, backend='multiprocessing', verbose=50)(delayed(worker_fn)(dataset[item_idx[0]][item_idx[1]]['mesh']['vs'][:],
                                                                     dataset[item_idx[0]][item_idx[1]]['mesh']['faces'][:],
                                                                     item_idx)
                                                  for item_idx in items_idx)
    dataset.close()
    update_dataset(opt.input_file, opt.output_dir)
