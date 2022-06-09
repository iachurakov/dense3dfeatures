import pickle

import point_cloud_utils as pcu
import numpy as np
import argparse
from functools import partial
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from string import ascii_uppercase


cat2synset = {
    'Airplane': '02691156',
    'Bag': '02773838',
    'Cap': '02954340',
    'Car': '02958343',
    'Chair': '03001627',
    'Earphone': '03261776',
    'Guitar': '03467517',
    'Knife': '03624134',
    'Lamp': '03636649',
    'Laptop': '03642806',
    'Motorbike': '03790512',
    'Mug': '03797390',
    'Pistol': '03948459',
    'Rocket': '04099429',
    'Skateboard': '04225987',
    'Table': '04379243'
}


def parse_mesh(file_path, output_dir, split):
    v, f = pcu.load_mesh_vf(file_path)
    labels = np.zeros(f.shape[0])
    parts = file_path.split('/')
    cat = cat2synset[parts[-2]]

    with open(f'{file_path[:-4]}_labels.txt', 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip() != '']
        labels_set = []
        for label_idx in lines[0::2]:
            labels_set.append(ascii_uppercase.index(label_idx[-1]))

        for label_idx, faces_idx in zip(labels_set, lines[1::2]):
            labels[[int(face_idx) - 1 for face_idx in faces_idx.split(' ')]] = label_idx

    shape_id = f'{parts[-1][:-4]}_{cat}'
    part = split[shape_id]
    np.savez(f'{output_dir}/{part}/{shape_id}', vs=v, faces=f,
             labels=labels.astype(int))
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert off meshes to npz and parse faces\' labels')

    parser.add_argument('--input_mask',
                        action='store',
                        type=str,
                        help='mask for off files')

    parser.add_argument('--split_file',
                        action='store',
                        type=str,
                        help='path to pkl of dict with train/val/test split')

    parser.add_argument('--output_dir',
                        action='store',
                        type=str,
                        help='path to output dir')

    opt = parser.parse_args()

    with open(opt.split_file, 'rb') as file:
        split_dict = pickle.load(file)

    files = glob(opt.input_mask)
    worker = partial(parse_mesh, output_dir=opt.output_dir, split=split_dict)

    with mp.Pool(mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(worker, files), total=len(files)):
            pass
