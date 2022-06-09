import argparse
import h5py
from glob import glob
from random import shuffle, seed
from os import remove


def parse_arguments():
    parser = argparse.ArgumentParser(description='Merge multiple h5 files in one')

    parser.add_argument('--input_dir',
                        action='store',
                        type=str,
                        help='path to dir with h5 file')

    parser.add_argument('--output_file',
                        action='store',
                        type=str,
                        help='path to output file')

    parser.add_argument('--train_percentage',
                        action='store',
                        type=float)

    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_arguments()
    total_records = 0
    tmp_file = opts.output_file.split('.')[0]
    tmp_file = f'{opts.output_file}_tmp.h5'
    with h5py.File(tmp_file, 'w') as out:
        for file in glob(f'{opts.input_dir}/*.h5'):
            with h5py.File(file, 'r') as part:
                for key in part.keys():
                    total_records += 1
                    out.create_group(key)

                    for group_idx in part[key].keys():
                        part[key].copy(group_idx, out[key])

    train_size = int(total_records * opts.train_percentage)

    with h5py.File(opts.output_file, 'w') as out:
        train = out.create_group('train')
        val = out.create_group('val')

        with h5py.File(tmp_file, 'r') as tmp:
            keys = list(tmp.keys())
            seed(245325235)
            shuffle(keys)

            for k in keys[:train_size]:
                train.create_group(k)
                for group_idx in tmp[k].keys():
                    tmp[k].copy(group_idx, train[k])

            for k in keys[train_size:]:
                val.create_group(k)
                for group_idx in tmp[k].keys():
                    tmp[k].copy(group_idx, val[k])

    remove(tmp_file)
