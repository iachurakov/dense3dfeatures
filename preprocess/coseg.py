from functools import partial
from trimesh.base import Trimesh
from glob import glob
import point_cloud_utils as pcu
import numpy as np
import igl
import multiprocessing as mp
import h5py
import argparse
import pymeshlab as pml
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree


def clean_mesh(mesh_set):
    mesh_set.remove_unreferenced_vertices()
    mesh_set.remove_zero_area_faces()
    mesh_set.remove_duplicate_vertices()
    mesh_set.remove_duplicate_faces()
    mesh_set.repair_non_manifold_edges_by_removing_faces()
    mesh = mesh_set.current_mesh()

    return mesh.vertex_matrix(), mesh.face_matrix()


def find_neighbor(faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            return i

    return except_face


def sample_points(vertices: np.ndarray, faces: np.ndarray, n_points, radius=-1):
    f_i, bc = pcu.sample_mesh_poisson_disk(vertices, faces,
                                           int(n_points * 1.1), radius=radius)

    point_cloud = pcu.interpolate_barycentric_coords(faces, f_i, bc, vertices)

    if point_cloud.shape[0] > n_points:
        perm = np.random.choice(point_cloud.shape[0], n_points, replace=False)
        return point_cloud[perm]

    return point_cloud


def get_annotated_mesh(vertices: np.ndarray, faces: np.ndarray,
                       num_faces: int, faces_labels: np.ndarray):
    """
    Generate features for MeshNet and cluster mesh

    :param vertices: array of vertices, shape: n x 3
    :param faces: array of faces, shape: m x 3
    :param num_faces: max num of faces
    :return: status, features, was mesh decimated or not
    """

    decimated_vertices, decimated_faces = None, None
    if 0 < num_faces < faces.shape[0]:
        mesh_set = pml.MeshSet()
        mesh_set.add_mesh(pml.Mesh(vertices, faces))
        mesh_set.simplification_quadric_edge_collapse_decimation(targetfacenum=num_faces)
        decimated_vertices, decimated_faces = clean_mesh(mesh_set)

        if decimated_faces.shape[0] > num_faces:
            return False, (None, None), None, (None, None)

        decimated_faces_centers = decimated_vertices[decimated_faces].mean(axis=1)
        closest = igl.signed_distance(decimated_faces_centers, vertices, faces)[1]
        faces_labels = faces_labels[closest]

    v = vertices.copy() if decimated_vertices is None else decimated_vertices
    f = faces.copy() if decimated_faces is None else decimated_faces

    # move to center
    center = (np.max(v, 0) + np.min(v, 0)) / 2
    v = v - center

    # normalize
    max_len = np.max(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
    v = v / np.sqrt(max_len)

    # get normal vector
    face_normals = Trimesh(v, f, process=False, validate=False).face_normals

    # get neighbors
    faces_contain_this_vertex = []
    for i in range(len(v)):
        faces_contain_this_vertex.append(set([]))

    centers = []
    corners = []

    for i in range(len(f)):
        [v1, v2, v3] = f[i]
        x1, y1, z1 = v[v1]
        x2, y2, z2 = v[v2]
        x3, y3, z3 = v[v3]
        centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
        corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
        faces_contain_this_vertex[v1].add(i)
        faces_contain_this_vertex[v2].add(i)
        faces_contain_this_vertex[v3].add(i)

    neighbors = []
    for i in range(len(f)):
        [v1, v2, v3] = f[i]

        n1 = find_neighbor(faces_contain_this_vertex, v1, v2, i)
        n2 = find_neighbor(faces_contain_this_vertex, v2, v3, i)
        n3 = find_neighbor(faces_contain_this_vertex, v3, v1, i)
        neighbors.append([n1, n2, n3])

    centers = np.array(centers)
    corners = np.array(corners)
    features = np.concatenate([centers, corners, face_normals], axis=1)
    neighbors = np.array(neighbors)

    return True, (features, neighbors), faces_labels, (decimated_vertices, decimated_faces)


def get_annotated_point_cloud(vertices: np.ndarray,
                              faces: np.ndarray,
                              num_points: int,
                              faces_labels: np.ndarray):
    """
    Get patches from point cloud

    :param vertices: array of vertices, shape: n x 3
    :param faces: array of faces, shape: m x 3
    :param num_points: the number of points in point cloud
    :param faces_labels: patches' labels for faces, shape: m
    :return: list of patches, patch labels for points, segmentation labels for points
    """

    point_cloud = sample_points(vertices, faces, num_points)
    _, closest_faces, _ = igl.signed_distance(point_cloud, vertices, faces, False)

    return point_cloud, faces_labels[closest_faces]


def process_mesh(file_path, tmp_dir, num_points, num_faces):
    mesh = np.load(file_path)
    v, f, labels = mesh['vs'], mesh['faces'], mesh['seg_labels']

    success_status, mesh_features, mesh_labels, (decimated_v, decimated_f) = get_annotated_mesh(v, f,
                                                                                                num_faces,
                                                                                                labels.copy())
    if not success_status:
        return False, file_path

    point_cloud, point_labels = get_annotated_point_cloud(v, f, num_points, labels)

    if decimated_v is not None and decimated_f is not None:
        pts2faces = igl.signed_distance(point_cloud, decimated_v, decimated_f)[1]
        centers = decimated_v[decimated_f].mean(axis=1)
        faces2pts = pcu.k_nearest_neighbors(centers, point_cloud, 5)[1]
    else:
        pts2faces = igl.signed_distance(point_cloud, v, f)[1]
        centers = v[f].mean(axis=1)
        faces2pts = pcu.k_nearest_neighbors(centers, point_cloud, 5)[1]

    if decimated_v is not None and decimated_f is not None:
        v = decimated_v
        f = decimated_f

    point_cloud_data = point_cloud, point_labels, pts2faces
    mesh_data = mesh_features, mesh_labels, faces2pts, v, f

    write_to_h5(file_path, tmp_dir, point_cloud_data, mesh_data)

    return True, file_path


def write_to_h5(file_path, tmp_dir, point_cloud_data, mesh_data):
    point_cloud, point_labels, pts2faces = point_cloud_data
    mesh_features, mesh_labels, faces2pts, vs, faces = mesh_data

    parts = file_path.split('/')
    with h5py.File(f'{tmp_dir}/{parts[-2]}/{parts[-1][:-4]}.h5', 'w') as h5_file:
        mesh_group = h5_file.create_group('mesh')
        mesh_data_group = mesh_group.create_group('data')
        mesh_data_group.create_dataset('features', data=mesh_features[0],
                                       dtype='f4', compression='gzip')
        mesh_data_group.create_dataset('neighbors', data=mesh_features[1],
                                       dtype='i4', compression='gzip')
        mesh_group.create_dataset('patch_labels', data=np.array([0]), dtype='i4', compression='gzip')
        mesh_group.create_dataset('seg_labels', data=mesh_labels, dtype='i4', compression='gzip')
        mesh_group.create_dataset('faces2pts', data=faces2pts, dtype='i4', compression='gzip')
        mesh_group.create_dataset('vs', data=vs, dtype='f4', compression='gzip')
        mesh_group.create_dataset('faces', data=faces, dtype='i4', compression='gzip')

        pc_group = h5_file.create_group('point_cloud')
        pc_group.create_dataset('data', data=point_cloud, dtype='f4', compression='gzip')
        pc_group.create_dataset('patch_labels', data=np.array([0]), dtype='i4', compression='gzip')
        pc_group.create_dataset('seg_labels', data=point_labels, dtype='i4', compression='gzip')
        pc_group.create_dataset('pts2faces', data=pts2faces, dtype='i4', compression='gzip')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_mask',
                        action='store',
                        type=str,
                        help='mask for npz files')

    parser.add_argument('--output_file',
                        action='store',
                        type=str,
                        help='path to output h5 file')

    parser.add_argument('--num_faces',
                        nargs='?',
                        const=-1,
                        type=int,
                        help='num faces in simplified mesh, default - no simplification')

    parser.add_argument('--num_points',
                        action='store',
                        type=int,
                        help='num point in point cloud')

    return parser.parse_args()


def collect_h5(tmp_dir, output_file):
    with h5py.File(output_file, 'w') as f:

        f.create_group('train')
        f.create_group('test')

        for path in glob(f'{tmp_dir}/*/*.h5'):
            parts = path.split('/')
            split = parts[-2]
            idx = parts[-1][:-3]

            f[split].create_group(idx)

            with h5py.File(path, 'r') as record:
                for group_idx in record.keys():
                    record.copy(group_idx, f[f'{split}/{idx}'])


if __name__ == '__main__':
    opt = parse_arguments()
    files = glob(opt.input_mask)
    log = open('error_log.txt', 'w+')

    working_dir = '/'.join(opt.output_file.split('/')[:-1])

    tmp_dir = Path(f'{working_dir}/tmp_coseg')
    # rmtree(str(tmp_dir))

    (tmp_dir / 'train').mkdir(exist_ok=True, parents=True)
    (tmp_dir / 'test').mkdir(exist_ok=True, parents=True)

    worker = partial(process_mesh,
                     tmp_dir=str(tmp_dir),
                     num_points=opt.num_points,
                     num_faces=opt.num_faces)

    with mp.Pool(mp.cpu_count()) as pool:
        progress_bar = tqdm(pool.imap_unordered(worker, files), leave=True, position=0, total=len(files))
        for status, file_path in progress_bar:
            if not status:
                log.write(f'{file_path}\n')
                continue

    collect_h5(str(tmp_dir), opt.output_file)
