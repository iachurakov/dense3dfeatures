import igl
from joblib import Parallel, delayed
import numpy as np
import h5py
import pymeshlab as pml
import point_cloud_utils as pcu
import argparse
from .abc_data import *
from datetime import datetime
from functools import partial
import trimesh
import yaml
from os.path import exists
from scipy.spatial.transform import Rotation as R
from .mesh_segmentation import MeshSegmentation
from .depth_image import DataGenerationException
from .depth_image.imaging import RaycastingImaging
from .depth_image.camera_pose_manager import CompositePoseManager
from .depth_image.camera_pose_manager import SphereOrientedToWorldOrigin, \
    ZRotationInCameraFrame, XYTranslationInCameraFrame
from pathlib import Path
from multiprocessing import cpu_count
from glob import glob

MIN_SHARP_FEATURES = 50
MAX_VERTICES = 70_000

POSE_MANAGER_BY_TYPE = {
    'composite': CompositePoseManager,
    'sphere_to_origin': SphereOrientedToWorldOrigin,
    'z_rotation': ZRotationInCameraFrame,
    'xy_translation': XYTranslationInCameraFrame,
}

IMAGING_BY_TYPE = {
    'raycasting': RaycastingImaging,
}

config = {
    'camera_pose': {
        'type': 'composite',
        'sequences': [
            {
                'type': 'sphere_to_origin',
                'n_images': 10
            },
            {
                'type': 'xy_translation',
                'n_images': 1
            },
            {
                'type': 'z_rotation',
                'n_images': 1
            }
        ]
    },
    'imaging': {
        'type': 'raycasting',
        'projection': 'ortho',
        'resolution_image': 64,
        'resolution_3d': 0.045,
        'validate_image': True
    }
}


def random_rotate(sample, axis='xyz', low=-45, high=45):
    rotation_matrix = R.from_euler(
        axis,
        np.random.randint(low, high + 1, len(axis)),
        degrees=True
    ).as_matrix()

    return sample @ rotation_matrix


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


def cluster_mesh(nu: float, delta: float, vertices: np.ndarray, faces: np.ndarray, n_patches: int):
    """
    Generate patches from mesh

    :param nu: parameter for MeshSegmentation instance
    :param delta: parameter for MeshSegmentation instance
    :param vertices: array of vertices, shape: n x 3
    :param faces: array of faces, shape: m x 3
    :param n_patches: the number of patches
    :return: labels for faces
    """

    fast_fit = faces.shape[0] > MeshSegmentation.max_faces

    return MeshSegmentation(nu, delta, n_patches, fast_fit).fit_predict(vertices, faces)


def get_annotated_mesh(vertices: np.ndarray, faces: np.ndarray,
                       num_faces: int):
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
            return False, (None, None), (None, None)

    v = vertices.copy() if decimated_vertices is None else decimated_vertices
    f = faces.copy() if decimated_faces is None else decimated_faces

    # move to center
    center = (np.max(v, 0) + np.min(v, 0)) / 2
    v = v - center

    # normalize
    max_len = np.max(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
    v = v / np.sqrt(max_len)

    # get normal vector
    face_normals = trimesh.base.Trimesh(v, f, process=False, validate=False).face_normals

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

    return True, (features, neighbors), (decimated_vertices, decimated_faces)


def get_annotated_point_cloud(vertices: np.ndarray,
                              faces: np.ndarray,
                              num_points: int,
                              faces_labels: np.ndarray):
    point_cloud = sample_points(vertices, faces, num_points)

    _, closest_faces, _ = igl.signed_distance(point_cloud, vertices, faces, False)
    points_labels = faces_labels[closest_faces]

    return point_cloud, points_labels


def get_annotated_sdf(
        vertices: np.ndarray,
        faces: np.ndarray,
        faces_labels: np.ndarray,
        seg_labels: np.ndarray,
        resolution: int = 64
):
    grid_margin = 2 * 1. / resolution

    mesh_extend = vertices.max(axis=0) - vertices.min(axis=0)

    scaled_vertices = vertices.view()
    scaled_vertices = scaled_vertices * (2 - 2 * grid_margin) / np.max(mesh_extend)
    scaled_vertices -= ((1 - grid_margin) + scaled_vertices.min(axis=0))

    grid = np.linspace(-1, 1, 64)
    xx, yy, zz = np.meshgrid(grid, grid, grid)
    voxelgrid_xyz = np.stack((xx, yy, zz)).reshape(3, -1).T

    dist, closest_faces, _ = igl.signed_distance(voxelgrid_xyz, scaled_vertices, faces)

    grid_patches = faces_labels[closest_faces]

    return dist, grid_patches, seg_labels[closest_faces]


def get_annotated_depth_images(
        vertices: np.ndarray,
        faces: np.ndarray,
        faces_labels: np.ndarray,
        seg_labels: np.ndarray
):
    mesh = trimesh.base.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False
    )

    # mesh scaling/translation
    mesh_extent = np.max(mesh.bounding_box.extents)
    mesh = mesh.apply_scale(2.0 / mesh_extent)
    mesh_extent = mesh.bounding_box.extents
    mesh_bounds = mesh.bounding_box.bounds
    translation = mesh_bounds[0] + mesh_extent / 2
    mesh = mesh.apply_translation(-translation)

    # create poses #n_images : "sphere_to_origin" in configuration_file
    pose_manager = POSE_MANAGER_BY_TYPE[config['camera_pose']['type']].from_config(config['camera_pose'])
    pose_manager.prepare(mesh)
    imaging = IMAGING_BY_TYPE[config['imaging']['type']].from_config(config['imaging'])
    n_images = []
    patch_maps = []
    seg_maps = []

    for camera_pose in pose_manager:
        try:
            image, patch_map, seg_map = imaging.get_image_from_pose(mesh, faces_labels, seg_labels, camera_pose)
            n_images.append(image)
            patch_maps.append(patch_map)
            seg_maps.append(seg_map)
        except DataGenerationException as e:
            continue

    return np.array(n_images), np.array(patch_maps), np.array(seg_maps)


def log_p(part, idx, msg):
    time_track = datetime.now().strftime('%H:%M:%S:%f %d')
    p = Path(f'datasets/debug/{part}/{idx}')
    p.mkdir(parents=True, exist_ok=True)
    (p / f'{msg}_{time_track}.l').open(mode='w+')


def parse_obj(data):
    file_contents = data.read()
    file_contents = file_contents.decode('utf-8')

    vertices = []
    faces = []

    for line in file_contents.splitlines():
        values = line.strip().split()
        if not values:
            continue

        if values[0] == 'v':
            vertices.append(values[1:4])

        elif values[0] == 'f':
            faces.append([value.split('//')[0] for value in values[1:4]])
        else:
            pass

    return np.array(vertices, dtype='float'), np.array(faces, dtype='int') - 1


def get_n_components(v, f):
    mesh = trimesh.base.Trimesh(v, f, process=False, validate=False)
    return trimesh.graph.connected_component_labels(mesh.edges).max()


def get_record(item, tmp_dir, nu, delta, num_patches, num_faces, num_points):
    vertices, faces = parse_obj(item.obj)

    if vertices.shape[0] > MAX_VERTICES:
        return False, 'max_vertices', item.item_id

    features = yaml.load(item.feat, Loader=yaml.Loader)
    if len([c for c in features['curves'] if c['sharp']]) < MIN_SHARP_FEATURES:
        return False, 'min_sharp_features', item.item_id

    if get_n_components(vertices, faces) > 0:
        return False, 'n_components', item.item_id

    center = (vertices.max(0) + vertices.min(0)) / 2
    vertices -= center
    max_len = np.linalg.norm(vertices, axis=1).max()
    vertices /= max_len

    mesh_set = pml.MeshSet()
    mesh_set.add_mesh(pml.Mesh(vertices.copy(), faces.copy()))
    cleaned_vertices, cleaned_faces = clean_mesh(mesh_set)

    vertices, faces = cleaned_vertices, cleaned_faces

    try:
        faces_labels = cluster_mesh(nu, delta, vertices, faces, num_patches)
    except:
        return False, 'clustering', item.item_id

    status, (features, neighbors), (decimated_v, decimated_f) = get_annotated_mesh(vertices, faces, num_faces)

    if not status:
        return False, 'mesh_features', item.item_id

    point_cloud_data = get_annotated_point_cloud(vertices, faces, num_points,
                                                 faces_labels)

    if decimated_v is not None and decimated_f is not None:
        pts2faces = igl.signed_distance(point_cloud_data[0], decimated_v, decimated_f)[1]
        centers = decimated_v[decimated_f].mean(axis=1)
        faces2pts = pcu.k_nearest_neighbors(centers, point_cloud_data[0], 1)[1]
        closest = igl.signed_distance(centers, vertices, faces)[1]
        faces_labels_decimated = faces_labels[closest]
        mesh_data = features, neighbors, faces_labels_decimated, decimated_v, decimated_f
    else:
        pts2faces = igl.signed_distance(point_cloud_data[0], vertices, faces)[1]
        centers = vertices[faces].mean(axis=1)
        faces2pts = pcu.k_nearest_neighbors(centers, point_cloud_data[0], 1)[1]
        mesh_data = features, neighbors, faces_labels, vertices, faces

    available_patches = set(mesh_data[2]) & set(point_cloud_data[1])
    available_patches = np.array(list(available_patches))

    write_to_h5(item.item_id, tmp_dir, mesh_data, faces2pts, point_cloud_data, pts2faces, available_patches)
    return True, 'ok', item.item_id


def parse_arguments():
    parser = argparse.ArgumentParser(description='Multimodal dataset preparation (ABC, mesh and point cloud)')

    parser.add_argument('--chunk_path',
                        action='store',
                        type=str,
                        help='path to dir with ABC chunks')

    parser.add_argument('--chunk_num',
                        action='store',
                        type=str,
                        help='num of chunk')

    parser.add_argument('--output_file',
                        action='store',
                        type=str,
                        help='path to output h5 file')

    parser.add_argument('--delta',
                        action='store',
                        type=float,
                        help='delta param for mesh segmentation')

    parser.add_argument('--nu',
                        action='store',
                        type=float,
                        help='nu param for mesh segmentation')

    parser.add_argument('--num_patches',
                        action='store',
                        type=int,
                        help='num of patches')

    parser.add_argument('--num_jobs',
                        action='store',
                        type=int,
                        help='num of jobs')

    parser.add_argument('--num_faces',
                        nargs='?',
                        const=-1,
                        type=int,
                        help='num faces in simplified mesh, default - no simplification')

    parser.add_argument('--num_points',
                        action='store',
                        type=int,
                        help='num point in point cloud')

    parser.add_argument('--skip_indices',
                        action='store',
                        type=str,
                        help='path to file with items\' ids to skip',
                        default='')

    return parser.parse_args()


def process_chunk(chunk_start, chunk_end, chunk_path, chunk_num,
                  tmp_dir, nu, delta, num_patches, num_faces,
                  num_points, skip_ids):
    obj_filename = ABC_7Z_FILEMASK.format(
        chunk=chunk_num,
        modality=ABCModality.OBJ.value,
        version='00'
    )

    feat_filename = ABC_7Z_FILEMASK.format(
        chunk=chunk_num,
        modality=ABCModality.FEAT.value,
        version='00'
    )

    obj_path = f'{chunk_path}/{obj_filename}'
    feat_path = f'{chunk_path}/{feat_filename}'

    with ABCChunk([obj_path, feat_path]) as data_holder:
        for item in data_holder[chunk_start:chunk_end]:
            if exists(f'{tmp_dir}/{item.item_id}.h5') or item.item_id in skip_ids:
                continue
            status, reason, item_id = get_record(item, tmp_dir, nu, delta, num_patches, num_faces, num_points)
            # print('End item:', item_id, reason)


def write_to_h5(idx, tmp_dir, mesh_data, faces2pts, point_cloud_data, pts2faces, available_patches):
    with h5py.File(f'{tmp_dir}/{idx}.h5', 'w') as h5_file:
        mesh_group = h5_file.create_group('mesh')
        mesh_data_group = mesh_group.create_group('data')
        mesh_data_group.create_dataset('features', data=mesh_data[0],
                                       dtype='f4', compression='gzip')
        mesh_data_group.create_dataset('neighbors', data=mesh_data[1],
                                       dtype='i4', compression='gzip')
        mesh_group.create_dataset('patch_labels', data=mesh_data[2], dtype='i4', compression='gzip')
        mesh_group.create_dataset('vs', data=mesh_data[3], dtype='f4', compression='gzip')
        mesh_group.create_dataset('faces', data=mesh_data[4], dtype='i4', compression='gzip')
        mesh_group.create_dataset('faces2pts', data=faces2pts, dtype='i4', compression='gzip')

        point_cloud_group = h5_file.create_group('point_cloud')
        point_cloud_group.create_dataset('data', data=point_cloud_data[0],
                                         dtype='f4', compression='gzip')
        point_cloud_group.create_dataset('patch_labels', dtype='i4', data=point_cloud_data[1], compression='gzip')
        point_cloud_group.create_dataset('pts2faces', data=pts2faces, dtype='i4', compression='gzip')

        h5_file.create_dataset('available_patches', data=available_patches, dtype='i4')


def collect_h5(tmp_dir, output_file):
    with h5py.File(output_file, 'w') as f:
        for path in glob(f'{tmp_dir}/*.h5'):
            idx = path.split('/')[-1][:-3]
            f.create_group(idx)

            with h5py.File(path, 'r') as record:
                for group_idx in record.keys():
                    record.copy(group_idx, f[idx])


if __name__ == '__main__':
    opt = parse_arguments()

    working_dir = '/'.join(opt.output_file.split('/')[:-1])
    tmp_dir = Path(f'{working_dir}/tmp2_chunk_{opt.chunk_num}')
    tmp_dir.mkdir(exist_ok=True, parents=True)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    tmp_dir.mkdir(exist_ok=True, parents=True)

    skip_ids = set()
    if opt.skip_indices != '':
        with open(opt.skip_indices) as f:
            skip_ids = set([idx.strip() for idx in f.readlines()])

    process_fn = partial(process_chunk,
                         chunk_path=opt.chunk_path,
                         chunk_num=opt.chunk_num,
                         tmp_dir=str(tmp_dir),
                         nu=opt.nu,
                         delta=opt.delta,
                         num_patches=opt.num_patches,
                         num_faces=opt.num_faces,
                         num_points=opt.num_points,
                         skip_ids=skip_ids)

    obj_filename = ABC_7Z_FILEMASK.format(
        chunk=opt.chunk_num,
        modality=ABCModality.OBJ.value,
        version='00'
    )

    obj_path = f'{opt.chunk_path}/{obj_filename}'

    with ABCChunk([obj_path]) as abc_data:
        total_items = len(abc_data)

    print('Total:', total_items)
    processes_to_spawn = 10 * cpu_count()
    chunk_size = max(1, total_items // processes_to_spawn)
    abc_data_slices = [(start, start + chunk_size)
                       for start in range(0, total_items, chunk_size)]

    results = Parallel(n_jobs=opt.num_jobs, backend='multiprocessing', verbose=50)(delayed(process_fn)(start, end) for start, end in abc_data_slices)
    collect_h5(str(tmp_dir), opt.output_file)
