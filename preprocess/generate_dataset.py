import igl
from joblib import Parallel, delayed
import numpy as np
import h5py
import pymeshlab as pml
import point_cloud_utils as pcu
import argparse
from glob import glob
from datetime import datetime
from functools import partial
from trimesh.base import Trimesh
from scipy.spatial.transform import Rotation as R
from .mesh_segmentation import MeshSegmentation
from depth_image import DataGenerationException
from depth_image.imaging import RaycastingImaging
from depth_image.camera_pose_manager import CompositePoseManager
from depth_image.camera_pose_manager import SphereOrientedToWorldOrigin, \
    ZRotationInCameraFrame, XYTranslationInCameraFrame
from pathlib import Path


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

    return True, (features, neighbors), (decimated_vertices, decimated_faces)


def get_annotated_point_cloud(vertices: np.ndarray,
                              faces: np.ndarray,
                              num_points: int,
                              faces_labels: np.ndarray,
                              seg_labels: np.ndarray):

    point_cloud = sample_points(vertices, faces, num_points)
    _, closest_faces, _ = igl.signed_distance(point_cloud, vertices, faces, False)
    points_labels = faces_labels[closest_faces]

    return point_cloud, points_labels, seg_labels[closest_faces]


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
    mesh = Trimesh(
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


def get_record(file_path, tmp_dir, nu, delta, num_rotations, num_patches, num_faces, num_points, vox_res):
    parts = file_path.split('/')
    split = parts[-2]
    idx = parts[-1].replace('.npz', '')

    mesh = np.load(file_path)
    faces = mesh['faces']
    vertices = mesh['vs']
    seg_labels = mesh['labels']

    mesh_set = pml.MeshSet()
    mesh_set.add_mesh(pml.Mesh(vertices.copy(), faces.copy()))
    cleaned_vertices, cleaned_faces = clean_mesh(mesh_set)

    # correct segmentation labels after cleaning
    new_centers = cleaned_vertices[cleaned_faces].mean(axis=1)
    closest_faces = igl.signed_distance(new_centers, vertices, faces)[1]
    seg_labels = seg_labels[closest_faces]
    # swap to use cleaned mesh
    vertices, faces = cleaned_vertices, cleaned_faces

    # move to center
    center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
    vertices = vertices - center

    # normalize
    max_len = np.max(vertices[:, 0] ** 2 + vertices[:, 1] ** 2 + vertices[:, 2] ** 2)
    vertices = vertices / np.sqrt(max_len)

    try:
        faces_labels = cluster_mesh(nu, delta, vertices, faces, num_patches)
    except:
        return False, file_path

    status, (features, neighbors), (decimated_v, decimated_f) = get_annotated_mesh(vertices, faces, num_faces)

    if not status:
        return False, file_path

    point_cloud_data = get_annotated_point_cloud(vertices, faces, num_points,
                                                 faces_labels, seg_labels)

    if decimated_v is not None and decimated_f is not None:
        pts2faces = igl.signed_distance(point_cloud_data[0], decimated_v, decimated_f)[1]
        centers = decimated_v[decimated_f].mean(axis=1)
        faces2pts = pcu.k_nearest_neighbors(centers, point_cloud_data[0], 1)[1]
        closest = igl.signed_distance(centers, vertices, faces)[1]
        faces_labels_decimated = faces_labels[closest]
        seg_labels_decimated = seg_labels[closest]
        mesh_data = features, neighbors, faces_labels_decimated, seg_labels_decimated, decimated_v, decimated_f
    else:
        pts2faces = igl.signed_distance(point_cloud_data[0], vertices, faces)[1]
        centers = vertices[faces].mean(axis=1)
        faces2pts = pcu.k_nearest_neighbors(centers, point_cloud_data[0], 1)[1]
        mesh_data = features, neighbors, faces_labels, seg_labels, vertices, faces

    sdf_data = get_annotated_sdf(vertices, faces, faces_labels, seg_labels, vox_res)

    sdf_dists = [sdf_data[0]]
    sdf_grid_patches = [sdf_data[1]]
    sdf_seg_labels = [sdf_data[2]]

    depth_image_data = get_annotated_depth_images(vertices, faces, faces_labels, seg_labels)

    depth_images = [depth_image_data[0]]
    depth_images_patch_labels = [depth_image_data[1]]
    depth_images_seg_labels = [depth_image_data[2]]

    for i in range(num_rotations):
        rotated_vs = random_rotate(vertices)

        depth_image_data = get_annotated_depth_images(rotated_vs, faces, faces_labels, seg_labels)
        depth_images.append(depth_image_data[0])
        depth_images_patch_labels.append(depth_image_data[1])
        depth_images_seg_labels.append(depth_image_data[2])

        sdf_data = get_annotated_sdf(rotated_vs, faces, faces_labels, seg_labels, vox_res)
        sdf_dists.append(sdf_data[0])
        sdf_grid_patches.append(sdf_data[1])
        sdf_seg_labels.append(sdf_data[2])

    sdf_data = (np.vstack(sdf_dists), np.vstack(sdf_grid_patches), np.vstack(sdf_seg_labels))
    depth_image_data = (np.vstack(depth_images), np.vstack(depth_images_patch_labels), np.vstack(depth_images_seg_labels))

    sdf_patches = set(sdf_data[1].flatten())
    depth_images_patches = set(depth_image_data[1].flatten())

    available_patches = set(mesh_data[2]) & set(point_cloud_data[1]) & sdf_patches & depth_images_patches
    available_patches = np.array(list(available_patches))

    write_to_h5(split, idx, tmp_dir, mesh_data, faces2pts, point_cloud_data, pts2faces,
                sdf_data, depth_image_data, available_patches)
    return True, file_path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Multimodal dataset preparation')

    parser.add_argument('--input_mask',
                        action='store',
                        type=str,
                        help='mask for npz files with vertices and faces')

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

    parser.add_argument('--num_rot',
                        action='store',
                        type=int,
                        help='num of mesh rotation')

    parser.add_argument('--num_patches',
                        action='store',
                        type=int,
                        help='num of patches')

    parser.add_argument('--num_depth_imgs',
                        action='store',
                        type=int,
                        help='num of view for depth images generation')

    parser.add_argument('--depth_img_res',
                        action='store',
                        type=int,
                        help='resolution of depth image')

    parser.add_argument('--num_faces',
                        nargs='?',
                        const=-1,
                        type=int,
                        help='num faces in simplified mesh, default - no simplification')

    parser.add_argument('--num_points',
                        action='store',
                        type=int,
                        help='num point in point cloud')

    parser.add_argument('--vox_res',
                        action='store',
                        type=int,
                        help='resolution of voxel grid')

    return parser.parse_args()


def write_to_h5(split, idx, tmp_dir, mesh_data, faces2pts, point_cloud_data, pts2faces,
                sdf_data, depth_image_data, available_patches):

    with h5py.File(f'{tmp_dir}/{split}/{idx}.h5', 'w') as h5_file:
        mesh_group = h5_file.create_group('mesh')
        mesh_data_group = mesh_group.create_group('data')
        mesh_data_group.create_dataset('features', data=mesh_data[0],
                                       dtype='f4', compression='gzip')
        mesh_data_group.create_dataset('neighbors', data=mesh_data[1],
                                       dtype='i4', compression='gzip')
        mesh_group.create_dataset('patch_labels', data=mesh_data[2], dtype='i4', compression='gzip')
        mesh_group.create_dataset('seg_labels', data=mesh_data[3], dtype='i4', compression='gzip')
        mesh_group.create_dataset('vs', data=mesh_data[4], dtype='f4', compression='gzip')
        mesh_group.create_dataset('faces', data=mesh_data[5], dtype='i4', compression='gzip')
        mesh_group.create_dataset('faces2pts', data=faces2pts, dtype='i4', compression='gzip')

        point_cloud_group = h5_file.create_group('point_cloud')
        point_cloud_group.create_dataset('data', data=point_cloud_data[0],
                                         dtype='f4', compression='gzip')
        point_cloud_group.create_dataset('patch_labels', dtype='i4', data=point_cloud_data[1], compression='gzip')
        point_cloud_group.create_dataset('seg_labels', dtype='i4', data=point_cloud_data[2], compression='gzip')
        point_cloud_group.create_dataset('pts2faces', data=pts2faces, dtype='i4', compression='gzip')

        sdf_group = h5_file.create_group('sdf')
        sdf_group.create_dataset('data', data=sdf_data[0], dtype='f4', compression='gzip')
        sdf_group.create_dataset('patch_labels', data=sdf_data[1], dtype='i4', compression='gzip')
        sdf_group.create_dataset('seg_labels', data=sdf_data[2], dtype='i4', compression='gzip')

        depth_image_group = h5_file.create_group('depth_images')
        depth_image_group.create_dataset('data', data=depth_image_data[0], compression='gzip')
        depth_image_group.create_dataset('patch_labels', data=depth_image_data[1], dtype='i4', compression='gzip')
        depth_image_group.create_dataset('seg_labels', data=depth_image_data[2], dtype='i4', compression='gzip')

        h5_file.create_dataset('available_patches', data=available_patches, dtype='i4')


def collect_h5(tmp_dir, output_file):
    with h5py.File(output_file, 'w') as f:

        f.create_group('train')
        f.create_group('val')
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

    config['camera_pose']['sequences'][0]['n_images'] = opt.num_depth_imgs
    config['imaging']['resolution_image'] = opt.depth_img_res

    working_dir = '/'.join(opt.output_file.split('/')[:-1])

    tmp_dir = Path(f'{working_dir}/tmp')

    (tmp_dir / 'train').mkdir(exist_ok=True, parents=True)
    (tmp_dir / 'val').mkdir(exist_ok=True, parents=True)
    (tmp_dir / 'test').mkdir(exist_ok=True, parents=True)

    get_record_fn = partial(get_record,
                            tmp_dir=str(tmp_dir),
                            nu=opt.nu,
                            delta=opt.delta,
                            num_rotations=opt.num_rot,
                            num_patches=opt.num_patches,
                            num_faces=opt.num_faces,
                            num_points=opt.num_points,
                            vox_res=opt.vox_res)

    results = Parallel(n_jobs=-1, verbose=50)(delayed(get_record_fn)(file) for file in files)
    collect_h5(str(tmp_dir), opt.output_file)
    with open('error_log.txt', 'w+') as log:
        for status, path in results:
            if not status:
                log.write(path + '\n')
