from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import igl
import trimesh

from . import DataGenerationException
from .camera_pose import CameraPose
from .raycasting import generate_rays, ray_cast_mesh


class ImagingFunc(ABC):
    """Implements obtaining depth maps from meshes."""

    @abstractmethod
    def get_image_from_pose(self, mesh, faces_labels, seg_labels, pose, *args, **kwargs):
        """Extracts a point cloud.

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)

        :param pose: camera pose to shoot from
        :type pose: CameraPose
        """
        pass


class RaycastingImaging(ImagingFunc):
    def __init__(self, resolution_image, resolution_3d, projection=None, validate_image=False):
        if isinstance(resolution_image, tuple):
            assert len(resolution_image) == 2
        else:
            resolution_image = (resolution_image, resolution_image)
        self.resolution_image = resolution_image
        self.resolution_3d = resolution_3d
        self.projection = projection
        self.validate_image = validate_image
        self.rays_screen_coords, self.rays_origins, self.rays_directions = generate_rays(
            self.resolution_image, self.resolution_3d)

    @classmethod
    def from_config(cls, config):
        return cls(config['resolution_image'],
                   config['resolution_3d'],
                   config['projection'],
                   config['validate_image'])

    def get_image_from_pose(self, mesh: trimesh.base.Trimesh, faces_labels: np.ndarray, seg_labels: np.ndarray,
                            pose: CameraPose, **kwargs):

        if any(value is None for value in [self.rays_screen_coords,
                                           self.rays_origins,
                                           self.rays_directions]):
            raise DataGenerationException('Raycasting was not prepared')

        # get a point cloud with corresponding indexes
        mesh_face_indexes, ray_indexes, points = ray_cast_mesh(
            mesh,
            pose.camera_to_world(self.rays_origins),
            pose.camera_to_world(self.rays_directions, translate=0)
        )

        if len(points) == 0:  # we hit nothing; discard this attempt
            raise DataGenerationException('Object out of frame; discarding patch')

        # patches
        _, closest_faces, _ = igl.signed_distance(points, mesh.vertices, mesh.faces, False)
        point_patch_labels = faces_labels[closest_faces]
        patches = []

        for patch_idx in np.unique(point_patch_labels):
            patches.append(np.where(point_patch_labels == patch_idx)[0])

        seg_patches = []
        seg_patch_labels = seg_labels[closest_faces]

        #for seg_label in np.unique(seg_patch_labels):
        #    seg_patches.append(np.where(seg_patch_labels == seg_label)[0])

        # compute an image, defined in camera frame
        '''image, patch_map, seg_map = self.points_to_image(pose.world_to_camera(points),
                                                                     ray_indexes,
                                                                     zip(np.unique(faces_labels), patches),
                                                                     zip(np.unique(seg_patch_labels), seg_patches))'''

        for seg_label in np.unique(seg_patch_labels):
            seg_patches.append(np.where(seg_patch_labels == seg_label)[0])

        image, patch_map, seg_map = self.points_to_image(pose.world_to_camera(points),
                                                         ray_indexes,
                                                         zip(np.unique(point_patch_labels), patches),
                                                         zip(np.unique(seg_patch_labels), seg_patches))

        if self.validate_image and np.any(image < 0.):
            raise DataGenerationException('Negative values found in depthmap; discarding image')

        return image, patch_map, seg_map

    def points_to_image(self, points, ray_indexes, patches, seg_patches, assign_channels=None):
        xy_to_ij = self.rays_screen_coords[ray_indexes]
        # note that `points` can have many data dimensions
        if None is assign_channels:
            assign_channels = [2]
        data_channels = len(assign_channels)
        image = np.zeros((self.resolution_image[1], self.resolution_image[0], data_channels))
        # rays origins (h, w, 3), z is the same for all points of matrix
        # distance is absolute value
        image[xy_to_ij[:, 0], xy_to_ij[:, 1]] = points[:, assign_channels]

        patch_map = np.full((self.resolution_image[1], self.resolution_image[0]), -1)
        seg_map = np.full((self.resolution_image[1], self.resolution_image[0]), -1)

        for patch_label, patch in patches:
            patch_map[xy_to_ij[patch][:, 0], xy_to_ij[patch][:, 1]] = patch_label

        for seg_label, seg_patch in seg_patches:
            seg_map[xy_to_ij[seg_patch][:, 0], xy_to_ij[seg_patch][:, 1]] = seg_label

        return image.squeeze(), patch_map, seg_map

    def image_to_points(self, image):
        i = np.where(image.ravel() != 0)[0]
        points = np.zeros((len(i), 3))
        points[:, 0] = self.rays_origins[i, 0]
        points[:, 1] = self.rays_origins[i, 1]
        points[:, 2] = image.ravel()[i]
        return points


IMAGING_BY_TYPE = {
    'raycasting': RaycastingImaging,
}
