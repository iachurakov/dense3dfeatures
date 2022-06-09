from torch.nn import functional as F
from ...datasets.transforms import *
import k3d


def visualize_elements_heatmap_pc(point_cloud, features, anchor_idx=-1):
    '''
    :param point_cloud: point cloud, tensor of size (n_points, 3)
    :param features: point cloud features, tensor of size (n_points, emb_dim)
    '''
    if anchor_idx == -1:
        anchor_idx = np.random.randint(0, point_cloud.size(0), size=(1,))[0]
        
    features = F.normalize(features, dim=-1)
                
    sims = features[anchor_idx] @ features.t()
    plot = k3d.plot()
    plot += k3d.points(point_cloud, point_size=0.025, attribute=sims)
    plot += k3d.points(point_cloud[anchor_idx].unsqueeze(0), point_size=0.05)

    return plot


def visualize_elements_heatmap_mesh(mesh, features, anchor_idx=-1):
    '''
    :param mesh: tuple (vertices, faces)
    :param features: faces features, tensor of size (>=n_faces, emb_dim)
    '''
    vertices, faces = mesh
    vertices = PointCloudNormalize()(vertices)
    faces_num = faces.shape[0]
    features = features[:faces_num]
    
    if anchor_idx == -1:
        anchor_idx = np.random.randint(0, faces_num, size=(1,))[0]
        
    features = F.normalize(features, dim=-1)
    sims = features[anchor_idx] @ features.t()
    plot = k3d.plot()
    
    anchor_face = faces[anchor_idx]
    anchor_point = (vertices[anchor_face[0]] +
                    vertices[anchor_face[1]] +
                    vertices[anchor_face[2]]) / 3
    plot += k3d.mesh(vertices, faces, triangles_attribute=sims[:faces.shape[0]])
    plot += k3d.points(anchor_point[None, ...], point_size=0.1, color=0xff0000)
    
    return plot


def visualize_elements_heatmap_pc_to_mesh(point_cloud, mesh, features_pc, features_mesh, anchor_idx=-1):
    '''
    :param point_cloud: point cloud, tensor of size (n_points, 3)
    :param mesh: tuple (vertices, faces)
    :param features_pc: point cloud features, tensor of size (n_points, emb_dim)
    :param features_mesh: faces features, tensor of size (>=n_faces, emb_dim)
    '''
    vertices, faces = mesh
    faces_num = faces.shape[0]
    features_mesh = features_mesh[:faces_num]
    features_mesh = F.normalize(features_mesh, dim=-1)
    features_pc = F.normalize(features_pc, dim=-1)
    
    
    if anchor_idx == -1:
        anchor_idx = np.random.randint(0, point_cloud.size(0), size=(1,))[0]
        
    sims_pc = features_pc[anchor_idx] @ features_pc.T
    sims_mesh = features_pc[anchor_idx] @ features_mesh.T
    plot = k3d.plot()
    vertices = PointCloudNormalize()(vertices)
    vertices[:, 0] += 2
    
    plot += k3d.mesh(vertices, faces, triangles_attribute=sims_mesh)
    plot += k3d.points(point_cloud, point_size=0.025, attribute=sims_pc)
    plot += k3d.points(point_cloud[anchor_idx].unsqueeze(0), point_size=0.1, color=0xff0000)

    return plot


def visualize_elements_heatmap_mesh_to_pc(point_cloud, mesh, features_pc, features_mesh, anchor_idx=-1):
    '''
    :param point_cloud: point cloud, tensor of size (n_points, 3)
    :param mesh: tuple (vertices, faces)
    :param features_pc: point cloud features, tensor of size (n_points, emb_dim)
    :param features_mesh: faces features, tensor of size (>=n_faces, emb_dim)
    '''
    vertices, faces = mesh
    faces_num = faces.shape[0]
    features_mesh = features_mesh[:faces_num]
    features_mesh = F.normalize(features_mesh, dim=-1)
    features_pc = F.normalize(features_pc, dim=-1)
    
    
    if anchor_idx == -1:
        anchor_idx = np.random.randint(0, faces_num, size=(1,))[0]
        
    sims_pc = features_mesh[anchor_idx] @ features_pc.T
    sims_mesh = features_mesh[anchor_idx] @ features_mesh.T
    plot = k3d.plot()
    vertices = PointCloudNormalize()(vertices)
    vertices[:, 0] += 2
    
    anchor_face = faces[anchor_idx]
    anchor_point = (vertices[anchor_face[0]] +
                    vertices[anchor_face[1]] +
                    vertices[anchor_face[2]]) / 3
    
    plot += k3d.mesh(vertices, faces, triangles_attribute=sims_mesh)
    plot += k3d.points(point_cloud, point_size=0.025, attribute=sims_pc)
    plot += k3d.points(anchor_point[None, ...], point_size=0.1, color=0xff0000)

    return plot