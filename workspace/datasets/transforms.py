import numpy as np
from scipy.spatial.transform import Rotation as R


class PointCloudNormalize:
    def __call__(self, x):
        # move to center
        center = (np.max(x, 0) + np.min(x, 0)) / 2
        x = x - center

        # normalize
        max_len = np.max(np.linalg.norm(x, axis=1))
        return x / max_len


class RandomRotation:
    def __init__(self, low, high, axis):
        self.low = low
        self.high = high
        self.axis = axis

    def __call__(self, x):
        rot_matrix = self.get_rotation_matrix()
        return x @ rot_matrix

    def get_rotation_matrix(self):
        rotation_matrix = R.from_euler(
            self.axis,
            np.random.randint(self.low, self.high + 1, len(self.axis)),
            degrees=True
        ).as_matrix()

        return rotation_matrix


class RandomJitter:
    def __init__(self, std, clip_bound):
        self.std = std
        self.clip_bound = clip_bound

    def __call__(self, x):
        noise = self.std * np.random.randn(*x.shape)
        noise = np.clip(noise, -self.clip_bound, self.clip_bound)

        return x + noise


class RandomScale:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, x):
        scale_vector = np.random.uniform(self.low, self.high, 3)
        return x * scale_vector


class MeshNetRandomRotation(RandomRotation):
    def __init__(self, low, high, axis):
        super().__init__(low, high, axis)

    def __call__(self, x):
        return super().__call__(x.reshape(-1, 5, 3)).reshape(-1, 15)


class MeshNetRandomJitter(RandomJitter):
    def __init__(self, std, clip_bound):
        super().__init__(std, clip_bound)

    def __call__(self, x):
        jittered = super().__call__(x[:, :12])
        return np.concatenate((jittered, x[:, 12:]), 1)


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)

        return x
