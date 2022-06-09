class DatasetMeta:
    dataset_name = None
    n_classes = None
    n_parts = None
    cat2label = None
    label2num = None
    n_patches = None
    seg_labels = False


class CosegMeta(DatasetMeta):
    dataset_name = 'coseg_aliens'
    n_classes = 1
    n_parts = 4
    cat2label = {
        'Aliens': '0'
    }
    label2num = {
        '0': 0
    }
    n_patches = 5
    seg_labels = True


class ShapeNetMeta(DatasetMeta):
    dataset_name = 'shapenet_part'
    n_classes = 16
    n_parts = 50
    cat2label = {
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
    offsets = {
        '02691156': 0,
        '02773838': 4,
        '02954340': 6,
        '02958343': 8,
        '03001627': 12,
        '03261776': 16,
        '03467517': 19,
        '03624134': 22,
        '03636649': 24,
        '03642806': 28,
        '03790512': 30,
        '03797390': 36,
        '03948459': 38,
        '04099429': 41,
        '04225987': 44,
        '04379243': 47
    }
    label2num = {key: idx for idx, key in enumerate(cat2label.values())}
    n_patches = 10
    seg_labels = True


class ABCMeta(DatasetMeta):
    dataset_name = 'abc'
    n_classes = 0
    n_patches = 5


datasets_meta_dict = {
    'shapenet_part': ShapeNetMeta,
    'coseg': CosegMeta,
    'abc': ABCMeta
}
