from pycocotools.coco import COCO
from urllib.request import urlretrieve
import multiprocessing as mp
from tqdm import tqdm

ann_file = 'coco/annotations/instances_train2017.json'
coco = COCO(ann_file)
cats_ids = coco.getCatIds(catNms=['cat'])
cats_ids = coco.getImgIds(catIds=cats_ids)

dogs_ids = coco.getCatIds(catNms=['dog'])
dogs_ids = coco.getImgIds(catIds=dogs_ids)

imgs = coco.loadImgs(cats_ids) + coco.loadImgs(dogs_ids)


def download(img):
    fname = img['file_name']
    urlretrieve(img['coco_url'], f'coco/imgs/{fname}')


with mp.Pool(mp.cpu_count()) as pool:
    for _ in tqdm(pool.imap_unordered(download, imgs), leave=True, position=0):
        pass
