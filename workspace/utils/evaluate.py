import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        super().__init__()
        self.embeddings = embeddings
        self.batch_size = embeddings[0].shape[0]
        self.len_of_last_batch = embeddings[-1].shape[0]
        self.len = (len(self.embeddings) - 1) * self.batch_size + self.len_of_last_batch
        self.labels = labels

    def __getitem__(self, idx):
        batch_num, sample_in_batch = divmod(idx, self.batch_size)
        return self.embeddings[batch_num][sample_in_batch], self.labels[batch_num][sample_in_batch]

    def __len__(self):
        return self.len


def send_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)

    return [t.to(device) for t in data]


class SegmentationModel(nn.Module):
    def __init__(self,
                 feature_extractor,
                 n_features,
                 train_loader,
                 val_loader,
                 train_only_top,
                 device):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = nn.Conv1d(n_features, train_loader.dataset.n_parts, 1).to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_only_top = train_only_top
        self.device = device
        self.sklearn_model = None

        if self.train_only_top:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()

    def forward(self, x):
        features = self.feature_extractor.forward_features(x)
        return self.head(features)

    def parameters(self, recurse: bool = True):
        params = super().parameters()
        return filter(lambda p: p.requires_grad, params)

    def get_embeddings(self, loader, get_patch_embedding=False):
        self.feature_extractor.eval()

        per_elem_features = []
        patch_labels = []
        labels = []

        with torch.no_grad():
            for data, patch_labels_, seg_labels in tqdm(loader, leave=True, position=0):
                features = self.feature_extractor.forward_features(send_to_device(data, self.device))
                per_elem_features.append(features.cpu())
                patch_labels.append(patch_labels_.cpu())
                labels.append(seg_labels.long())

        # per_elem_features = torch.cat(per_elem_features, dim=0)
        # patch_labels = torch.cat(patch_labels, dim=0)
        # labels = torch.cat(labels, dim=0)

        if not get_patch_embedding:
            return per_elem_features, labels

        patch_embeddings = []

        for shape_features, shape_patch_labels in zip(per_elem_features, patch_labels):
            patch_embeddings.append(self.feature_extractor.get_patch_embeddings(shape_features.to(self.device).unsqueeze(0),
                                                                                shape_patch_labels.to(self.device).unsqueeze(0)).cpu())

        return per_elem_features, patch_embeddings, labels

    def train_model(self, epoch_num, optimizer, scheduler=None):
        current_loss = 0
        current_iter = 0

        for epoch_n in range(epoch_num):
            if self.train_only_top:
                self.head.train()
            else:
                self.train()

            bar = tqdm(self.train_loader, leave=True, position=0)

            for features, _, labels in bar:
                optimizer.zero_grad()
                logits = self(send_to_device(features, self.device))
                loss = F.cross_entropy(logits, labels.long().to(self.device), ignore_index=-1)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                current_iter += 1

                bar.set_postfix({'Epoch': epoch_n,
                                 'Loss': current_loss / current_iter
                                 })

                if scheduler is not None:
                    scheduler.step()

            if self.val_loader is not None:
                print('Val loss', self.calc_val_loss())

    def train_on_fixed_embeddings(self, epoch_num, optimizer, scheduler=None):
        train_embeddings, train_labels = self.get_embeddings(self.train_loader)
        train_loader = torch.utils.data.DataLoader(EmbeddingDataset(train_embeddings, train_labels),
                                                   shuffle=True,
                                                   batch_size=64)
        if self.val_loader is not None:
            val_embeddings, val_labels = self.get_embeddings(self.val_loader)
            val_loader = torch.utils.data.DataLoader(EmbeddingDataset(val_embeddings, val_labels),
                                                     shuffle=True,
                                                     batch_size=64)

        current_loss = 0
        current_iter = 0

        for epoch_n in range(epoch_num):
            self.head.train()
            bar = tqdm(train_loader, leave=True, position=0)
            for features, labels in bar:
                optimizer.zero_grad()
                logits = self.head(send_to_device(features, self.device))
                loss = F.cross_entropy(logits, labels.to(self.device), ignore_index=-1)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                current_iter += 1

                bar.set_postfix({'Epoch': epoch_n,
                                 'Loss': current_loss / current_iter
                                 })

                if scheduler is not None:
                    scheduler.step()

            if self.val_loader is not None:
                print('Val loss', self.calc_val_loss(val_loader))

    def train_sklearn(self, C=1):
        train_embeddings, labels = self.get_embeddings(self.train_loader)
        train_embeddings = train_embeddings.transpose(2, 1).reshape(-1, train_embeddings.size(2)).numpy()
        labels = labels.reshape(-1).numpy()
        self.sklearn_model = LogisticRegression(C=C, verbose=2, n_jobs=-1)
        self.sklearn_model.fit(train_embeddings, labels)

    def calc_val_loss(self, loader=None):
        self.eval()
        loss = 0
        val_loader = self.val_loader if loader is None else loader
        with torch.no_grad():
            for sample in val_loader:
                if len(sample) == 3:
                    features, _, labels = sample
                else:
                    features, labels = sample
                if loader is None:
                    logits = self(send_to_device(features, self.device))
                else:
                    logits = self.head(send_to_device(features, self.device))
                loss += F.cross_entropy(logits, labels.long().to(self.device), ignore_index=-1).item()

        return loss / len(val_loader)


class Evaluator:
    def __init__(self,
                 model,
                 test_loader,
                 device,
                 mask_pred=True):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.mask_pred = mask_pred

    def calc_iou(self, target, pred, mask_pred=True):
        classes = torch.arange(self.test_loader.dataset.n_parts)[:, None]

        oh_target = target[:, None] == classes
        oh_pred = pred[:, None] == classes

        if mask_pred:
            mask = target != -1
            oh_pred = oh_pred & mask.unsqueeze(1)

        intersection = (oh_target & oh_pred).sum(axis=-1)
        union = (oh_target | oh_pred).sum(axis=-1)
        iou = intersection / torch.where(union == 0, 1, union)
        present_parts_count = (union != 0).sum(axis=1)
        return iou.sum(axis=1) / present_parts_count

    def eval(self, predict_fn):
        cat_counter = torch.zeros(self.test_loader.dataset.n_classes)
        cat2iou = torch.zeros(self.test_loader.dataset.n_classes)
        cat2acc = torch.zeros(self.test_loader.dataset.n_classes)
        all_shapes_iou = []
        with torch.no_grad():
            for features, _, labels, inst_labels in tqdm(self.test_loader, leave=True, position=0):
                pred = predict_fn(features)
                batch_acc = (pred == labels).float().mean(1)
                batch_iou = self.calc_iou(labels, pred, self.mask_pred)
                for label, sample_acc, sample_iou in zip(inst_labels, batch_acc, batch_iou):
                    cat2acc[label] += sample_acc
                    cat2iou[label] += sample_iou
                    cat_counter[label] += 1

                all_shapes_iou.append(batch_iou)

        cat2iou /= cat_counter
        cat2acc /= cat_counter
        mIoU = torch.mean(cat2iou)
        inst_mIoU = torch.cat(all_shapes_iou, dim=0).mean()
        cat2iou = dict(zip(self.test_loader.dataset.cat_names, cat2iou.tolist()))
        cat2acc = dict(zip(self.test_loader.dataset.cat_names, cat2acc.tolist()))
        return cat2iou, cat2acc, mIoU.item(), inst_mIoU.item()

    def eval_sgd(self):
        self.model.eval()

        def predict_fn(features):
            with torch.no_grad():
                logits = self.model(send_to_device(features, self.device))
                return torch.argmax(logits, dim=1).cpu()

        return self.eval(predict_fn)

    def eval_sk(self):
        self.model.eval()

        def predict_fn(features):
            with torch.no_grad():
                embeddings = self.model.feature_extractor.forward_features(send_to_device(features, self.device))
                embeddings = embeddings.transpose(2, 1).reshape(-1, embeddings.size(1)).numpy()

            return self.model.sklearn_model.predict(embeddings)

        return self.eval(predict_fn)