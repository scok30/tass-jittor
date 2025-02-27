import jittor as jt
from jittor import nn
import torchvision.transforms as transforms
import os
import numpy as np
import pdb
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from myNetwork_jit import joint_network_dual
from iCIFAR100 import iCIFAR100
import logging
import copy

import jittor as jt

def torch_to_jittor(tensor: torch.Tensor) -> jt.Var:
    """
    将 PyTorch Tensor 转换为 Jittor Tensor，并对齐 device 和 dtype。
    
    Args:
        tensor (torch.Tensor): 输入的 PyTorch 张量。

    Returns:
        jt.Var: 转换后的 Jittor 张量。
    """
    # 映射 dtype
    dtype_map = {
        torch.float32: jt.float32,
        torch.float64: jt.float64,
        torch.float16: jt.float16,
        torch.int32: jt.int32,
        torch.int64: jt.int64,
        torch.uint8: jt.uint8,
        torch.bool: jt.bool
    }
    
    jt_dtype = dtype_map.get(tensor.dtype, jt.float32)  # 默认转换为 float32
    
    # 判断是否使用 GPU
    use_cuda = tensor.is_cuda and jt.has_cuda
    if use_cuda:
        jt.flags.use_cuda = 1  # 启用 Jittor CUDA

    # 转换为 Jittor 张量
    jt_tensor = jt.array(tensor.cpu().numpy(), dtype=jt_dtype)
    
    return jt_tensor

class jointSSL:
    def __init__(self, args, encoder, numsuperclass, task_size, device):
        self.args = args
        self.size = 32
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = joint_network_dual(args.fg_nc, numsuperclass, encoder)
        self.radius = 0
        self.prototype = None
        self.numsamples = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        
        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.train_dataset = iCIFAR100(args.root, transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100(args.root, transform=self.test_transform, test_transform=self.test_transform, train=False, download=True)

        self.train_loader = None
        self.test_loader = None

    def beforeTrain(self, current_task):
        self.model.eval()
        if current_task == 0:
            classes = [0, self.numclass]
        else:
            classes = [self.numclass-self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            self.model.Incremental_learning(self.numclass)
        self.model.train()

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size,
                                  pin_memory=True)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=False,
                                 batch_size=self.args.batch_size,
                                 pin_memory=True)
        
        
        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=False,
                                 batch_size=self.args.batch_size,
                                 pin_memory=True)
        return test_loader


    def train(self, current_task, old_class=0):
        opt = jt.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
        scheduler = jt.lr_scheduler.CosineAnnealingLR(opt, T_max=32)
        accuracy = 0
        # Print and log accuracy periodically
        for epoch in range(self.epochs):
            running_loss = 0.0
            for step, (indexs, images, labels, coarse_labels) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
                # Move data to the correct device
                for k in images:
                    images[k] = images[k].to(self.device)
                images, edge, sal = images['img'], images['edge'], images['sal']
                origin_img = images

                labels, coarse_labels = labels.to(self.device), coarse_labels.to(self.device)

                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, self.size, self.size)
                joint_labels = torch.stack([labels * 4 + k for k in range(4)], 1).view(-1)
                
                images, joint_labels, labels, coarse_labels, edge, sal, origin_img = map(torch_to_jittor, [images, joint_labels, labels, coarse_labels, edge, sal, origin_img])
                
                loss = self._compute_loss(images, joint_labels, labels, coarse_labels, old_class, edge=edge, sal=sal, oi=origin_img, args=self.args)
                
                running_loss += loss.item()
                opt.step(loss)

            scheduler.step()
            
            # Print and log accuracy periodically
            if epoch % self.args.print_freq == 0 or epoch == self.epochs - 1:
                accuracy = self._test(self.test_loader)
                print(f'epoch:{epoch}, accuracy:{accuracy:.5f}')
                logging.info(f'task:{current_task}, epoch:{epoch}, accuracy:{accuracy:.5f}')
                
            logging.info(f'train loss:{running_loss / len(self.train_loader):.6f}')
        
        # Save model prototype after training
        self.protoSave(self.model, self.train_loader, current_task)

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for step, (indexs, imgs, labels, _) in enumerate(tqdm(testloader, desc=f'Test')):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            imgs, labels = map(torch_to_jittor, [imgs, labels])
            # Using Jittor's no_grad context
            with jt.no_grad():
                fine_outputs, _, _ = self.model(imgs, noise=False, sal=False)
            
            # Jittor equivalent of torch.max
            predicts = jt.argmax(fine_outputs, dim=1)[0]
            correct += (predicts == labels).sum().item()  # .item() to get the scalar value
            total += len(labels)
        
        accuracy = correct / total
        self.model.train()
        return accuracy

    def dilation_boundary_loss(self, origin_edgemap, intermediate_x):
        dbs_loss = 0
        for i in range(len(intermediate_x)):
            kernel_size = i * 2 + 3
            dilation_kernel = jt.ones((1, 1, kernel_size, kernel_size), dtype=jt.float32)
            
            if origin_edgemap.ndim == 3:
                origin_edgemap = origin_edgemap.unsqueeze(1)
                
            # Apply dilated convolution
            dilated_edge_maps = jt.nn.conv2d(origin_edgemap, dilation_kernel, padding=kernel_size // 2)
            dilated_edge_maps = (dilated_edge_maps > 0.5).float()
            
            # Interpolate to match the size of intermediate_x[i]
            dilated_edge_maps = jt.nn.interpolate(dilated_edge_maps, size=intermediate_x[i].shape[2:], mode="bilinear", align_corners=False)
            
            numel = dilated_edge_maps.shape[-2] * dilated_edge_maps.shape[-1]

            # Compute binary cross entropy with logits
            dbs_loss = dbs_loss + jt.nn.binary_cross_entropy_with_logits(intermediate_x[i], 1 - dilated_edge_maps) / numel
        return dbs_loss


    def _compute_loss(self, imgs, joint_labels, labels, coarse_labels, old_class=0, **kwargs):
        args = kwargs['args']
        
        fine_output, coarse_feature, fine_feature, (oi_sal, oi_edge, inx) = self.model(imgs, noise=False, sal=True)

        joint_preds = self.model.fc(fine_feature)
        single_preds = fine_output[::4]
        
        joint_labels, labels = joint_labels.astype(jt.int32), labels.astype(jt.int32)

        joint_loss = jt.nn.CrossEntropyLoss()(joint_preds / self.args.temp, joint_labels)
        signle_loss = jt.nn.CrossEntropyLoss()(single_preds / self.args.temp, labels)

        edge, sal, oi = kwargs['edge'][:, 0], kwargs['sal'][:, 0], kwargs['oi']
        numel = sal.shape[-2] * sal.shape[-1]
        dbs_loss = self.dilation_boundary_loss(edge, inx)
        lms_loss = (jt.nn.MSELoss()(jt.sigmoid(oi_sal), sal) + jt.nn.MSELoss()(jt.sigmoid(oi_edge), edge)) / numel

        agg_preds = 0
        for i in range(4):
            agg_preds = agg_preds + joint_preds[i::4, i::4] / 4

        distillation_loss = jt.nn.KLDivLoss(reduction='batchmean')(
            jt.nn.log_softmax(single_preds, 1),
            jt.nn.softmax(agg_preds.detach(), 1)
        )

        if old_class == 0:
            return joint_loss + signle_loss + distillation_loss + dbs_loss + lms_loss
        else:
            self.old_model.eval()
            with jt.no_grad():
                _, coarse_feature_old, feature_old = self.old_model(imgs, noise=False, sal=False)
            loss_kd = (fine_feature - feature_old.detach()).pow(2).sum().sqrt()

            proto_aug = []
            proto_aug_label = []
            old_class_list = list(self.prototype.keys())
            for _ in range(fine_feature.shape[0] // 4):  # batch_size = feature.shape[0] // 4
                i = np.random.randint(0, fine_feature.shape[0])
                np.random.shuffle(old_class_list)
                lam = np.random.beta(0.5, 0.5)
                if lam > 0.6: 
                    lam = lam * 0.6
                if np.random.random() >= 0.5:
                    temp = (1 + lam) * self.prototype[old_class_list[0]] - lam * fine_feature.detach().cpu().numpy()[i]
                else:
                    temp = (1 - lam) * self.prototype[old_class_list[0]] + lam * fine_feature.detach().cpu().numpy()[i]

                proto_aug.append(temp)
                proto_aug_label.append(old_class_list[0])
            proto_aug = jt.array(np.float32(np.asarray(proto_aug))).float()
            proto_aug_label = jt.array(np.asarray(proto_aug_label)).astype(jt.int32)

            aug_preds = self.model.classifier(proto_aug)
            joint_aug_preds = self.model.fc(proto_aug)

            agg_preds = 0
            agg_preds = agg_preds + joint_aug_preds[:, ::4]

            aug_distillation_loss = jt.nn.KLDivLoss(reduction='batchmean')(
                jt.nn.log_softmax(aug_preds, 1),
                jt.nn.softmax(agg_preds.detach(), 1)
            )

            loss_protoAug = jt.nn.CrossEntropyLoss()(aug_preds / self.args.temp, proto_aug_label) + jt.nn.CrossEntropyLoss()(joint_aug_preds / self.args.temp, proto_aug_label * 4) + aug_distillation_loss

            return joint_loss + signle_loss + dbs_loss + lms_loss + distillation_loss + \
                self.args.protoAug_weight * loss_protoAug + self.args.kd_weight * loss_kd

    def afterTrain(self, log_root):
        path = log_root + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        
        # Saving the model using Jittor
        jt.save(self.model.state_dict(), filename)
        # Loading the model in evaluation mode
        # self.old_model = self.model.__class__()  # Reinitialize the model
        self.old_model = copy.deepcopy(self.model)
        self.old_model.load_state_dict(jt.load(filename))
        self.old_model.eval()

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with jt.no_grad():
            for i, (indexs, images, target, _) in enumerate(loader):
                if isinstance(images, dict):
                    images = images['img']
                images, target = map(torch_to_jittor, [images, target])
                # Forward pass through the model
                _, _, feature = model(images.to(self.device), noise=False, sal=False)
                
                # Check if batch size is correct
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())  # Convert labels to numpy
                    features.append(feature.cpu().numpy())  # Convert features to numpy
        
        labels_set = np.unique(labels)  # Get unique labels
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])  # Flatten the labels
        
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))  # Flatten the features
        
        prototype = {}
        class_label = []
        numsamples = {}
        
        # Compute prototype for each class
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype[item] = np.mean(feature_classwise, axis=0)
            numsamples[item] = feature_classwise.shape[0]

        if current_task == 0:
            self.prototype = prototype
            self.class_label = class_label
            self.numsamples = numsamples
        else:
            self.prototype.update(prototype)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
            self.numsamples.update(numsamples)
