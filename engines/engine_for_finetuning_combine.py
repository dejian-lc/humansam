import os
import time
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from datasets.mixup import Mixup
from timm.utils import accuracy
import utils
from scipy.special import softmax
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt


def train_class_batch(model, samples, depths, target, criterion, confs=None, use_conf=False):
    if depths is not None and depths.numel() > 0:
        outputs = model(samples, depths)
    else:
        outputs = model(samples)
    loss = criterion(outputs, target)
    
    if use_conf and confs is not None and confs.numel() > 0:
        weights = confs

        # 将损失乘以对应的权重
        weighted_losses = loss * weights

        # 计算加权损失的平均值
        loss = torch.mean(weighted_losses)
    else:
        loss = loss.mean()
    return loss, outputs


# def train_class_batch(model, samples, target, criterion):
#     outputs = model(samples)
#     loss = criterion(outputs, target)
#     return loss, outputs


def train_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None, log_writer=None,
        start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
        num_training_steps_per_epoch=None, update_freq=None,
        bf16=False, accelerator=None, use_conf=False
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    optimizer.zero_grad()

    for data_iter_step, (samples, targets, confs, depths) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        confs = confs.to(device, non_blocking=True)
        confs = torch.log(confs + torch.exp(torch.ones(1, device=device)))
        
        depths = torch.cat(depths, dim=0)
        depths = depths.to(device, non_blocking=True)

        with accelerator.autocast():
            loss, output = train_class_batch(
                model, samples, depths, targets, criterion, confs, use_conf=use_conf)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        
        accelerator.backward(loss)
        grad_norm = None
        if (data_iter_step + 1) % update_freq == 0:
            if max_norm is not None:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm)
            else:
                # 如果没有设置 max_norm，我们仍然计算梯度范数用于记录
                # accelerator.clip_grad_norm_ 会返回范数，即使 max_norm 很大
                # 但为了避免不必要的裁剪操作，我们可以手动计算
                # 或者简单地调用 clip_grad_norm_ 并传入一个非常大的值
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1e9)
            
            optimizer.step()
            optimizer.zero_grad()
            
        loss_scale_value = 0

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, bf16=False, accelerator=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        depth = batch[2]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        depths = depth.to(device, non_blocking=True)

        # compute output
        with accelerator.autocast():
            if depths.numel() > 0:
                output = model(videos, depths)
            else:
                output = model(videos)
            loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1,))[0]

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, bf16=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        depth = batch[5]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        depths = depth.to(device, non_blocking=True)

        # compute output
        dtype = torch.bfloat16 if bf16 else torch.float16
        with torch.autocast(device_type=device.type, dtype=dtype):
            if depths.numel() > 0:
                output = model(videos, depths)
            else:
                output = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                               str(output.data[i].float().cpu().numpy().tolist()), \
                                               str(int(target[i].cpu().numpy())), \
                                               str(int(chunk_nb[i].cpu().numpy())), \
                                               str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1 = accuracy(output, target, topk=(1,))[0]

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}\n".format(acc1))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, nb_classes):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split(' ')[0]
            label = line.split(']')[-1].split(' ')[1]
            chunk_nb = line.split(']')[-1].split(' ')[2]
            split_nb = line.split(']')[-1].split(' ')[3]
            data = np.fromstring(' '.join(line.split(' ')[1:]).split('[')[1].split(']')[0], dtype=np.float32, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    # top5 = [x[2] for x in ans] # Removed Top-5
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    pred_probs = [x[4] for x in ans]  # 获取每个样本的预测概率
    final_top1 = np.mean(top1)
    
    # Initialize variables
    precision = 0
    recall = 0
    macro_f1 = 0
    auc = 0
    binary_f1 = 0
    new_acc = 0
    new_auc = 0
    
    if nb_classes == 4:
        class_names = ["Appearance Anomaly", "Spatial Anomaly", "Motion Anomaly", "Real"]
        
        # Multi-class metrics
        precision = precision_score(label, pred, average='macro')
        recall = recall_score(label, pred, average='macro')
        f1_per_class = f1_score(label, pred, average=None)
        macro_f1 = f1_score(label, pred, average='macro')
        
        print("Per-class F1 scores:")
        for i, class_name in enumerate(class_names):
            if i < len(f1_per_class):
                print(f"{class_name}: {f1_per_class[i]:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")

        if len(label) > 0 and len(pred_probs) > 0:
            try:
                auc = roc_auc_score(label, pred_probs, multi_class='ovr')
            except ValueError as e:
                print(f"Warning: Could not calculate multi-class AUC: {e}")
                auc = float('nan')
        else:
            auc = float('nan')

        # Confusion Matrix (4-class)
        conf_matrix = confusion_matrix(label, pred)
        plt.figure(figsize=(8, 6))
        conf_pct = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
        annot_text = np.array([[f'{val}\n({pct:.1f}%)' for val, pct in zip(row_val, row_pct)] 
                               for row_val, row_pct in zip(conf_matrix, conf_pct)])
        ax = sns.heatmap(conf_matrix, annot=annot_text, fmt='', cmap='Blues', 
                     annot_kws={"size": 15, "weight": "bold"}, 
                     cbar_kws={"shrink": 0.8})
        tick_labels = ["Appearance", "Spatial", "Motion", "Real"]
        ax.set_xticklabels(tick_labels, fontsize=10, fontweight='bold')
        ax.set_yticklabels(tick_labels, fontsize=10, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Mapping to Binary (0,1,2 -> 0 (Forgery), 3 -> 1 (Real))
        new_label = [0 if lbl in [0, 1, 2] else 1 for lbl in label]
        new_pred = [0 if prd in [0, 1, 2] else 1 for prd in pred]
        
        binary_f1 = f1_score(new_label, new_pred)
        print(f"Binary F1 Score (Anomaly vs Normal): {binary_f1:.4f}")
        new_acc = np.mean(np.array(new_label) == np.array(new_pred))
        
        if len(label) > 0 and len(pred_probs) > 0:
            pred_probs_np = np.array(pred_probs)
            # Probability of Real class (index 3)
            new_pred_probs = pred_probs_np[:, 3]
            try:
                new_auc = roc_auc_score(new_label, new_pred_probs)
            except ValueError as e:
                print(f"Warning: Could not calculate binary AUC from 4-class: {e}")
                new_auc = float('nan')
        else:
            new_auc = float('nan')

        # Binary Confusion Matrix
        binary_conf_matrix = confusion_matrix(new_label, new_pred)
        plt.figure(figsize=(8, 6))
        binary_conf_pct = binary_conf_matrix.astype('float') / (binary_conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
        binary_annot_text = np.array([[f'{val}\n({pct:.1f}%)' for val, pct in zip(row_val, row_pct)] 
                                     for row_val, row_pct in zip(binary_conf_matrix, binary_conf_pct)])
        ax = sns.heatmap(binary_conf_matrix, annot=binary_annot_text, fmt='', cmap='Blues',
                        annot_kws={"size": 15, "weight": "bold"},
                        cbar_kws={"shrink": 0.8})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks([0.5, 1.5], ['Forgery', 'Real'], fontsize=11, fontweight='bold')
        plt.yticks([0.5, 1.5], ['Forgery', 'Real'], fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(eval_path, 'binary_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    elif nb_classes == 2:
        class_names = ["Forgery", "Real"]
        
        # Binary metrics directly
        precision = precision_score(label, pred, average='binary')
        recall = recall_score(label, pred, average='binary')
        macro_f1 = f1_score(label, pred, average='binary') # Using macro_f1 var for binary f1
        binary_f1 = macro_f1 # Same
        
        print(f"Binary F1 Score: {macro_f1:.4f}")
        
        if len(label) > 0 and len(pred_probs) > 0:
            pred_probs_np = np.array(pred_probs)
            # Probability of positive class (index 1)
            try:
                auc = roc_auc_score(label, pred_probs_np[:, 1])
            except ValueError as e:
                print(f"Warning: Could not calculate binary AUC: {e}")
                auc = float('nan')
        else:
            auc = float('nan')
            
        # For consistency with return values, set "double" metrics to same values
        new_acc = final_top1
        new_auc = auc

        # Confusion Matrix
        conf_matrix = confusion_matrix(label, pred)
        plt.figure(figsize=(8, 6))
        conf_pct = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
        annot_text = np.array([[f'{val}\n({pct:.1f}%)' for val, pct in zip(row_val, row_pct)] 
                               for row_val, row_pct in zip(conf_matrix, conf_pct)])
        ax = sns.heatmap(conf_matrix, annot=annot_text, fmt='', cmap='Blues', 
                     annot_kws={"size": 15, "weight": "bold"}, 
                     cbar_kws={"shrink": 0.8})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks([0.5, 1.5], ['Forgery', 'Real'], fontsize=11, fontweight='bold')
        plt.yticks([0.5, 1.5], ['Forgery', 'Real'], fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(eval_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Save results
    with open(os.path.join(eval_path, 'merged_results.txt'), 'w') as f:
        f.write(f'name\tlabel\tpred\tlabel_name\tpred_name\n')
        for name, lbl, prd in zip(dict_feats.keys(), label, pred):
            lbl_name = class_names[lbl] if lbl < len(class_names) else "Unknown"
            prd_name = class_names[prd] if prd < len(class_names) else "Unknown"
            f.write(f'{name}\t{lbl}\t{prd}\t{lbl_name}\t{prd_name}\n')

    with open(os.path.join(eval_path, 'metrics.txt'), 'w') as f:
        f.write(f'Metric\tValue\n')
        f.write(f'Accuracy\t{final_top1 * 100:.2f}\n')
        # f.write(f'Top-5 Accuracy\t{final_top5:.2f}\n')
        f.write(f'Precision\t{precision * 100:.2f}\n')
        f.write(f'Recall\t{recall * 100:.2f}\n')
        f.write(f'F1\t{macro_f1 * 100:.2f}\n')
        f.write(f'AUC\t{auc * 100:.2f}\n')
        if nb_classes == 4:
            f.write(f'Binary Accuracy\t{new_acc * 100:.2f}\n')
            f.write(f'Binary AUC\t{new_auc * 100:.2f}\n')
            f.write(f'Binary F1\t{binary_f1 * 100:.2f}\n\n')
            f.write(f'Per-class F1 scores:\n')
            # Re-calculate f1_per_class for writing if needed or use existing
            f1_per_class = f1_score(label, pred, average=None)
            for i, class_name in enumerate(class_names):
                if i < len(f1_per_class):
                    f.write(f'{class_name}\t{f1_per_class[i] * 100:.2f}\n')

    return final_top1 * 100, precision * 100, recall * 100, auc * 100, new_acc * 100, new_auc * 100, len(dict_feats)


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label), feat]


def merge2(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split(' ')[0]
            label = line.split(']')[-1].split(' ')[1]
            chunk_nb = line.split(']')[-1].split(' ')[2]
            split_nb = line.split(']')[-1].split(' ')[3]
            data = np.fromstring(' '.join(line.split(' ')[1:]).split('[')[1].split(']')[0], dtype=np.float32, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    pred_probs = [x[4] for x in ans]  # 获取每个样本的预测概率
    final_top1, final_top5 = np.mean(top1), np.mean(top5)    # 计算精确率、召回率、F1分数和AUC值
    precision = precision_score(label, pred, average='binary')
    recall = recall_score(label, pred, average='binary')
    f1 = f1_score(label, pred, average='binary')
    
    # 计算每个类别的F1分数
    f1_per_class = f1_score(label, pred, average=None)
    class_names = ["Forgery", "Real"]
    
    # 输出总体F1分数和每个类别的F1分数
    print(f"Binary F1 Score: {f1:.4f}")
    print(f"Per-class F1 scores:")
    for i, class_name in enumerate(class_names):
        if i < len(f1_per_class):
            print(f"{class_name}: {f1_per_class[i]:.4f}")
    
    # 将预测概率转换为numpy数组
    pred_probs = np.array(pred_probs)
    
    # 检查label和pred_probs是否为空
    if len(label) > 0 and len(pred_probs) > 0:
        # 针对二分类使用第1列的预测概率（索引1对应正类）
        auc = roc_auc_score(label, pred_probs[:, 1])
    else:
        auc = float('nan')  # 如果为空，设置AUC为NaN
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(label, pred)
    
    # 定义类别标签
    class_names = ["Forgery", "Real"]
    
    # 可视化并保存混淆矩阵，使用增强的可视化设置
    plt.figure(figsize=(7, 6))  # 统一尺寸
    
    # 计算百分比
    conf_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建注释文本（同时显示数量和百分比）
    annot_text = np.array([[f'{val}\n({pct:.1f}%)' for val, pct in zip(row_val, row_pct)] 
                           for row_val, row_pct in zip(conf_matrix, conf_pct)])
    
    # 使用自定义注释文本和更大的字体
    ax = sns.heatmap(conf_matrix, annot=annot_text, fmt='', cmap='Blues', 
                 annot_kws={"size": 15, "weight": "bold"}, 
                 cbar_kws={"shrink": 0.8})
    
    # 调整文本位置，使其居中
    for text in ax.texts:
        text.set_horizontalalignment('center')
    
    # 设置坐标轴标签
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    
    # 设置二分类标签
    plt.xticks([0.5, 1.5], class_names, fontsize=12, fontweight='bold')
    plt.yticks([0.5, 1.5], class_names, fontsize=12, fontweight='bold')
    
    plt.tight_layout()  # 自动调整布局以适应标签
    plt.savefig(os.path.join(eval_path, 'binary_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存样本名字、标签和预测结果到文本文件
    with open(os.path.join(eval_path, 'binary_merged_results.txt'), 'w') as f:
        f.write(f'name\tlabel\tpred\tlabel_name\tpred_name\n')
        for name, lbl, prd in zip(dict_feats.keys(), label, pred):
            lbl_name = class_names[lbl] if lbl < len(class_names) else "Unknown"
            prd_name = class_names[prd] if prd < len(class_names) else "Unknown"
            f.write(f'{name}\t{lbl}\t{prd}\t{lbl_name}\t{prd_name}\n')
      # 将指标保存到文件中
    with open(os.path.join(eval_path, 'binary_metrics.txt'), 'w') as f:
        f.write(f'Metric\tValue\n')
        f.write(f'Accuracy\t{final_top1:.2f}\n')
        f.write(f'Precision\t{precision * 100:.2f}\n')
        f.write(f'Recall\t{recall * 100:.2f}\n')
        f.write(f'F1 Score\t{f1 * 100:.2f}\n')
        f.write(f'AUC\t{auc * 100:.2f}\n\n')
        
        # 添加每个类别的F1分数
        f.write(f'Per-class F1 scores:\n')
        for i, class_name in enumerate(class_names):
            if i < len(f1_per_class):
                f.write(f'{class_name}\t{f1_per_class[i] * 100:.2f}\n')
    
    print(f"二分类评估结果已保存到 {eval_path}")
    print(f"准确率: {final_top1:.2f}%, 精确率: {precision*100:.2f}%, 召回率: {recall*100:.2f}%, F1分数: {f1*100:.2f}%, AUC: {auc*100:.2f}%")
    
    return final_top1 * 100, final_top5 * 100, precision * 100, recall * 100, f1 * 100, auc * 100