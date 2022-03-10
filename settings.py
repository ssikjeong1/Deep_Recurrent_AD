import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Define the function to print and write log
def writelog(file, line):
    file.write(line + '\n')
    print(line)

# Define Weighted Sample Loader
def weighted_sample_loader(data, data_mask, label, batch_size):
    # Calculate weights
    class_sample_count = np.array([len(np.where(label == t)[0]) for t in np.unique(label)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in label])

    # Define sampler using WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    # Concatenate data and its mask
    d = np.expand_dims(data, axis=0)
    dm = np.expand_dims(data_mask, axis=0)
    alldata = np.vstack([d, dm]).transpose(1,0,2,3)

    # Convert to tensor
    data = torch.tensor(alldata).float()
    label = torch.tensor(label).float()

    # Define the loader
    dataset = torch.utils.data.TensorDataset(data, label)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=1,
                        sampler=sampler,
                        pin_memory=True)
    return loader

def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(11):
        if h == 0:
            # deltas.append(np.zeros(6)*12)
            deltas.append(np.zeros(9) * 12)
        else:
            # deltas.append((np.ones(6)*12) + ((1 - masks[h]) * deltas[-1]))
            deltas.append((np.ones(9) * 12) + ((1 - masks[h]) * deltas[-1]))

    return np.array(deltas)


def parse_rec(ori_values, ori_masks, values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values

    rec = {}

    rec['ori_values'] = np.nan_to_num(ori_values).tolist()
    rec['ori_masks'] = ori_masks.astype('int32').tolist()

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec

def collate_fn(recs):
    forward = [r['forward'] for r in recs]
    backward = [r['backward'] for r in recs]

    def to_tensor_dict(recs):
        ori_values = torch.FloatTensor(np.array([r['ori_values'] for r in recs]))
        ori_masks = torch.FloatTensor(np.array([r['ori_masks'] for r in recs]))

        values = torch.FloatTensor(np.array([r['values'] for r in recs]))
        masks = torch.FloatTensor(np.array([r['masks'] for r in recs]))
        deltas = torch.FloatTensor(np.array([r['deltas'] for r in recs]))

        evals = torch.FloatTensor(np.array([r['evals'] for r in recs]))
        eval_masks = torch.FloatTensor(np.array([r['eval_masks'] for r in recs]))
        forwards = torch.FloatTensor(np.array([r['forwards'] for r in recs]))

        return {'ori_values': ori_values,
                'ori_masks': ori_masks,
                'values': values,
                'masks': masks,
                'deltas': deltas,
                'evals': evals,
                'eval_masks': eval_masks,
                'forwards': forwards}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(np.array([r['label'] for r in recs]))
    ret_dict['is_train'] = torch.FloatTensor(np.array([r['is_train'] for r in recs]))

    return ret_dict

# Define Sample Loader
def sample_loader(data, mask, label, batch_size, is_train=False):

    mri_data = data[:,:,:6]
    cog_data = data[:,:,6:]

    mri_mask = mask[:,:,:6]
    cog_mask = mask[:,:,6:]

    # Get Dimensionality
    [N, T, D] = mri_data.shape
    [N2, T2, D2] = cog_data.shape

    # Reshape
    mri_data = mri_data.reshape(N, T*D)
    mri_mask = mri_mask.reshape(N, T*D)

    cog_data = cog_data.reshape(N2, T2 * D2)
    cog_mask = cog_mask.reshape(N2, T2 * D2)

    recs = []
    for i in range(N):

        ori_masks = mri_mask[i].reshape(T, D)
        tmp_ori_masks = ori_masks.flatten()

        ori_values = mri_data[i].reshape(T, D)
        ori_values[np.where(ori_masks == 0)] = np.nan
        tmp_ori_values = ori_values.flatten()

        # ori_masks = np.ones_like(tmp_ori_values)
        # ori_masks[np.where(tmp_ori_values == 0)] = 0
        # ori_masks = ori_masks.reshape(t_n, t_td)

        # ori_masks = np.where(ori_values[:,3:9]).reshape(T, D)

        # randomly eliminate 10% values as the imputation ground-truth
        indices = np.where(tmp_ori_masks!=0)[0]
        indices = np.random.choice(indices, len(indices) // 10)

        # indices = np.where(~np.isnan(data[i]))[0].tolist()
        # indices = np.random.choice(indices, len(indices) // 10)

        values = tmp_ori_values.copy()
        # values = data[i].copy()

        ## Check this part
        values[indices] = np.nan

        # masks = np.ones_like(tmp_ori_values)
        # masks[np.where(tmp_ori_values == 0)] = 0


        masks = ~np.isnan(values)
        eval_masks = (~np.isnan(values)) ^ (~np.isnan(tmp_ori_values))

        evals = tmp_ori_values.reshape(T, D)
        values = values.reshape(T, D)

        masks = masks.reshape(T, D)
        eval_masks = eval_masks.reshape(T, D)

        rec = {'label': label[i]}

        ## Check this part as well
        ori_values_cog = cog_data[i].reshape(T2, D2)
        ori_masks_cog = cog_mask[i].reshape(T2, D2)

        ori_values_ = np.concatenate((ori_values, ori_values_cog), axis=1)
        ori_masks_ = np.concatenate((ori_masks, ori_masks_cog), axis=1)

        values_ = np.concatenate((values, ori_values_cog), axis=1)
        masks_ = np.concatenate((masks, ori_masks_cog), axis=1)

        evals_ = np.concatenate((evals, ori_values_cog), axis=1)
        eval_masks_ = np.concatenate((eval_masks, ori_values_cog), axis=1)

        # prepare the model for both directions
        rec['forward'] = parse_rec(ori_values_, ori_masks_, values_, masks_, evals_, eval_masks_, dir_='forward')
        rec['backward'] = parse_rec(ori_values_, ori_masks_, values_[::-1], masks_[::-1], evals_[::-1], eval_masks_[::-1], dir_='backward')

        if is_train:
            rec['is_train'] = 1
        else:
            rec['is_train'] = 0

        recs.append(rec)

    loader = DataLoader(recs,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=True,
                        pin_memory=True,
                        collate_fn=collate_fn)

    return loader

# def sample_loader(data, mask, label, batch_size, is_train=False):
#
#     # Get Dimensionality
#     [N, T, D] = data.shape
#
#     # Reshape
#     data = data.reshape(N, T*D)
#
#     mask = mask.reshape(N, T*D)
#
#     recs = []
#     for i in range(N):
#
#         ori_masks = mask[i].reshape(T, D)
#         tmp_ori_masks = ori_masks.flatten()
#
#         ori_values = data[i].reshape(T, D)
#         ori_values[np.where(ori_masks == 0)] = np.nan
#         tmp_ori_values = ori_values.flatten()
#
#         # ori_masks = np.ones_like(tmp_ori_values)
#         # ori_masks[np.where(tmp_ori_values == 0)] = 0
#         # ori_masks = ori_masks.reshape(t_n, t_td)
#
#         # ori_masks = np.where(ori_values[:,3:9]).reshape(T, D)
#
#         # randomly eliminate 10% values as the imputation ground-truth
#         indices = np.where(tmp_ori_masks!=0)[0]
#         indices = np.random.choice(indices, len(indices) // 10)
#
#         # indices = np.where(~np.isnan(data[i]))[0].tolist()
#         # indices = np.random.choice(indices, len(indices) // 10)
#
#         values = tmp_ori_values.copy()
#         # values = data[i].copy()
#
#         ## Check this part
#         values[indices] = np.nan
#
#         # masks = np.ones_like(tmp_ori_values)
#         # masks[np.where(tmp_ori_values == 0)] = 0
#
#
#         masks = ~np.isnan(values)
#         eval_masks = (~np.isnan(values)) ^ (~np.isnan(tmp_ori_values))
#
#         evals = tmp_ori_values.reshape(T, D)
#         values = values.reshape(T, D)
#
#         masks = masks.reshape(T, D)
#         eval_masks = eval_masks.reshape(T, D)
#
#         rec = {'label': label[i]}
#
#         # prepare the model for both directions
#         rec['forward'] = parse_rec(ori_values, ori_masks, values, masks, evals, eval_masks, dir_='forward')
#         rec['backward'] = parse_rec(ori_values, ori_masks, values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')
#
#         if is_train:
#             rec['is_train'] = 1
#         else:
#             rec['is_train'] = 0
#
#         recs.append(rec)
#     # Concatenate data and its mask
#     # d = np.expand_dims(data, axis=0)
#     # dm = np.expand_dims(data_mask, axis=0)
#     # alldata = np.vstack([d, dm]).transpose(1,0,2,3)
#     #
#     # # Convert to tensor
#     # data = torch.tensor(alldata).float()
#     # label = torch.tensor(label).float()
#
#     # Define the loader
#     loader = DataLoader(recs,
#                         batch_size=batch_size,
#                         num_workers=0,
#                         shuffle=True,
#                         pin_memory=True,
#                         collate_fn=collate_fn)
#
#     return loader

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def calculate_performance(y, y_score, y_pred, classes=None):
    # Calculate Evaluation Metrics
    acc = accuracy_score(y_pred, y) * 100
    if classes == None:
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    if classes == True:
        a, b, c, \
        d, e, f, \
        g, h, i = confusion_matrix(y, y_pred, labels=[0, 1, 2]).ravel()
        tn, fp, fn, tp = (a + c + g + i), (b + h), (d + f), (e)
    # total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        if classes == None:
            p, r, t = precision_recall_curve(y, y_score)
            auprc = np.nan_to_num(metrics.auc(r, p))
        else:
            p, r, t = 0, 0, 0
            auprc = 0

    spec = np.nan_to_num(tn / (tn + fp))
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    try:
        if classes == None:
            auc = roc_auc_score(y, y_score)
        else:
            auc = 0
    except ValueError:
        auc = 0

    return auc, auprc, acc, balacc, sen, spec, prec, recall


def calculate_performance_ver2(y_label, y_pred, classes=None):
    if classes == None:
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred, labels=[0, 1]).ravel()

        return metrics.precision_score(y_label, y_pred), metrics.recall_score(y_label, y_pred), (tn / (tn + fp)), (tn / (tn + fp)), ((tn / (tn + fp)) + (tn / (tn + fp))) / 2, None
    elif classes == True:
        a, b, c, \
        d, e, f, \
        g, h, i = confusion_matrix(y_label, y_pred, labels=[0, 1, 2]).ravel()

        precision = np.zeros(3)  # PPV: TP/(TP+FP)
        recall = np.zeros(3)  # TPR (sensitivity): TP/(TP+FN)
        specificity = np.zeros(3)  # TNR: TN/(TN+FP)

        tp_a = a
        fp_a = d + g
        fn_a = b + c
        tn_a = e + f + h + i
        precision[0] = tp_a / (tp_a + fp_a)
        recall[0] = tp_a / (tp_a + fn_a)
        specificity[0] = tn_a / (tn_a + fp_a)

        tp_b = e
        fp_b = d + f
        fn_b = b + h
        tn_b = a + c + g + i
        precision[1] = tp_b / (tp_b + fp_b)
        recall[1] = tp_b / (tp_b + fn_b)
        specificity[1] = tn_b / (tn_b + fp_b)

        tp_c = i
        fp_c = g + h
        fn_c = c + f
        tn_c = a + b + d + e
        precision[2] = tp_c / (tp_c + fp_c)
        recall[2] = tp_c / (tp_c + fn_c)
        specificity[2] = tn_c / (tn_c + fp_c)

        return np.mean(precision), np.mean(recall), np.mean(specificity), np.mean(recall), (
                np.mean(recall) + np.mean(specificity)) / 2, 2 * np.mean(precision) * np.mean(recall) / (
                       np.mean(precision) + np.mean(recall))
    else:
        print('Error!')

from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_analysis(f, imputations, evals, eval_masks, labels, c, m):
    mae = []
    mse = []
    x_hat = np.array(imputations).reshape(-1,9,6)
    x_true = np.array(evals)[:,1:,:]
    masks = np.array(eval_masks)[:,1:,:]
    t_labels = labels.reshape(-1, 10, 1)[:,1:]

    xhat = x_hat[np.where(t_labels[:, -1].squeeze() == 2)[0].squeeze()][0] ## Estimated value
    xtrue = x_true[np.where(t_labels[:, -1].squeeze() == 2)[0].squeeze()][0] ## real value
    xmasks = masks[np.where(t_labels[:, -1].squeeze() == 2)[0].squeeze()][0] ## mask value
    xmasks[np.where(xmasks == 2)] = 1
    i = 0
    y_hat = (xhat[np.where(xmasks[:, 0] == 1)[0]][:, i] - c[i]) / m[i]
    y_true = (xtrue[np.where(xmasks[:, 0] == 1)[0]][:, i] - c[i]) / m[i]


    for i in range(x_hat.shape[1]):
        y_hat = (x_hat[np.where(masks[:,i]==1)[0], i] - c[i]) / m[i]
        y_true = (x_true[np.where(masks[:,i]==1)[0], i] - c[i]) / m[i]


        mae_ = mean_absolute_error(y_true, y_hat)
        mse_ = mean_squared_error(y_true, y_hat)
        writelog(f,'MAE feature{} : {}'.format(i+1, mae_))
        # print('MSE feature{} {}'.format(i+1, mse_))

        mae.append(mae_)
        mse.append(mse_)
    return mae, mse

def regression(f, imputations, evals, eval_masks, c, m):
    mae = []
    mse = []
    # if flag== True:
    #     x_hat = np.array(imputations)
    # else:
    #     x_hat = np.array(imputations).reshape(-1, 6)
    # x_hat = np.array(imputations).reshape(-1, 6)
    x_hat = np.array(imputations)
    x_true = np.array(evals).reshape(-1,6)
    masks = np.array(eval_masks).reshape(-1,6)

    for i in range(x_hat.shape[1]):
        y_hat = (x_hat[np.where(masks[:, i] == 1)[0], i] - c[i]) / m[i]
        y_true = (x_true[np.where(masks[:, i] == 1)[0], i] - c[i]) / m[i]
        mae_ = mean_absolute_error(y_true, y_hat)
        mse_ = mean_squared_error(y_true, y_hat)
        writelog(f, 'MAE feature{} : {}'.format(i + 1, mae_))

        mae.append(mae_)
        mse.append(mse_)
    return mae, mse

def regression_cog(f, imputations, evals, eval_masks, original_cog=None):
    x_hat = np.array(imputations).reshape(-1, 1)
    x_true = np.array(evals).reshape(-1, 1)
    masks = np.array(eval_masks).reshape(-1, 1)
    # original_cog = [30,70,85]
    y_hat = (x_hat[np.where(masks == 1)])
    y_true = (x_true[np.where(masks == 1)])
    mae = (mean_absolute_error(y_true, y_hat)) * original_cog[0]
    mse = np.sqrt(mean_squared_error(y_true, y_hat)) * original_cog[0]
    writelog(f, 'RMSE feature : {}'.format(mae))
    # mae_ = mean_absolute_error(y_true, y_hat)
    # mse_ = mean_squared_error(y_true, y_hat)
    # writelog(f,'MAE feature{} : {}'.format(i+1, mae_))
    # print('MSE feature{} {}'.format(i+1, mse_))


    return mae, mse

# def plot_imputation(dir, x, x_recon_vae, x_imp_vae, x_imp_rnn, k, phase, epoch):
def plot_imputation(dir,
                    ori_x, ori_m,
                    x, m,
                    eval_x, eval_m,
                    imp_x,
                    k, phase, epoch):
    # np.repeat(ori_m.T, 12, axis=1)
    cmap = 'RdBu_r'
    fig, axes = plt.subplots(3, 2, sharey=True)

    cx1 = axes[0, 0].imshow(ori_x.T, vmin=-1, vmax=1, cmap=cmap)#hsv)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_ylabel('Variables')
    axes[0, 0].title.set_text('ori x')

    cx2 = axes[0, 1].imshow(ori_m.T, vmin=0, vmax=1)
    axes[0, 1].set_xticks([])
    axes[0, 1].title.set_text('ori m')

    cx3 = axes[1, 0].imshow(x.T, vmin=-1, vmax=1, cmap=cmap)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_ylabel('Variables')
    axes[1, 0].title.set_text('removed x')

    cx4 = axes[1, 1].imshow(eval_m.T, vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Hours')
    axes[1, 1].title.set_text('removed m')

    cx5 = axes[2, 0].imshow(imp_x.T, vmin=-1, vmax=1, cmap=cmap)
    axes[2, 0].set_ylabel('Variables')
    axes[2, 0].set_xlabel('Hours')
    axes[2, 0].title.set_text('imputed x')

    axes[2, 1].set_visible(False)

    fig.colorbar(cx2, ax=axes[0,:].ravel().tolist(), orientation='vertical')
    fig.colorbar(cx1, ax=axes[2,:].ravel().tolist(), orientation='vertical')

    strk = str(k)
    stre = str(epoch)
    # fig, axs = plt.subplots(3, 2, sharey=True)
    # cm = ['RdBu_r', 'viridis']
    # data = [ori_x.T, x.T, imp_x.T, ori_m.T, eval_m.T, np.zeros_like(imp_x.T)]
    # title = ['Ori x', 'Ori m', 'Removed x', 'Removed m', 'Imputed x']
    # for col in range(2):
    #     for row in range(3):
    #         if col == 1 and row == 2:
    #             continue
    #         ax = axs[row, col]
    #         pcm = ax.pcolormesh(data[(col * 3) + row], cmap=cm[col])
    #         ax.title.set_text(title[(col * 3) + row])
    #         if col == 0:
    #             ax.set_ylabel('Variables')
    #
    #         if row == 2:
    #             ax.set_xlabel('Hours (x12)')
    #     plt.colorbar(pcm, ax=axs[:, col], shrink=0.6, pad=0.01)
    # plt.subplots_adjust(top=0.95, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
    #                     wspace=0.5)
    # plt.savefig(dir + '/img/' + phase + '/x_' + strk + '_' + stre.rjust(4, '0') + '.png')
    # plt.show()


    plt.subplots_adjust(left=0.25, right=0.75, top=0.9, bottom=0.1)
    plt.savefig(dir + '/img/' + phase + '/x_' + strk + '_' + stre.rjust(4, '0') + '.png',
                bbox_inches='tight')
    # plt.show()
    plt.close('all')
    plt.clf()
    print('wait')
