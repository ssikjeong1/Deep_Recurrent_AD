import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import os
gpu_id = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
import model
from settings import *
import random
import datetime
import argparse
import glob

from utils import *
# from torch.utils.tensorboard import SummaryWriter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch.nn.functional as F

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--dataset', type=str, default='Zero')
parser.add_argument('--data_path', type=str, default='./ADNI/TADPOLE/')
parser.add_argument('--feature', type=str, default='each')
parser.add_argument('--task', type=int, default=1)
parser.add_argument("--whichmodel", help="which model", type=str, default='model')
parser.add_argument('--hid_size', type=int, default=64)
parser.add_argument('--impute_weight', type=float, default=.1)
parser.add_argument('--reg_weight', type=float, default=0.5)
parser.add_argument('--label_weight', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=5.0)
parser.add_argument('--binary_case', type=int, default=-1)
parser.add_argument('--cognitive_score', type=bool, default=True)
args = parser.parse_args()

class FocalLoss(nn.Module):
    '''
    Multi-class Focal loss implementation
    '''
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = ((1-pt)**self.gamma) * logpt
        loss = F.nll_loss(logpt, target, self.weight, None, self.ignore_index, reduce=None, reduction='mean')
        return loss

# GPU Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tensor Seed
random.seed(1)
torch.manual_seed(1)

# kfold performance
kfold_acc = []
kfold_balacc = []
kfold_auc = []
kfold_auprc = []
kfold_sen = []
kfold_spec = []
kfold_prec = []
kfold_recall = []
kfold_feature1 = []
kfold_feature2 = []
kfold_feature3 = []
kfold_feature4 = []
kfold_feature5 = []
kfold_feature6 = []
kfold_cog1 = []
kfold_cog2 = []
kfold_cog3 = []
kfold_mae = []
kfold_mre = []

# For logging purpose, create several directories
dir = './{}_{}/{}/{}/{}/{}/{}/'.format(args.learning_rate, args.weight_decay, args.hid_size, args.label_weight, args.impute_weight, args.reg_weight,
                                                                          args.gamma) + args.whichmodel + '_LSTM' + str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S')) + '/'
if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(dir + 'img/')
    os.makedirs(dir + 'img/train/')
    os.makedirs(dir + 'img/valid/')
    os.makedirs(dir + 'img/test/')
    os.makedirs(dir + 'model/')
    os.makedirs(dir + 'tflog/')
    for k in range(args.kfold):
        os.makedirs(dir + 'model/' + str(k) + '/')
        os.makedirs(dir + 'model/' + str(k) + '/img')

# Text Logging
f = open(dir + 'log.txt', 'a')
writelog(f, '---------------')
writelog(f, 'MODEL: ' + args.whichmodel)
writelog(f, 'TRAINING PARAMETER')
writelog(f, 'Dataset: ' + str(args.dataset))
writelog(f, 'Learning Rate : ' + str(args.learning_rate))
writelog(f, 'Weight Decay : ' + str(args.weight_decay))
writelog(f, 'Batch Size : ' + str(args.batch_size))
writelog(f, 'Hidden Size : ' + str(args.hid_size))
writelog(f, 'Impute Weight: ' + str(args.impute_weight))
writelog(f, 'Reg Weight: ' + str(args.reg_weight))
writelog(f, 'Label Weight: ' + str(args.label_weight))
writelog(f, 'gamma: ' + str(args.gamma))
writelog(f, '---------------')
writelog(f, 'TRAINING LOG')

# Loop for kfold
for k in range(args.kfold-1,-1,-1):
    writelog(f, '---------------')
    writelog(f, 'FOLD ' + str(k))


    '''
    Tensorboard Logging
    '''
    # train_writer = SummaryWriter('{}tflog/kfold_{}/train_'.format(dir,k))
    # valid_writer = SummaryWriter('{}tflog/kfold_{}/valid_'.format(dir, k))
    # test_writer = SummaryWriter('{}tflog/kfold_{}/test_'.format(dir, k))

    # Tensorboard Logging
    # tfw_train = tf.compat.v1.summary.FileWriter(dir + 'tflog/kfold_' + str(k) + '/train_')
    # tfw_valid = tf.compat.v1.summary.FileWriter(dir + 'tflog/kfold_' + str(k) + '/valid_')
    # tfw_test = tf.compat.v1.summary.FileWriter(dir + 'tflog/kfold_' + str(k) + '/test_')

    '''
    Load Dataset and Data
    '''
    if args.dataset == 'Zero':
        Dataset = np.load('{}'.format(args.data_path), allow_pickle=True)

    train_data = Dataset['Train_data']
    train_label = Dataset['Train_label']

    valid_data = Dataset['Valid_data']
    valid_label = Dataset['Valid_label']

    test_data = Dataset['Test_data']
    test_label = Dataset['Test_label']

    '''
    Normalization
    '''
    writelog(f, 'Normalization')
    # normalize to ICV
    # Ventricles, Hippocampus, WholeBrain, Entorhinal, Fusiform, MidTemp
    train_feature, train_mask = normalize_feature(train_data)
    valid_feature, valid_mask = normalize_feature(valid_data)
    test_feature, test_mask = normalize_feature(test_data)

    # linearly normalize each of volumes because the RNNs output activation is tanh [-1,1]
    # I used to this normalization in two ways.
    if args.feature == 'total':
        norm_train_feature, estim_m, estim_c = scaling_feature_t(train_feature, None, None, train=True)
        norm_valid_feature, v_estim_m, v_estim_c = scaling_feature_t(valid_feature, estim_m, estim_c, train=False)
        norm_test_feature, t_estim_m, t_estim_c = scaling_feature_t(test_feature, estim_m, estim_c, train=False)
    else:
        norm_train_feature, estim_m, estim_c = scaling_feature_e(train_feature, None, None, train=True)
        norm_valid_feature, v_estim_m, v_estim_c = scaling_feature_e(valid_feature, estim_m, estim_c, train=False)
        norm_test_feature, t_estim_m, t_estim_c = scaling_feature_e(test_feature, estim_m, estim_c, train=False)

    ## Class case
    if args.binary_case == 1: #AD vs. MCI
        train_label[np.where(train_label == args.binary_case)] = -1
        valid_label[np.where(valid_label == args.binary_case)] = -1
        test_label[np.where(test_label == args.binary_case)] = -1

    if args.binary_case == 2: #AD vs. CN
        train_label[np.where(train_label == args.binary_case)] = -1
        valid_label[np.where(valid_label == args.binary_case)] = -1
        test_label[np.where(test_label == args.binary_case)] = -1

    if args.binary_case == 3: #MCI vs. CN
        train_label[np.where(train_label == args.binary_case)] = -1
        valid_label[np.where(valid_label == args.binary_case)] = -1
        test_label[np.where(test_label == args.binary_case)] = -1

    ## Cognitive score case
    if args.cognitive_score == True:
        mmse_train_feature = train_data[:, :, 3:6]
        mmse_valid_feature = valid_data[:, :, 3:6]
        mmse_test_feature = test_data[:, :, 3:6]

        train_cog_norm_feature, train_cog_norm_mask = masking_cogntive_score(mmse_train_feature)
        valid_cog_norm_feature, valid_cog_norm_mask = masking_cogntive_score(mmse_valid_feature)
        test_cog_norm_feature, test_cog_norm_mask = masking_cogntive_score(mmse_test_feature)

        model_train_input = np.concatenate((norm_train_feature, train_cog_norm_feature), axis=2)
        model_train_mask = np.concatenate((train_mask, train_cog_norm_mask), axis=2)
        model_valid_input = np.concatenate((norm_valid_feature, valid_cog_norm_feature), axis=2)
        model_valid_mask = np.concatenate((valid_mask, valid_cog_norm_mask), axis=2)
        model_test_input = np.concatenate((norm_test_feature, test_cog_norm_feature), axis=2)
        model_test_mask = np.concatenate((test_mask, test_cog_norm_mask), axis=2)

    train_label = train_label - 1
    valid_label = valid_label - 1
    test_label = test_label - 1

    '''
    Define Dataloader
    '''
    if args.cognitive_score == True:
        writelog(f, 'Considering the cognitive scores')
        writelog(f, 'Training Dataset Loading')
        train_loader = sample_loader(model_train_input, np.asarray(model_train_mask), train_label, args.batch_size, is_train=True)
        writelog(f, 'Validation Dataset Loading')
        valid_loader = sample_loader(model_valid_input, np.asarray(model_valid_mask), valid_label, model_valid_input.shape[0])
        writelog(f, 'Test Dataset Loading')
        test_loader = sample_loader(model_test_input, np.asarray(model_test_mask), test_label, model_test_input.shape[0])
        dataloaders = {'train': train_loader,
                       'valid': valid_loader,
                       'test': test_loader}
    else:
        writelog(f, 'Not Considering the cognitive scores')
        writelog(f, 'Training Dataset Loading')
        train_loader = sample_loader(norm_train_feature, np.asarray(train_mask), train_label, args.batch_size, is_train=True)
        writelog(f, 'Validation Dataset Loading')
        valid_loader = sample_loader(norm_valid_feature, np.asarray(valid_mask), valid_label, norm_valid_feature.shape[0])
        writelog(f, 'Test Dataset Loading')
        test_loader = sample_loader(norm_test_feature, np.asarray(test_mask), test_label, norm_test_feature.shape[0])
        dataloaders = {'train': train_loader,
                       'valid': valid_loader,
                       'test': test_loader}

    # Define Model & Optimizer
    criterion_reg = nn.MSELoss()
    criterion_cls = FocalLoss(gamma=args.gamma, ignore_index=-2)

    if args.cognitive_score == True:
        model = model.Cog_Model(args.hid_size, args.impute_weight, args.reg_weight, args.label_weight, args.task).to(device)
    # else:
    #     model = model.Model(args.hid_size, args.impute_weight, args.reg_weight,
    #                                                        args.label_weight, args.task).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writelog(f, 'Total params is {}'.format(total_params))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Storing the Best Evaluation Metrics
    best_epoch = 0
    bestValidAUC = 0
    best_acc = 0
    best_balacc = 0
    best_auc = 0
    best_auprc = 0
    best_sen = 0
    best_spec = 0
    best_prec = 0
    best_recall = 0
    best_mae = 0
    best_mre = 0
    best_feature1 = 0
    best_feature2 = 0
    best_feature3 = 0
    best_feature4 = 0
    best_feature5 = 0
    best_feature6 = 0
    best_mae = 0
    best_mre = 0

    # Training, Validation, Test Loop
    for epoch in range(args.epochs):
        # Define several Phases
        writelog(f, '---------------')
        writelog(f, 'Epoch ' + str(epoch))
        for phase in ['train', 'valid', 'test']:
            if phase == 'train':
                model.train()
                lda_features = []
                lda_labels = []
            else:
                lda_v_features = []
                lda_v_labels = []
                labels = []
                pred_scores = []
                pred_labels = []

                evals = []
                eval_masks = []
                imputations = []

                save_impute = []
                save_label = []
                model.eval()

            phase_loss = 0
            n_batches = 0

            # Loop over the minibatch
            for i, data in enumerate(dataloaders[phase]):

                # Set Grad to be True only for phase=train
                with torch.set_grad_enabled(phase == 'train'):
                    data = to_var(data)

                    if phase == 'train':
                        ret = model.run_on_batch(data, optimizer, criterion_reg, criterion_cls, args.task, epoch)
                    else:
                        ret = model.run_on_batch(data, None, criterion_reg, criterion_cls, args.task)

                    # Check whether lossR is nan
                    phase_loss += ret['loss'].item()
                    n_batches += 1

                    # Calculate performance
                    if phase == 'train':
                        ########################################################################################
                        pred_feature = ret['predictions_feature'].data.cpu().numpy()
                        label = ret['labels'].data.cpu().numpy()

                        # collect test label & prediction for training LDA classifier
                        if args.task == 0: #single-task
                            lda_features += pred_feature.tolist()
                            lda_labels += label.tolist()

                    if phase != 'train':
                        if args.task == 1:  # single-task

                            pred_score = ret['predictions'].data.cpu().numpy()
                            pred_label = np.argmax(ret['predictions'].data.cpu().numpy(), axis=1)
                            label = ret['labels'].data.cpu().numpy()

                            filtered_pred_label = pred_label[np.where(label != -2)[0]]
                            filtered_pred_score = pred_score[np.where(label != -2)[0]]
                            filtered_label = label[np.where(label != -2)[0]]

                            #---------------------------------------------------------------------------------------------
                            ## AD vs. CN
                            label_y_AN = filtered_label[np.where(filtered_label != 1)[0]]
                            label_y_AN[label_y_AN == 2] = 1
                            pred_A_N = filtered_pred_score[np.where(filtered_label != 1)[0]][:,(0,2)][:,1:]
                            pred_A_N_label = np.argmax(pred_A_N, 1)

                            an_auc, an_auprc, an_acc, _, _, _, _, _ = calculate_performance(label_y_AN, pred_A_N, pred_A_N_label, None)
                            an_prec, an_recall, an_spec, an_sen, an_balacc, _ = calculate_performance_ver2(label_y_AN, pred_A_N_label, None)
                            writelog(f, '---------------')
                            writelog(f, 'AD vs CN case')
                            writelog(f, 'AUC : ' + str(an_auc))
                            writelog(f, 'AUC PRC : ' + str(an_auprc))
                            writelog(f, 'Accuracy : ' + str(an_acc))
                            writelog(f, 'BalACC : ' + str(an_balacc))
                            writelog(f, 'Sensitivity : ' + str(an_sen))
                            writelog(f, 'Specificity : ' + str(an_spec))
                            writelog(f, 'Precision : ' + str(an_prec))
                            writelog(f, 'Recall : ' + str(an_recall))

                            # ---------------------------------------------------------------------------------------------
                            ## AD vs. MCI
                            label_y_AM = filtered_label[np.where(filtered_label != 0)[0]]
                            label_y_AM[label_y_AM == 1] = 0
                            label_y_AM[label_y_AM == 2] = 1
                            pred_A_M = filtered_pred_score[np.where(filtered_label != 0)[0]][:, 1:][:, 1:]
                            pred_A_M_label = np.argmax(pred_A_M, 1)

                            am_auc, am_auprc, am_acc, _, _, _, _, _ = calculate_performance(label_y_AM, pred_A_M, pred_A_M_label, None)
                            am_prec, am_recall, am_spec, am_sen, am_balacc, _ = calculate_performance_ver2(label_y_AM, pred_A_M_label, None)
                            writelog(f, '---------------')
                            writelog(f, 'AD vs MCI case')
                            writelog(f, 'AUC : ' + str(am_auc))
                            writelog(f, 'AUC PRC : ' + str(am_auprc))
                            writelog(f, 'Accuracy : ' + str(am_acc))
                            writelog(f, 'BalACC : ' + str(am_balacc))
                            writelog(f, 'Sensitivity : ' + str(am_sen))
                            writelog(f, 'Specificity : ' + str(am_spec))
                            writelog(f, 'Precision : ' + str(am_prec))
                            writelog(f, 'Recall : ' + str(am_recall))
                            # ---------------------------------------------------------------------------------------------
                            ## MCI vs. CN
                            label_y_MN = filtered_label[np.where(filtered_label != 2)[0]]
                            pred_M_N = filtered_pred_score[np.where(filtered_label != 2)[0]][:, 1:][:, 1:]
                            pred_M_N_label = np.argmax(pred_M_N, 1)

                            mn_auc, mn_auprc, mn_acc,_ ,_ ,_ ,_ ,_  = calculate_performance(label_y_MN, pred_M_N, pred_M_N_label, None)
                            mn_prec, mn_recall, mn_spec, mn_sen, mn_balacc, _ = calculate_performance_ver2(label_y_MN, pred_M_N_label, None)
                            writelog(f, '---------------')
                            writelog(f, 'MCI vs CN case')
                            writelog(f, 'AUC : ' + str(mn_auc))
                            writelog(f, 'AUC PRC : ' + str(mn_auprc))
                            writelog(f, 'Accuracy : ' + str(mn_acc))
                            writelog(f, 'BalACC : ' + str(mn_balacc))
                            writelog(f, 'Sensitivity : ' + str(mn_sen))
                            writelog(f, 'Specificity : ' + str(mn_spec))
                            writelog(f, 'Precision : ' + str(mn_prec))
                            writelog(f, 'Recall : ' + str(mn_recall))
                            # ---------------------------------------------------------------------------------------------

                            mauc = MAUC(np.concatenate((filtered_label, filtered_pred_score), axis=1), num_classes=3)
                            auc, m_auprc, m_acc, _, _, _, _, _ = calculate_performance(filtered_label, filtered_pred_score, filtered_pred_label, True)
                            m_prec, m_recall, m_spec, m_sen, m_balacc, _ = calculate_performance_ver2(filtered_label, filtered_pred_label, True)
                            writelog(f, '---------------')
                            writelog(f, 'AD vs MCI vs CN case')
                            writelog(f, 'MAUC : ' + str(mauc))
                            writelog(f, 'AUC PRC : ' + str(m_auprc))
                            writelog(f, 'Accuracy : ' + str(m_acc))
                            writelog(f, 'BalACC : ' + str(m_balacc))
                            writelog(f, 'Sensitivity : ' + str(m_sen))
                            writelog(f, 'Specificity : ' + str(m_spec))
                            writelog(f, 'Precision : ' + str(m_prec))
                            writelog(f, 'Recall : ' + str(m_recall))
                            writelog(f, '---------------')

                            eval_mask = ret['eval_masks'].data.cpu().numpy()
                            eval_ = ret['evals'].data.cpu().numpy()
                            imputation = ret['imputations'].data.cpu().numpy()

                            eval_masks += eval_mask.tolist()
                            evals += eval_.tolist()
                            imputations += imputation.tolist()

                            output_data = ret['predictions_feature'].data.cpu().numpy()
                            shift_mask = ret['shifted_mask'].data.cpu().numpy()
                            shift_data = ret['shifted_data'].data.cpu().numpy()

                            plot_mae, plot_mse = regression(f, output_data, shift_data, shift_mask, estim_c, estim_m)
                            writelog(f,'-------Cognitive score (Root)----------')
                            plot_mae_cog1, plot_mse_cog1 = regression_cog(f, ret['predict_mmse'].data.cpu().numpy(), data['forward']['values'][:, 1:, 6:7].data.cpu().numpy(), data['forward']['masks'][:, 1:, 6:7].data.cpu().numpy(), original_cog=np.array([30]))
                            plot_mae_cog2, plot_mse_cog2 = regression_cog(f, ret['predict_ad11'].data.cpu().numpy(), data['forward']['values'][:, 1:, 7:8].data.cpu().numpy(), data['forward']['masks'][:, 1:, 7:8].data.cpu().numpy(), original_cog=np.array([70]))
                            plot_mae_cog3, plot_mse_cog3 = regression_cog(f, ret['predict_ad13'].data.cpu().numpy(), data['forward']['values'][:, 1:, 8:9].data.cpu().numpy(), data['forward']['masks'][:, 1:, 8:9].data.cpu().numpy(), original_cog=np.array([85]))

                        ##imputation metric
                        eval_masks = np.asarray(eval_masks)[:,:10,:6].reshape(-1,6)
                        evals = np.asarray(evals)[:,:10,:6].reshape(-1,6)[np.where(eval_masks == 1)]
                        imputations = np.asarray(imputations)[:,:,:6].reshape(-1,6)[np.where(eval_masks == 1)]

                        mae = np.abs(evals - imputations).mean()
                        mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()

                        writelog(f, 'MAE : ' + str(mae))
                        writelog(f, 'MRE : ' + str(mre))

            # Log and save the batch losses
            losses = phase_loss / n_batches
            writelog(f, phase + ' loss : ' + str(losses))

            if phase != 'train':
                if phase == 'valid':
                    # Save the model if we got the best AUC
                    if mauc > bestValidAUC:
                        [os.remove(f) for f in glob.glob(os.path.join(dir + 'model/'+ str(k) + '/' + '*_{}.pt'.format(args.whichmodel)))]
                        torch.save(model, dir + 'model/'+ str(k) + '/' + str(epoch) + '_{}.pt'.format(args.whichmodel))
                        writelog(f, 'Best validation AUC is found! Validation MAUC : ' + str(mauc))
                        if args.task == 0:
                            writelog(f, 'AN AUC : ' + str(an_auc))
                            writelog(f, 'AM AUC : ' + str(am_auc))
                            writelog(f, 'MN AUC : ' + str(mn_auc))
                            writelog(f, 'Models at Epoch ' + str(k) + '/' + str(epoch) + ' are saved!')
                        bestValidAUC = mauc
                        best_epoch = epoch
                elif phase == 'test':
                    # Store KFold Performance based on best epoch of validation
                    if epoch == best_epoch:
                        best_acc = m_acc
                        best_balacc = m_balacc
                        best_auc = mauc
                        best_auprc = m_auprc
                        best_sen = m_sen
                        best_spec = m_spec
                        best_prec = m_prec
                        best_recall = m_recall
                        best_mae = mae
                        best_mre = mre
                        best_feature1 = plot_mae[0]
                        best_feature2 = plot_mae[1]
                        best_feature3 = plot_mae[2]
                        best_feature4 = plot_mae[3]
                        best_feature5 = plot_mae[4]
                        best_feature6 = plot_mae[5]
                        best_cog1 = plot_mse_cog1
                        best_cog2 = plot_mse_cog2
                        best_cog3 = plot_mse_cog3

            # Tensorflow Logging
            if phase=='train':
                info = {'loss': losses}
            else:
                if args.task == 0:
                    info = {'loss': losses,
                            'mae': mae,
                            'mre': mre,
                            'balacc': m_balacc,
                            'AN auc': an_auc,
                            'AM auc': am_auc,
                            'MN auc': mn_auc,
                            'MAUC': mauc,
                            'auc_prc': m_auprc,
                            'sens': m_sen,
                            'spec': m_spec,
                            'precision': m_prec,
                            'recall': m_recall,
                            'f1_mae': plot_mae[0],
                            'f2_mae': plot_mae[1],
                            'f3_mae': plot_mae[2],
                            'f4_mae': plot_mae[3],
                            'f5_mae': plot_mae[4],
                            'f6_mae': plot_mae[5],
                            'f1_mse': plot_mse[0],
                            'f2_mse': plot_mse[1],
                            'f3_mse': plot_mse[2],
                            'f4_mse': plot_mse[3],
                            'f5_mse': plot_mse[4],
                            'f6_mse': plot_mse[5]
                            }
                else:
                    info = {'loss': losses,
                            'mae': mae,
                            'mre': mre,
                            'balacc': m_balacc,
                            'MAUC': mauc,
                            'auc_prc': m_auprc,
                            'sens': m_sen,
                            'spec': m_spec,
                            'precision': m_prec,
                            'recall': m_recall,
                            'f1_mae': plot_mae[0],
                            'f2_mae': plot_mae[1],
                            'f3_mae': plot_mae[2],
                            'f4_mae': plot_mae[3],
                            'f5_mae': plot_mae[4],
                            'f6_mae': plot_mae[5],
                            'cog1_mse': plot_mse_cog1,
                            'cog2_mse': plot_mse_cog2,
                            'cog3_mse': plot_mse_cog3
                            }

    writelog(f, 'END OF KFOLD ' + str(k))
    writelog(f, '---------------')
    writelog(f, 'Best Epoch For Testing : ' + str(best_epoch))
    writelog(f, 'AUC : ' + str(best_auc))
    writelog(f, 'AUC PRC : ' + str(best_auprc))
    writelog(f, 'Accuracy : ' + str(best_acc))
    writelog(f, 'BalACC : ' + str(best_balacc))
    writelog(f, 'Sensitivity : ' + str(best_sen))
    writelog(f, 'Specificity : ' + str(best_spec))
    writelog(f, 'Precision : ' + str(best_prec))
    writelog(f, 'Recall : ' + str(best_recall))
    writelog(f, 'Feature : ' + str(best_feature1))
    writelog(f, 'Feature : ' + str(best_feature2))
    writelog(f, 'Feature : ' + str(best_feature3))
    writelog(f, 'Feature : ' + str(best_feature4))
    writelog(f, 'Feature : ' + str(best_feature5))
    writelog(f, 'Feature : ' + str(best_feature6))
    writelog(f, 'Cog1 : ' + str(best_cog1))
    writelog(f, 'Cog2 : ' + str(best_cog2))
    writelog(f, 'Cog3 : ' + str(best_cog3))
    writelog(f, 'MAE : ' + str(mae))
    writelog(f, 'MRE : ' + str(mre))


    kfold_auc.append(best_auc)
    kfold_auprc.append(best_auprc)
    kfold_acc.append(best_acc)
    kfold_balacc.append(best_balacc)
    kfold_sen.append(best_sen)
    kfold_spec.append(best_spec)
    kfold_prec.append(best_prec)
    kfold_recall.append(best_recall)
    kfold_feature1.append(best_feature1)
    kfold_feature2.append(best_feature2)
    kfold_feature3.append(best_feature3)
    kfold_feature4.append(best_feature4)
    kfold_feature5.append(best_feature5)
    kfold_feature6.append(best_feature6)
    kfold_cog1.append(best_cog1)
    kfold_cog2.append(best_cog2)
    kfold_cog3.append(best_cog3)
    kfold_mae.append(best_mae)
    kfold_mre.append(best_mre)


writelog(f, '---------------')
writelog(f, 'SUMMARY OF ALL KFOLD')
k_fold = 5
mean_acc, std_acc = round(np.mean(kfold_acc), k_fold), round(np.std(kfold_acc), k_fold)
mean_auc, std_auc = round(np.mean(kfold_auc), k_fold), round(np.std(kfold_auc), k_fold)
mean_auprc, std_auprc = round(np.mean(kfold_auprc), k_fold), round(np.std(kfold_auprc), k_fold)
mean_sen, std_sen = round(np.mean(kfold_sen), k_fold), round(np.std(kfold_sen), k_fold)
mean_spec, std_spec = round(np.mean(kfold_spec), k_fold), round(np.std(kfold_spec), k_fold)
mean_prec, std_prec = round(np.mean(kfold_prec), k_fold), round(np.std(kfold_prec), k_fold)
mean_recall, std_recall = round(np.mean(kfold_recall), k_fold), round(np.std(kfold_recall), k_fold)
mean_balacc, std_balacc = round(np.mean(kfold_balacc), k_fold), round(np.std(kfold_balacc), k_fold)

mean_feature1, std_feature1 = round(np.mean(kfold_feature1), k_fold), round(np.std(kfold_feature1), k_fold)
mean_feature2, std_feature2 = round(np.mean(kfold_feature2), k_fold), round(np.std(kfold_feature2), k_fold)
mean_feature3, std_feature3 = round(np.mean(kfold_feature3), k_fold), round(np.std(kfold_feature3), k_fold)
mean_feature4, std_feature4 = round(np.mean(kfold_feature4), k_fold), round(np.std(kfold_feature4), k_fold)
mean_feature5, std_feature5 = round(np.mean(kfold_feature5), k_fold), round(np.std(kfold_feature5), k_fold)
mean_feature6, std_feature6 = round(np.mean(kfold_feature6), k_fold), round(np.std(kfold_feature6), k_fold)

mean_cog1, std_cog1 = round(np.mean(kfold_cog1), k_fold), round(np.std(kfold_cog1), k_fold)
mean_cog2, std_cog2 = round(np.mean(kfold_cog2), k_fold), round(np.std(kfold_cog2), k_fold)
mean_cog3, std_cog3 = round(np.mean(kfold_cog3), k_fold), round(np.std(kfold_cog3), k_fold)


mean_mae, std_mae = round(np.mean(kfold_mae), k_fold), round(np.std(kfold_mae), k_fold)
mean_mre, std_mre = round(np.mean(kfold_mre), k_fold), round(np.std(kfold_mre), k_fold)

writelog(f, 'AUC : ' + str(mean_auc) + ' + ' + str(std_auc))
writelog(f, 'AUC PRC : ' + str(mean_auprc) + ' + ' + str(std_auprc))
writelog(f, 'Accuracy : ' + str(mean_acc) + ' + ' + str(std_acc))
writelog(f, 'BalACC : ' + str(mean_balacc) + ' + ' + str(std_balacc))
writelog(f, 'Sensitivity : ' + str(mean_sen) + ' + ' + str(std_sen))
writelog(f, 'Specificity : ' + str(mean_spec) + ' + ' + str(std_spec))
writelog(f, 'Precision : ' + str(mean_prec) + ' + ' + str(std_prec))
writelog(f, 'Recall : ' + str(mean_recall) + ' + ' + str(std_recall))
writelog(f, 'Feature : ' + str(mean_feature1) + ' + ' + str(std_feature1))
writelog(f, 'Feature : ' + str(mean_feature2) + ' + ' + str(std_feature2))
writelog(f, 'Feature : ' + str(mean_feature3) + ' + ' + str(std_feature3))
writelog(f, 'Feature : ' + str(mean_feature4) + ' + ' + str(std_feature4))
writelog(f, 'Feature : ' + str(mean_feature5) + ' + ' + str(std_feature5))
writelog(f, 'Feature : ' + str(mean_feature6) + ' + ' + str(std_feature6))
writelog(f, 'MMSE : ' + str(mean_cog1) + ' + ' + str(std_cog1))
writelog(f, 'ADAS11 : ' + str(mean_cog2) + ' + ' + str(std_cog2))
writelog(f, 'ADAS13 : ' + str(mean_cog3) + ' + ' + str(std_cog3))
writelog(f, 'MAE : ' + str(mean_mae) + ' + ' + str(std_mae))
writelog(f, 'MRE : ' + str(mean_mre) + ' + ' + str(std_mre))
writelog(f, '---------------')
writelog(f, 'END OF CROSS VALIDATION TRAINING')
f.close()
