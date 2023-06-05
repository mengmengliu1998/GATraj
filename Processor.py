'''
Author: Mengmeng Liu
Date: 2022/09/24
'''
from utils import *
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
class Processor():
    def __init__(self, args):
        self.args = args
        Dataloader = DataLoader_bytrajec2
        self.lr=self.args.learning_rate
        self.dataloader_gt = Dataloader(args,is_gt=True)
        model = import_class(args.model)
        self.net = model(args)
        if self.args.phase == "train":
            print("self.args.phase",self.args.phase)
            self.net.train()
        else:
            self.net.eval()
        self.init_lr = self.args.learning_rate
        self.step_ratio = self.args.step_ratio
        self.lr_step=self.args.lr_step
        self.set_optimizer()
        self.epoch = 0
        self.load_model()
        # self.save_model(self.epoch)
        if self.args.using_cuda:
            self.net=self.net.cuda()
        else:
            self.net=self.net.cpu()
        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')



    def save_model(self,epoch):
        model_path= self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' +\
                                   str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path,_use_new_zipfile_serialization=False)


    def load_model(self):
        if self.args.load_model > 0:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path,map_location={'cuda:0': 'cuda:'+str(self.args.gpu)})
                # print("self.args.model_save_path",self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.epoch = int(model_epoch) + 1
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)
                for i in range(self.args.load_model):
                    self.scheduler.step()


    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss(reduce=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,\
        T_max = self.args.num_epochs, eta_min=self.args.eta_min)

    def playtest(self):
        print('Testing begin')
        test_error, test_final_error, _= self.test_epoch(self.args.load_model)
        print('Set: {}, epoch: {:.5f},test_error: {:.5f} test_final_error: {:.5f}'.format(self.args.test_set,self.args.load_model,test_error,test_final_error))

    def playtrain(self):
        print('Training begin')
        test_error, test_final_error,first_erro_test,val_final_error,val_error,val_erro_first=0,0,0,0,0,0
        for epoch in range(self.epoch, self.args.num_epochs+1):
            print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            train_loss = self.train_epoch(epoch)
            val_error, val_final_error, val_erro_first = self.val_epoch(epoch)
            self.scheduler.step()
            # if epoch == self.args.num_epochs:
            self.save_model(epoch)
            if epoch == self.args.num_epochs:
                test_error, test_final_error, first_erro_test = self.test_epoch(epoch)
            #log files
            self.log_file_curve.write(str(epoch) + ',' + str(train_loss) + ',' + str(
                val_error) + ',' + str(val_final_error) + ','+str(val_erro_first)+ ','\
                +str(test_error) + ',' + str(test_final_error) + ','+str(first_erro_test)+ '\n')

            self.log_file_curve.close()
            self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')
            #console log
            print('----epoch {} \n train_loss={:.5f}, valid_error={:.3f}, valid_final={:.3f}, valid_first={:.3f}\n\
                test_error={:.3f},test_final={:.3f},test_first={:.3f}\n'\
            .format(epoch, train_loss,val_error, val_final_error,val_erro_first,test_error,test_final_error,first_erro_test))
            model_path= self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + str(epoch) + '.tar'



    def train_epoch(self,epoch):
        """   batch_abs: the (orientated) batch
              batch_norm: the batch shifted by substracted the last position. ??? What is the impact of zeros
              shift_value: the last observed position
              seq_list: [seq_length, num_peds], mask for position with actual values at each frame for each ped
              nei_list: [seq_length, num_peds, num_peds], mask for neigbors at each frame
              nei_num: [seq_length, num_peds], neighbors at each frame for each ped
              batch_pednum: list, number of peds in each batch"""
        self.net.train()
        loss_epoch=0
        for batch in range(self.dataloader_gt.trainbatchnums):
            start = time.time()
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_train_batch(batch,epoch)#batch_split:[batch_size, 2]
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split  #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H]
            self.net.zero_grad()
            GATraj_loss, full_pre_tra = self.net.forward(inputs_fw, epoch, iftest=False)
            if GATraj_loss == 0:
                continue
            loss_epoch += GATraj_loss.item()
            GATraj_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            end= time.time()
            if batch%self.args.show_step == 0 and self.args.ifshow_detail:
                print('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f}'.\
                format(batch,self.dataloader_gt.trainbatchnums, epoch,GATraj_loss.item(), end - start))
        train_loss_epoch = loss_epoch / self.dataloader_gt.trainbatchnums


        return train_loss_epoch

    def val_epoch(self,epoch):
        self.net.eval()
        error_epoch,final_error_epoch, first_erro_epoch = 0,0,0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list= [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5,1e-5,1e-5

        for batch in range(self.dataloader_gt.valbatchnums):
            if batch%100 == 0:
                print('testing batch',batch,self.dataloader_gt.valbatchnums)
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_val_batch(batch,epoch) #batch_split:[batch_size, 2]
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt=tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split  #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H]
            GATraj_loss,full_pre_tra= self.net.forward(inputs_fw, epoch, iftest=True)
            if GATraj_loss == 0:
                continue
            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro,first_erro_cnt = \
                L2forTest(pre_tra, batch_norm_gt[1:, :, :2],self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)
                
            first_erro_epoch += min(first_erro_epoch_list)
            final_error_epoch += min(final_error_epoch_list)
            error_epoch += min(error_epoch_list)
            error_cnt_epoch += error_cnt
            final_error_cnt_epoch += final_error_cnt
            first_erro_cnt_epoch += first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch,first_erro_epoch/ first_erro_cnt_epoch

    def test_epoch(self,epoch):
        self.net.eval()
        error_epoch, final_error_epoch, first_erro_epoch = 0, 0, 0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5, 1e-5, 1e-5

        
        for batch in range(self.dataloader_gt.testbatchnums):
            if batch%100 == 0:
                print('testing batch',batch,self.dataloader_gt.testbatchnums)
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_test_batch(batch,epoch)
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H]
            GATraj_loss,full_pre_tra = self.net.forward(inputs_fw, epoch, iftest=True)
            if GATraj_loss == 0:
                continue

            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro,first_erro_cnt = \
                L2forTest(pre_tra, batch_norm_gt[1:, :, :2],self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)
            first_erro_epoch += min(first_erro_epoch_list)
            final_error_epoch += min(final_error_epoch_list)
            error_epoch += min(error_epoch_list)
            error_cnt_epoch += error_cnt
            final_error_cnt_epoch += final_error_cnt
            first_erro_cnt_epoch += first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch,first_erro_epoch/ first_erro_cnt_epoch
