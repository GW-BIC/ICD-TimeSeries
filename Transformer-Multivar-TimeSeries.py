import pandas as pd
import numpy as np
import torch
from torch import nn
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.preprocessing import MinMaxScaler
from joblib import load
from sklearn import metrics

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def hasTask(task):
    tasks = [1,2,3,3.1,4,5,6,7,8,9]

    if task in tasks:
        print('-'*10, 'Task:', task, '-'*10)
        print()
        return True
    return False

def printDuration(seconds):
    seconds = int(seconds)
    timestrs = [str(seconds%60)+' sec']
    if seconds>60:
        minutes = seconds//60
        timestrs.append(str(minutes%60)+' min')
        if minutes>60:
            timestrs.append(str(minutes//60)+' hr')
    print('Total time:', ' '.join(timestrs[::-1]))

mainDir = r'C:\Users\'
plt.close('all')
for i in range(1,701):
    cluster = i
    print(cluster)
    
    # Task 1: read data file    
    if hasTask(1):
        #Due to the data access rule, we didn't provide the input data. But the columns of the input data were provided in DataColumns.xlxs
        df = pd.read_csv(mainDir +'.\data\cl'+ str(cluster) +'_input_data_scaled.csv') 
        df.drop(columns=['wk_id'],inplace=True)

    # Task 2: define variables           
    if hasTask(2):
        transition_id = df.index[df['weekbegin']=='2015-09-27'].tolist()[0]
        longseq_len = df.shape[0]-1 
        seq_len = 12         
        test_start = transition_id - seq_len    
        n_seqs = longseq_len-seq_len
        n_train_vali_seqs = test_start - seq_len
        test_vali_split = int(transition_id*0.8)
        n_vars = df.shape[1]-1

        weight_decay = 0.0001
        dropout = 0.1
        d_embed = 32
        d_hid = 30

    # Task 3: create tensors for covariates and targets    
    if hasTask(3):
        var_vals = torch.zeros(seq_len,n_seqs,n_vars)
        for i in range(n_seqs): 
            var_vals[:,i,:] = torch.tensor(df.iloc[i:i+seq_len,1:n_vars+1].values.astype(np.float32)) 
    
        targets = torch.tensor(df['total_visit_c'].values[seq_len:longseq_len].astype(np.float32))
        cov_curr = torch.tensor(df.iloc[seq_len:longseq_len,2:n_vars+1].values.astype(np.float32))
        print()
    
    # Task 3.1 create data split for training, validation and test   
    if hasTask(3.1):
        train_vali_idxs = np.arange(n_train_vali_seqs)
        np.random.seed(1)
        np.random.shuffle(train_vali_idxs)
        slice_tr = train_vali_idxs[:test_vali_split]
        slice_vl = train_vali_idxs[test_vali_split:]
        slice_te = np.arange(n_train_vali_seqs,n_seqs)

    # Task 4: create batches   
    if hasTask(4):
        idxs = np.arange(n_seqs)
        np.random.seed(1)
        np.random.shuffle(idxs)
        batch_size = 8
        batches = []
        batch_idxs_tr = []
    
        for i in range((n_seqs-1)//batch_size+1):
            b = batch_size
            b_idxs = idxs[i*b:(i+1)*b]
            b_src = var_vals[:,b_idxs,:]

            batches.append([b_idxs,b_src])
            batch_idxs_tr.append(np.array([i for i,e in enumerate(b_idxs) if e in slice_tr]))
   
    # Task 5: To define the model 
    if hasTask(5):
        torch.manual_seed(1)
        class TemporalEncoding(nn.Module):
            def __init__(self,d_embed,max_len=seq_len):
                super(TemporalEncoding,self).__init__()
    
                te = torch.zeros(max_len,d_embed)
                times = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
                omegas = torch.exp(torch.arange(0,d_embed,2).float()*(-math.log(10000.0)/d_embed))
                te[:,0::2] = torch.sin(times*omegas)
                te[:,1::2] = torch.cos(times*omegas)
                self.register_buffer('te',te)
    
            def forward(self,x):
                x = x + self.te.unsqueeze(1)
                return x 
    
        class TransformerModel(nn.Module):
            def __init__(self,n_vars,d_embed,nhead,d_hid,nblocks,dropout=dropout):
                super(TransformerModel,self).__init__()
                from torch.nn import TransformerEncoder, TransformerEncoderLayer
                self.d_embed = d_embed
    
                self.enc_input = nn.Linear(n_vars,d_embed)
                self.time_encoder = TemporalEncoding(d_embed)                #the default input for encoder layers is [seq,bs,d] unless you set batch_first=True
                encoder_layers = TransformerEncoderLayer(d_embed,nhead,d_hid,dropout)
                self.transformer_encoder = TransformerEncoder(encoder_layers,nblocks)
    
            def forward(self,src,src_key_padding_mask=None):
                src = self.enc_input(src) 
                src = self.time_encoder(src) 
                result = self.transformer_encoder(src,src_key_padding_mask=src_key_padding_mask) 
                return result[-1] 
    
        class ResMLP(nn.Module):
            def __init__(self,d_in,d_hid,d_out):
                super(ResMLP,self).__init__()
                self.fc1 = nn.Linear(d_in,d_out)
                self.dropout1 = nn.Dropout(p=dropout)
                self.norm1 = nn.LayerNorm(d_out)
                self.fc2 = nn.Linear(d_out,d_hid)
                self.dropout2 = nn.Dropout(p=dropout)
                self.fc3 = nn.Linear(d_hid,d_out)
                self.dropout3 = nn.Dropout(p=dropout)
                self.norm2 = nn.LayerNorm(d_out)
                
            def forward(self,input):
                output = self.norm1(self.dropout1(self.fc1(input)))
                output2 = self.dropout2(nn.functional.relu(self.fc2(output)))
                output = output + self.dropout3(self.fc3(output2))
                output = self.norm2(output)
                return output
    
        class Model(nn.Module):
            def __init__(self):
                super(Model,self).__init__()
                self.transformer = TransformerModel(n_vars,d_embed=d_embed,nhead=4,d_hid=d_hid,nblocks=2)
               
                self.inp = ResMLP(n_vars-1,d_hid,d_embed)
                self.mlp = ResMLP(d_embed,d_hid,d_embed)
                self.out = nn.Linear(d_embed,1)
    
            def forward(self,cov_curr,src,src_key_padding_mask=None):
                result1 = self.inp(cov_curr) 
                result2 = self.transformer(src,src_key_padding_mask) 
                result = result1+result2
                result = self.mlp(result) 
                result = self.out(result) 
                return result.flatten() 
    
        model = Model()
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9,weight_decay = weight_decay,nesterov=True)        
        np.random.seed(1)
        torch.set_num_threads(1)
        MSEs_tr,MSEs_vl,MSEs_te = [],[],[]
        
        def mse(x,y):
            return ((x-y)**2).mean()
    
    # Task 6: training and validating data 
    if hasTask(6):                       
        # To train and evaluate the data
        num_epochs = 50
        start = time.time()
        print(time.strftime('%H:%M:%S'))

        def plot_pred(y_pred,targets):
            plt.cla()
            plt.figure(figsize=(15,8))
            scaler_name = mainDir +'/scaler/scaler_cl'+str(cluster)+'.joblib'
            scaler = load(scaler_name)
            
            pred_visit = scaler.inverse_transform(y_pred.reshape(-1,1))
            target_visit = scaler.inverse_transform(targets.reshape(-1,1))
            
            plot_title = 'Cluster' + str(cluster) + 'seq' + str(seq_len) + 'Epochs' + str(epochs) + 'Prediction'
            plt.plot(target_visit, label = 'targets',c='k')
            plt.plot(pred_visit, label = 'predicts',c='b')
            plt.plot(slice_vl,pred_visit[slice_vl],'o', color = 'orange',markersize = 4)
            plt.plot(slice_te,pred_visit[slice_te], label = 'test',c='purple')            #plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%Y'))
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(24))
            plt.xticks(rotation = 45)
            plt.title(plot_title)
            plt.xlim(0)
            plt.xlabel("Week")
            plt.ylabel("Visit Counts")

            plt.axvline( x= transition_id-seq_len, ls ='--',c='b')
           
            plt.legend(fontsize='x-large', frameon= False)
            plt.savefig(graph_dir+f"/{plot_title}.png")
            plt.close()
        

        for epoch in range(num_epochs+1):
            if epoch>=1:
                epochs += 1
                rand_idxs = list(range(len(batches)))
                np.random.shuffle(rand_idxs)
                model.train()
                for r,i in enumerate(rand_idxs):
                    if r%20==0:
                        print('.',end='')
                    idxs,src = batches[i]
                    ii = batch_idxs_tr[i]
                    if ii.shape[0]==0:
                        continue
                    output = model(cov_curr[idxs[ii]],src[:,ii,:])
                    loss = criterion(output,targets[idxs[ii]])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                print(' ',end='')
            elif epochs>0:
                continue
            y_pred = torch.zeros(targets.size(0))
            model.eval()
            with torch.no_grad():
                for i,batch in enumerate(batches):
                    if i%20==0:
                        print('.',end='')
                    idxs,src = batch
                    output = model(cov_curr[idxs],src)
                    y_pred[idxs] = output
                    
            print()
            mse_tr = mse(targets[slice_tr],y_pred[slice_tr])
            mse_vl = mse(targets[slice_vl],y_pred[slice_vl])
            mse_te = mse(targets[slice_te],y_pred[slice_te])    
            if  epochs%20 == 0:
                print('{:2d}: {:.6f}    {:.6f}     {:s}'.format(
                    epochs,mse_tr,mse_vl,time.strftime('%H:%M:%S')))
            MSEs_tr.append(mse_tr.detach().item())
            MSEs_vl.append(mse_vl.detach().item())
            MSEs_te.append(mse_te.detach().item())
            
            if  epochs!=0 and epochs%50 == 0:
                plot_pred(y_pred, targets)
                best_model_name = mainDir + '\model\Cluster' + str(cluster) + 'seq' + str(seq_len) +'Epochs' + str(epochs)+'.pt'
                torch.save(model.state_dict(),best_model_name)               
        print()
        printDuration(time.time()-start)
        
    # Task 7: Calculate prediction interval using Residual bootstrap 
    if hasTask(7):
        epochs = 150
        best_model_name = mainDir + '\model\epochs150\Cluster' + str(cluster) +  'seq' + str(seq_len) + 'Epochs' + str(epochs)+'.pt'
        checkpoint = torch.load(best_model_name)
        model.load_state_dict(checkpoint['model_state_dict'])

        y_pred = torch.zeros(targets.size(0))
        model.eval()
        with torch.no_grad():
            for j in range(n_seqs):
                    output = model(cov_curr[j].view(1,-1),var_vals[:,j,:].view(seq_len,1,-1))
                    y_pred[j] = output
         
        scaler_name = mainDir +'/scaler/scaler_cl'+str(cluster)+'.joblib'
        scaler = load(scaler_name)

        pred_visit = (scaler.inverse_transform(y_pred.reshape(-1,1)))
        target_visit = (scaler.inverse_transform(targets.reshape(-1,1)))

        res = (pred_visit - target_visit).reshape(-1)
        res_pre=res[:transition_id - seq_len] # predictive residual for prior 2015-10

        res_pre_mean = np.mean(res_pre)
        res_pre_stev = np.std(res_pre)       
        
        res_pre_te_mean = np.mean(res[test_start - seq_len:transition_id - seq_len])
        res_pre_te_stev = np.std(res[test_start - seq_len:transition_id - seq_len])    

        # Residual bootstrap
        alpha = 0.05
            
        res_tr = res[slice_tr]
        res_vl = res[slice_vl]
        res_te = res[slice_te]
        
        bootstrap = np.asarray([np.random.choice(res_pre, size = res_pre.shape) for _ in range(1000)])
        q_bootstrap = np.quantile(bootstrap, q = [alpha/2, 1 - alpha/2], axis=0)
            
        pred_visit_lower = pred_visit + q_bootstrap[0].mean()
        pred_visit_upper = pred_visit + q_bootstrap[1].mean()
        
        outlier = (target_visit.flatten() > pred_visit_upper.flatten()) | (target_visit.flatten() < pred_visit_lower.flatten())
        
        per_out_tr = round(outlier[slice_tr].sum()/outlier[slice_tr].shape[0],4)
        per_out_vl = round(outlier[slice_vl].sum()/outlier[slice_vl].shape[0],4)
        per_out_te = round(outlier[slice_te].sum()/outlier[slice_te].shape[0],4)

    # Task 8: Plot training loss, residual, prediction interval
    if hasTask(8):
        def plot_loss():
            plt.cla()
            title = 'Cluster' + str(cluster) + 'Training'
            plt.plot(MSEs_tr, label = 'training loss')
            plt.plot(MSEs_vl, label = 'validation loss')
            plt.plot(MSEs_te, label = 'testing loss')    
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.ylim(0,0.1)
            plt.legend()
            plt.savefig(graph_dir+f"/{title}.png")
            plt.close()
        
        def plot_res():          
            plot_title_line = 'Cluster' + str(cluster) + 'seq' + str(seq_len) + 'Epochs' + str(epochs) + 'residual'

            # plot residual line graph
            plt.cla()
            plt.figure(figsize=(15,8))            
            plt.plot(res, label = 'residual',c='c')
            plt.plot(pd.DataFrame(res).ewm(span=4).mean(), label = 'res_ewm4',c='orange')
        
            plt.axhline(res_pre_mean, c='k')
            plt.axhline(res_pre_mean+1.96*res_pre_stev, c='dimgray')
            plt.axhline(res_pre_mean-1.96*res_pre_stev, c='dimgray')
            
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(24))
            plt.xticks(rotation = 45)
            plt.title(plot_title_line)
            plt.xlim(0)

            #plt.text(n_train_vali_seqs,target_visit[n_train_vali_seqs],'test',c='b')
            plt.axvline( x= transition_id-seq_len, ls ='--',c='b')
           
            plt.legend(fontsize='x-large', frameon= False)
            plt.savefig(graph_dir+f"/{plot_title_line}.png")
            plt.close()

        def plot_res_dist():
            # plot residual distribution graph
            plt.cla()
            plt.figure(figsize=(16,8))

            plot_title_dist = 'Cluster' + str(cluster) + 'seq' + str(seq_len) + 'Epochs' + str(epochs) + 'training residual distribution'
            
            plt.subplot(2,2,1)            
            plt.title('Training Residuals Distribution')
            plt.hist(res_tr, bins=20)                      

            plt.subplot(2,2,2)            
            plt.title('Training Residuals Autocorrelation')
            plt.plot([pd.Series(res_tr).autocorr(lag=dt) for dt in range(1,52)])
            plt.ylim([-1,1])
            plt.axhline(0, ls ='--',c='k')
            plt.ylabel("Autocorrelation")
            plt.xlabel("Lags")          
            
            plt.subplot(2,2,3)            
            plt.title('Validation Residuals Distribution')
            plt.hist(res_vl, bins=20)  

            plt.subplot(2,2,4)            
            plt.title('Testing Residuals Distribution')
            plt.hist(res_te, bins=20)  
            plt.savefig(graph_dir+f"/{plot_title_dist}.png")
            plt.close()
        
        def plot_pred_interval():
            plt.cla()
            plt.figure(figsize=(15,8))
            
            plot_title = 'Cluster' + str(cluster) + 'seq' + str(seq_len) + 'Epochs' + str(epochs) + 'Prediction Interval'
            plt.plot(pd.DataFrame(target_visit).index,target_visit,'o', color = 'k',markersize = 2)

            plt.plot(target_visit, label = 'targets',c='k',linewidth = 2)
            plt.plot(pred_visit, label = 'predicts',c='b',linewidth = 2)
            plt.fill_between(pd.DataFrame(pred_visit).index, pred_visit_lower.flatten(),pred_visit_upper.flatten(),alpha=0.3)

            plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%Y'))
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(24))
            plt.xticks(rotation = 45)
            plt.title(plot_title)
            plt.xlim(0)
            plt.ylabel("Visit Counts")

            y_min, y_max = plt.gca().get_ylim()
            plt.text(10,y_max*0.95,f"per_outlier_tr: {per_out_tr}", fontsize= 14)
            plt.text(10,y_max*0.9,f"per_outlier_vl: {per_out_vl}", fontsize= 14)
            plt.text(10,y_max*0.85,f"per_out_te: {per_out_te}", fontsize= 14)

            plt.text(n_train_vali_seqs,target_visit[n_train_vali_seqs],'test',c='b')
            plt.axvline( x= transition_id-seq_len, ls ='--',c='b')       
            
            plt.legend(fontsize='x-large', frameon= False)
            plt.savefig(graph_dir+f"/{plot_title}.png")
            plt.close()
            
        plot_loss()
        plot_res()
        plot_res_dist()
        plot_pred_interval()
                 
    # Task 9: Write Prediction for the first 5 weeks of ICD transition
    if hasTask(9):
        import csv        
        file_name = mainDir + '\prediction\prediction_all_clusers_all_weeks.csv'               
        with open(file_name, 'a',newline='') as file:
            fieldnames = ['Cluster','Week','Ture_visit_ct','Pred_visit_ct'
            ,'pred_visit_upper','Pred_visit_lower','outlier','target_predict_ratio'            
            ,'residual_pre_mean','residual_pre_std','residual_pre_te_mean','residual_pre_te_std','residual','std_res'
            ]          
            writer=csv.DictWriter(file,fieldnames=fieldnames)
            #writer.writeheader()
            for i in range(len(target_visit)):
                writer.writerow({'Cluster': cluster
                ,'Week': df['weekbegin'][seq_len+i]
                ,'Ture_visit_ct': round(target_visit[i][0])
                ,'Pred_visit_ct': round(pred_visit[i][0])
                ,'pred_visit_upper': round(pred_visit_upper[i][0])
                ,'Pred_visit_lower': round(pred_visit_lower[i][0]) ,'outlier': outlier[i]
                ,'target_predict_ratio': round(target_visit[i][0]/pred_visit[i][0],2)
                ,'residual_pre_mean': round(res_pre_mean,2),'residual_pre_std': round(res_pre_stev,2)
                ,'residual_pre_te_mean': round(res_pre_te_mean,2),'residual_pre_te_std':round(res_pre_te_stev,2)
                ,'residual': round(res[i],2),'std_res': round(res[i]/res_pre_te_stev,2)
                })
            

    


