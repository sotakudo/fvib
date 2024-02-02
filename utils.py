import torch
import numpy as np
from tqdm import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
from fvib import calc_K, FVIB
from loss import FVIBLoss, KLLoss, DistortionTaylorApproxLoss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_FVIB (n_epochs, d, net, train_loader, test_loader, betas, path, optimizer, scheduler, n_samples=12, device=device):
    K = calc_K(d)
    FVIBs = [FVIB(K, beta, n_samples=n_samples) for beta in betas]
    criterion = FVIBLoss(K) 
    KL = KLLoss()
    CE = nn.CrossEntropyLoss()
    DTA = DistortionTaylorApproxLoss(d)
    losses = []
    running_CEs, running_KLs, running_DTAs = [], [], []
    running_CEs_test, running_KLs_test, running_DTAs_test, corrects = [], [], [], []
    test_losses = []
    os.makedirs(path, exist_ok=True)
    for epoch in tqdm(range(n_epochs)):
        running_loss = 0.0
        running_CE, running_KL, running_DTA = [0.0 for _ in betas], [0.0 for _ in betas], [0.0 for _ in betas]
        net.train()
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()/len(train_loader)
            
            with torch.no_grad():
                for i, beta in enumerate(betas):
                    out, mean, var, _ = FVIBs[i](outputs)
                    running_CE[i] += CE(out, labels).item()/len(train_loader)
                    running_KL[i] += KL(mean, var).item()/len(train_loader)
                    W = (1-FVIBs[i].beta)**(1/2)*FVIBs[i].L.T
                    running_DTA[i] += DTA(W, torch.zeros(1, device=device), mean, var, labels).item()/len(train_loader)
          
        if scheduler is not None:
            scheduler.step()
        losses.append(running_loss)
        running_CEs.append(running_CE)
        running_KLs.append(running_KL)
        running_DTAs.append(running_DTA)
        
        correct = [0 for _ in betas]
        total = 0
        running_loss_test = 0
        running_CE_test, running_KL_test, running_DTA_test = [0.0 for _ in betas], [0.0 for _ in betas], [0.0 for _ in betas]
        #n_mini_batch = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                #images, labels = data
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_loss_test += loss.item()/len(test_loader)
                total += labels.size(0)
                for i, beta in enumerate(betas):
                    out, mean, var, y_hat = FVIBs[i](outputs)
                    running_CE_test[i] += CE(out, labels).item()/len(test_loader)
                    running_KL_test[i] += KL(mean, var).item()/len(test_loader)
                    W = (1-FVIBs[i].beta)**(1/2)*FVIBs[i].L.T
                    running_DTA_test[i] += DTA(W, torch.zeros(1, device=device), mean, var, labels).item()/len(test_loader)
                    _, predicted = torch.max(y_hat.data, 1)
                    correct[i] += (predicted == labels).sum().item()
                    
        corrects.append(correct)
        test_losses.append(running_loss_test)
        running_CEs_test.append(running_CE_test)
        running_KLs_test.append(running_KL_test)
        running_DTAs_test.append(running_DTA_test)
        
    accs = 100 * np.array(corrects) / total
    losses = np.array(losses)
    test_losses = np.array(test_losses)
    running_CEs = np.array(running_CEs)
    running_KLs = np.array(running_KLs)
    running_DTAs = np.array(running_DTAs)
    running_CEs_test = np.array(running_CEs_test)
    running_KLs_test = np.array(running_KLs_test)
    running_DTAs_test = np.array(running_DTAs_test)
    
    np.save(path + "/acc", accs)
    np.save(path + "/train_fvib_loss", losses)
    np.save(path + "/test_fvib_loss", test_losses)
    np.save(path + "/train_ce_loss", running_CEs)
    np.save(path + "/train_kl_loss", running_KLs)
    np.save(path + "/train_dta_loss", running_DTAs)
    np.save(path + "/test_ce_loss", running_CEs_test)
    np.save(path + "/test_kl_loss", running_KLs_test)
    np.save(path + "/test_dta_loss", running_DTAs_test)
    torch.save(K, path + '/K.pt')
    torch.save(net.state_dict(), path + '/fe_weight.pth')
    ff = open(path+'/settings.txt', 'w')
    ff.write(f'epoch:{n_epochs}\nvib:FVIB\nbetas:{betas}')
    ff.close()
    
    

    
def train_VIB (n_epochs, d, net, vib, train_loader, test_loader, beta, path, net_optimizer, vib_optimizer, net_scheduler=None, vib_scheduler=None, use_dta_for_loss=False, is_deterministic= False, label_smoothing = 0.0, sq_vib = False, device=device):
    KL = KLLoss()
    CE = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
    DTA = DistortionTaylorApproxLoss(d)
    losses = []
    running_CEs, running_KLs, running_DTAs = [], [], []
    running_CEs_test, running_KLs_test, running_DTAs_test, corrects = [], [], [], []
    test_losses = []
    os.makedirs(path, exist_ok=True)
    for epoch in tqdm(range(n_epochs)):
        running_CE, running_KL, running_DTA = 0.0, 0.0, 0.0
        net.train()
        vib.train()
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            net_optimizer.zero_grad()
            vib_optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            out, mean, var, _ = vib(outputs)
            
            if use_dta_for_loss:
                dta = DTA(vib.fc.weight, vib.fc.bias, mean, var, labels)
                kl = KL(mean, var)
                loss = dta+beta*kl
                with torch.no_grad():
                    ce = CE(out, labels)
            else:
                ce = CE(out, labels)
                if is_deterministic:
                    loss = ce
                    kl = torch.zeros(1) #dummy value
                    dta = torch.zeros(1) #dummy value
                else:
                    kl = KL(mean, var)
                    if sq_vib:
                        loss = ce+beta*(kl**2)
                    else:
                        loss = ce+beta*kl
                    with torch.no_grad():
                        dta = DTA(vib.fc.weight, vib.fc.bias, mean, var, labels)
            
            loss.backward()
            net_optimizer.step()
            vib_optimizer.step()
            
            with torch.no_grad():
                running_CE += ce.item()/len(train_loader)
                running_KL += kl.item()/len(train_loader)
                running_DTA += dta.item()/len(train_loader)
                   
        if net_scheduler is not None:
            net_scheduler.step()
        if vib_scheduler is not None:
            vib_scheduler.step()
        running_CEs.append(running_CE)
        running_KLs.append(running_KL)
        running_DTAs.append(running_DTA)
        correct = 0
        total = 0
        running_CE_test, running_KL_test, running_DTA_test = 0.0, 0.0, 0.0
        net.eval()
        vib.eval()
        with torch.no_grad():
            for data in test_loader:
                #images, labels = data
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                out, mean, var, y_hat = vib(outputs)
                
                running_CE_test += CE(out, labels).item()/len(test_loader)
                if not is_deterministic:
                    running_KL_test += KL(mean, var).item()/len(test_loader)
                    running_DTA_test += DTA(vib.fc.weight, vib.fc.bias, mean, var, labels).item()/len(test_loader)
                _, predicted = torch.max(y_hat.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                    
        corrects.append(correct)
        running_CEs_test.append(running_CE_test)
        running_KLs_test.append(running_KL_test)
        running_DTAs_test.append(running_DTA_test)
        #print(100*correct/total)
        
    accs = 100 * np.array(corrects) / total
    running_CEs = np.array(running_CEs)
    running_KLs = np.array(running_KLs)
    running_DTAs = np.array(running_DTAs)
    running_CEs_test = np.array(running_CEs_test)
    running_KLs_test = np.array(running_KLs_test)
    running_DTAs_test = np.array(running_DTAs_test)
    
    np.save(path + "/acc", accs)
    np.save(path + "/train_ce_loss", running_CEs)
    np.save(path + "/train_kl_loss", running_KLs)
    np.save(path + "/train_dta_loss", running_DTAs)
    np.save(path + "/test_ce_loss", running_CEs_test)
    np.save(path + "/test_kl_loss", running_KLs_test)
    np.save(path + "/test_dta_loss", running_DTAs_test)
    torch.save(net.state_dict(), path + '/fe_weight.pth')
    torch.save(vib.state_dict(), path + '/vib_weight.pth')
    if not is_deterministic:
        np.save(path + "/sigma_bias", vib.sigma_bias)
    ff = open(path+'/settings.txt', 'w')
    ff.write(f'epoch:{n_epochs}\nbeta:{beta}')
    ff.close()
    

    

def fvib_ib_curve(betas, path, loader, net, n_samples=12, conf_after_ts=0.9995, device=device):
    K = torch.load(path + '/K.pt')
    FVIBs = [FVIB(K, beta, n_samples=n_samples, conf_after_ts=conf_after_ts) for beta in betas]
    KL = KLLoss()
    CE = nn.CrossEntropyLoss()
    DTA = DistortionTaylorApproxLoss(K.shape[0]+1)
    CEs, KLs, DTAs = [0.0 for _ in betas], [0.0 for _ in betas], [0.0 for _ in betas]
    correct = [0 for _ in betas]
    total = 0
    net.eval()
    with torch.no_grad():
        for data in loader:
            #images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            total += labels.size(0)
            for i, beta in enumerate(betas):
                out, mean, var, y_hat = FVIBs[i](outputs)
                CEs[i] += CE(out, labels).item()/len(loader)
                KLs[i] += KL(mean, var).item()/len(loader)
                W = (1-FVIBs[i].beta)**(1/2)*FVIBs[i].L.T
                DTAs[i] += DTA(W, torch.zeros(1, device=device), mean, var, labels).item()/len(loader)
                _, predicted = torch.max(y_hat.data, 1)
                correct[i] += (predicted == labels).sum().item()
    CEs = np.array(CEs)
    KLs = np.array(KLs)
    DTAs = np.array(DTAs)
    correct = np.array(correct)
    acc = 100*correct/total
    return CEs, KLs, DTAs, acc


def vib_calc_loss(d, loader, net, vib, n_samples=12, conf_after_ts=None, device=device):
    if conf_after_ts == None:
        temperature  =  1
    else:
        temperature  =  (d/ (np.log(d-1)+np.log(conf_after_ts/(1-conf_after_ts)))).item()
    KL = KLLoss()
    CE = nn.CrossEntropyLoss()
#     DTA = DistortionTaylorApproxLoss(K.shape[0]+1)
    CE_value, KL_value= 0.0, 0.0 #, DTA_value = 0.0, 0.0, 0.0
    net.eval()
    vib.eval()
    with torch.no_grad():
        for data in loader:
            #images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            out, mean, var, y_hat = vib(net(images))
            out = out/temperature
            CE_value += CE(out, labels).item()/len(loader)
            KL_value += KL(mean, var).item()/len(loader)
#             DTA_value += DTA(FVIBs[i].W, torch.zeros(1, device=device), mean, var, labels).item()/len(loader)
    return CE_value, KL_value