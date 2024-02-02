import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from fvib import FVIB


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CalibrateFVIB(nn.Module):
    """this code is based on https://github.com/gpleiss/temperature_scaling"""
    def __init__(self, model, K, num_samples, conf_after_ts=0.997, beta_as_param =True, max_iter=50, lr=0.1):
        super(CalibrateFVIB, self).__init__()
        self.model = model
        self.max_iter=max_iter
        self.lr =lr
        if beta_as_param:
            self.beta = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.beta = 0.
        self.fvib = FVIB(K, self.beta, num_samples, conf_after_ts)
    def forward(self, input):
        _, _, _, confs = self.fvib(self.model(input))
        return confs
    def set_beta(self, valid_loader):
        """
        Tune beta of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.NLLLoss().cuda()
        ece_criterion = ECELoss().cuda()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            
        before_temperature_nll = nll_criterion(torch.log(self.fvib(logits)[3]), labels).item()
        before_temperature_ece = ece_criterion(self.fvib(logits)[3], labels, False)[0].item()
        print('Before optimize beta - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.beta], lr=self.lr, max_iter=self.max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(torch.log(self.fvib(logits)[3]), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll =nll_criterion(torch.log(self.fvib(logits)[3]), labels).item()
        after_temperature_ece = ece_criterion(self.fvib(logits)[3], labels, False)[0].item()
        print('Optimal beta: %.3f' % torch.clamp(self.beta, min=0., max=1.).item())
        print('After optimize beta - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        # return self


def calc_logits(model, loader):
    model.eval()
    all_logits, all_labels = torch.tensor([]).to(device), torch.tensor([]).to(device)
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            logits = model(images)
            all_logits = torch.concat([all_logits, logits])
            all_labels = torch.concat([all_labels, labels])
    return all_logits, all_labels


def split_valset(testset, valsize, batch_size = 100):
    n_samples = len(testset) # n_samples is 60000
    subset1_indices = list(range(0,valsize)) 
    subset2_indices = list(range(valsize,n_samples))

    valset = Subset(testset, subset1_indices)
    testset   = Subset(testset, subset2_indices)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=False)
    return val_loader, test_loader

    
class ECELoss(nn.Module):
    """
    this code is from https://github.com/gpleiss/temperature_scaling
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, apply_softmax=True):
        if apply_softmax:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        accs_for_bin, confs_for_bin = [], []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                accuracy_in_bin = accuracy_in_bin.item()
                avg_confidence_in_bin = avg_confidence_in_bin.item()
                
            else:
                accuracy_in_bin = None
                avg_confidence_in_bin =None
            accs_for_bin.append(accuracy_in_bin)
            confs_for_bin.append(avg_confidence_in_bin)
        return ece, (accs_for_bin, confs_for_bin)
