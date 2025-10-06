from __future__ import print_function
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PreResNet import *
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pprint import pprint
from bert_mlp import BertMLP
import dataloader_tuned as dataloader

# ===== [A] START TIMER =====
start_wall = time.time()

parser = argparse.ArgumentParser(description='PyTorch General Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--image_size', default=28, type=int, help='image size for resize/crop')
parser.add_argument('--warm_up', default=10, type=int, help='number of warmup epochs')
# Dataset paths and columns
parser.add_argument('--dataset', default='fashion-mnist', type=str)
parser.add_argument('--noise_type', default='llm', type=str)
parser.add_argument('--data_type', default='image', type=str)
parser.add_argument('--train_csv_path', type=str, required=True)
parser.add_argument('--train_feather_path', type=str, required=True)
parser.add_argument('--train_data_column', type=str, required=True)
parser.add_argument('--train_label_column', type=str, required=True)
parser.add_argument('--train_image_dir', type=str)
parser.add_argument('--test_csv_path', type=str, required=True)
parser.add_argument('--test_data_column', type=str, required=True)
parser.add_argument('--test_label_column', type=str, required=True)
parser.add_argument('--test_image_dir', type=str)
parser.add_argument('--num_workers', default=4, type=int)
# ---- TEXT ----
parser.add_argument('--pretrained_name', type=str, default='bert-base-uncased')
parser.add_argument('--freeze_backbone', type=int, default=1, help='1=freeze BERT backbone, 0=finetune')
parser.add_argument('--max_length', type=int, default=256)
args = parser.parse_args()
args.pretrained_lm = getattr(args, 'pretrained_name', 'bert-base-uncased')

pprint(vars(args))

# Conditional requirements for image data
if args.data_type.lower() == 'image':
    if not args.train_image_dir:
        parser.error("--train_image_dir is required when --data_type=image")
    if not args.test_image_dir:
        parser.error("--test_image_dir is required when --data_type=image")

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def move_to_device(x):
    # Hỗ trợ dict từ tokenizer, list/tuple lồng nhau
    if isinstance(x, dict):
        return {k: v.cuda(non_blocking=True) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(xx) for xx in x)
    return x.cuda(non_blocking=True)

def train_text(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, scaler):
    net.train()
    net2.eval()
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size) + 1
    criterion = SemiLoss()

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)

        bs = labels_x.size(0)
        labels_x = torch.zeros(bs, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        w_x = w_x.view(-1,1).float()

        inputs_x  = move_to_device(inputs_x)
        inputs_x2 = move_to_device(inputs_x2)
        inputs_u  = move_to_device(inputs_u)
        inputs_u2 = move_to_device(inputs_u2)
        labels_x = labels_x.cuda(non_blocking=True)
        w_x = w_x.cuda(non_blocking=True)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=amp_dtype):
            x1 = net(inputs_x)
            x2 = net(inputs_x2)

            u11g = net(inputs_u)
            u12g = net(inputs_u2)

            with torch.no_grad():
                u21 = net2(inputs_u)
                u22 = net2(inputs_u2)
                pu = (torch.softmax(u11g,1) + torch.softmax(u12g,1)
                      + torch.softmax(u21,1) + torch.softmax(u22,1)) / 4
                ptu = pu ** (1/args.T)
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)

                px = (torch.softmax(x1,1) + torch.softmax(x2,1)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1/args.T)
                targets_x = (ptx / ptx.sum(dim=1, keepdim=True)).detach()

            outputs_x = (x1 + x2) / 2
            outputs_u = (u11g + u12g) / 2

            Lx, Lu, lamb = criterion(
                outputs_x, targets_x,
                outputs_u, targets_u,
                epoch + batch_idx/num_iter, args.warm_up
            )

            # Regularization giống nhánh image
            logits_all = torch.cat([outputs_x, outputs_u], dim=0)
            prior = torch.ones(args.num_class, device=logits_all.device) / args.num_class
            pred_mean = torch.softmax(logits_all, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + lamb * Lu + penalty

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, scaler):
    if args.data_type.lower() == 'text':
        return train_text(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, scaler)
    
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    # for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(tqdm(labeled_trainloader, desc=f"Train Epoch {epoch}")):
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with autocast(device_type="cuda", dtype=amp_dtype):
                # label co-guessing of unlabeled samples
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                outputs_u21 = net2(inputs_u)
                outputs_u22 = net2(inputs_u2)    
                # label refinement of labeled samples
                outputs_x = net(inputs_x)
                outputs_x2 = net(inputs_x2)        
                
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
              
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=amp_dtype):
            logits = net(mixed_input)
            logits_x = logits[:batch_size*2]
            logits_u = logits[batch_size*2:]

            Lx, Lu, lamb = criterion(
                logits_x, mixed_target[:batch_size*2],
                logits_u, mixed_target[batch_size*2:],
                epoch+batch_idx/num_iter, args.warm_up
            )

            # regularization
            prior = torch.ones(args.num_class, device=logits.device)/args.num_class
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            loss = Lx + lamb * Lu + penalty

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # print(f"{epoch=}, {batch_idx=}, Lx={Lx.item():.2f}, Lu={Lu.item():.2f}")

def warmup(epoch, net, optimizer, dataloader, scaler):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    # for batch_idx, (inputs, labels, path) in enumerate(tqdm(dataloader, desc=f"Warmup Epoch {epoch}")):
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs = move_to_device(inputs)
        labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype):
            outputs = net(inputs)
            loss = CEloss(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # print(f"Warmup {epoch=}, {batch_idx=}, loss={loss.item():.4f}")

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc=f"Test Epoch {epoch}")):
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = move_to_device(inputs)
            targets = targets.cuda(non_blocking=True)
            with autocast(device_type="cuda", dtype=amp_dtype):
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)           
                outputs = outputs1+outputs2
        
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))

def predict_testset(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # for inputs, targets in tqdm(test_loader, desc="Predict Testset"):
        for inputs, targets in test_loader:
            inputs = move_to_device(inputs)
            targets = targets.cuda(non_blocking=True)
            with autocast(device_type="cuda", dtype=amp_dtype):
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2

            predicted = outputs.argmax(dim=1)
            all_preds.append(predicted.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    if all_targets:
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"Accuracy: {(acc * 100):.2f}%")
        print(f"F1-macro: {(f1_macro * 100):.2f}%")
        print(f"F1-weighted: {(f1_weighted * 100):.2f}%")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

        return y_pred
    else:
        print("Accuracy: 0.00%\nF1-macro: 0.00%\nF1-weighted: 0.00%\n\nClassification report:\n(none)")
        return np.array([], dtype=np.int64)


def eval_train(model,all_loss):    
    model.eval()
    # Generalize losses size to match dataset
    dataset_size = len(eval_loader.dataset)
    losses = torch.zeros(dataset_size)
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # for batch_idx, (inputs, targets, index) in enumerate(tqdm(eval_loader, desc="Eval Train")):
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs = move_to_device(inputs)
            targets = targets.cuda(non_blocking=True)
            with autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(inputs) 
                loss = CE(outputs, targets)  
            
            bs = targets.size(0)
            for b in range(bs):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
    #     history = torch.stack(all_loss)
    #     input_loss = history[-5:].mean(0)
    #     input_loss = input_loss.reshape(-1,1)
    # else:
    #     input_loss = losses.reshape(-1,1)
    input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    if args.data_type.lower() == 'image':
        model = ResNet18(num_classes=args.num_class)
    else:
        model = BertMLP(
            pretrained_name=args.pretrained_name,
            num_classes=args.num_class,
            freeze_backbone=bool(args.freeze_backbone)
        )
    model = model.cuda()
    return model

if args.data_type.lower() == 'image':
    loader = dataloader.dataloader_tuned(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train_csv_path=args.train_csv_path,
        train_feather_path=args.train_feather_path,
        train_data_column=args.train_data_column,
        train_label_column=args.train_label_column,
        train_image_dir=args.train_image_dir,
        test_csv_path=args.test_csv_path,
        test_data_column=args.test_data_column,
        test_label_column=args.test_label_column,
        test_image_dir=args.test_image_dir
    )
else:
    import dataloader_tuned_text as text_loader
    loader = text_loader.Loader(args)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True


criterion = SemiLoss()
criterion = SemiLoss()
if args.data_type.lower() == 'image':
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
    params1 = [p for p in net1.parameters() if p.requires_grad]
    params2 = [p for p in net2.parameters() if p.requires_grad]
    optimizer1 = optim.AdamW(params1, lr=2e-5, weight_decay=0.01)
    optimizer2 = optim.AdamW(params2, lr=2e-5, weight_decay=0.01)

scaler1 = GradScaler(device="cuda")
scaler2 = GradScaler(device="cuda")

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

all_loss = [[],[]]

for epoch in tqdm(range(args.num_epochs+1), desc="Epochs"):
    lr=args.lr
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    if epoch < args.warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader, scaler1)
        print('Warmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader, scaler2)
    else:
        prob1, all_loss[0] = eval_train(net1, all_loss[0])
        prob2, all_loss[1] = eval_train(net2, all_loss[1])
        pred1 = (prob1 > args.p_threshold)
        pred2 = (prob2 > args.p_threshold)
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, scaler1)
        print('Train Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, scaler2)
    test(epoch, net1, net2)
    # Save test predictions at last epoch
    if epoch == args.num_epochs:
        preds = predict_testset(net1, net2, test_loader)
        out_name = f"{args.dataset}_{args.noise_type}_test-predictions.npy"
        np.save(out_name, preds)

# ===== [B] AFTER TRAIN =====
end_wall = time.time()
wall_sec = end_wall - start_wall
print(f"[TIME] Total wall time: {wall_sec:.2f}s ({wall_sec/3600:.4f}h)")
