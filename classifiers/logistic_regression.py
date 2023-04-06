import torch
from torch import nn
import numpy as np
import copy

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)
            
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

    def train(params, K_train, y_train, n_epochs, Kval=None, y_val=None):
        """
        Assumes precomputed kernels.
        """
        batch_size = params["batch_size"]
        lambda1 = params['lambda1']
        best_model = None
        best_acc = 0
        acc_list = []
        loss_list_train = []
            
        # Tensors
        X_train = torch.FloatTensor(K_train)
        if Kval is not None:
            X_val = torch.FloatTensor(Kval)

        # Get weights (imbalance)
        w_neg = torch.sum(y_train == 0)
        w_pos = torch.sum(y_train == 1)
        
        # define model and optimizer
        n = K_train.shape[0]
        model = LogisticRegression(input_dim=n, output_dim=1)
        criterion = torch.nn.BCELoss(reduction='none') # gives vector output
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=params['lr'])
        # Training
        for epoch in range(n_epochs):
            batch_loss = []
            for b in range(n//batch_size):
                X_batch = X_train[b*batch_size:(b+1)*batch_size, :]
                y_batch = y_train[b*batch_size:(b+1)*batch_size].reshape(-1, 1)

                # define weights to correct imbalanceness      
                weights = torch.FloatTensor([n/(2*w_neg) if val==0. else n/(2*w_pos) for val in y_batch])

                y_pred = model(X_batch)

                loss = torch.mean(criterion(y_pred, y_batch)*weights)
                for param in model.linear.parameters():
                    loss += lambda1*torch.norm(param, 1)

                batch_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            loss_list_train.append(np.mean(batch_loss))

            if Kval is not None:
                with torch.no_grad():
                    y_valpred = model(X_val).round()
                    acc = np.mean(np.equal(y_val, y_valpred.numpy()[:, 0]))
                    acc_list.append(acc)
                    if acc > best_acc:
                        best_acc = acc
                        best_model = copy.deepcopy(model)

                        print(f"New best acc={best_acc}")
        if best_model is None:
            best_model = model
            
        return {"params": params, 
                "best_acc": best_acc, 
                "best_model": best_model, 
                "loss_train": loss_list_train, 
                "acc_val": acc_list}