from environment import *

class CNN_Categorisation_Model(object):
    class Model():
        def __init__(self, CFG):
            pass

    def __init__(self, CFG, name="Model"):
        super().__init__()
        self.CFG   = CFG
        self.name  = name
        self.model = self.Model(self.CFG)

        torch.manual_seed(self.CFG.random_state)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.CFG.lr)
        self.criterion = self.CFG.loss
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CFG.device = self.device
        self.model.to(self.device)
        self.criterion.to(self.device)
        
    def __str__(self):
        return self.name
    
    def save(self, mark=''): # need to be changed
        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(self.CFG.rootpath, f'state/{self}{mark}.pt'))
    
    def load(self, mark=''):
        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(self.CFG.rootpath, f'state/{self}{mark}.pt'), map_location=self.device))
    
    def fit(self, train_list, val_list = None, batch_size=128, epochs=32, patience=8, scheduler=False, grad_clip=False, mark=''):
        self.model.train()
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=patience//2) if scheduler else None

        np.random.seed(self.CFG.random_state)
        seeds = [np.random.randint(1, 1000) for _ in range(epochs)]

        min_loss = math.inf
        for epoch in range(epochs):

            if not patience:
                break
            
            x_list, y_list = self.CFG.DataFactory_Categorisation(train_list, self.CFG.num_false, self.CFG.random_state, target = self.CFG.target)

            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            n_batch = len(x_list)//batch_size + 1

            np.random.seed(seeds[epoch])
            np.random.shuffle(x_list)
            np.random.seed(seeds[epoch])
            np.random.shuffle(y_list)

            for mini_batch_number in tqdm(range(n_batch)):
                
                x_left, x_right, y = self.CFG.DataLoader_Categorisation(x_list, y_list, mini_batch_number, batch_size, self.CFG.ROOT)

                x_left, x_right, y = x_left.to(self.device), x_right.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                pred, true = self.model(x_left, x_right), y

                predicted_labels = torch.argmax(pred, dim=1)
                true_labels = torch.argmax(true, dim=1)

                loss = self.criterion(pred, true)

                loss.backward()

                if grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()

                epoch_pred += predicted_labels.detach().cpu().tolist()
                epoch_true += true_labels.detach().cpu().tolist() 


            epoch_loss /= n_batch
            epoch_accuracy = accuracy_score(epoch_pred, epoch_true)
            print(f"Epoch: {epoch + 1:>3} - Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}")


            if val_list is not None:
                valid_loss = self.eval(val_list, batch_size=batch_size, return_loss=True)
                self.model.train()
                if valid_loss <= min_loss:
                    min_loss = valid_loss
                    self.save(mark)
                else:
                    patience -= 1
                if scheduler:
                    scheduler.step(valid_loss)
    
    
    def eval(self, val_list, batch_size=128, return_loss=False):
        self.model.eval()


        with torch.no_grad():
        

            x_list, y_list = self.CFG.DataFactory_Categorisation(val_list, self.CFG.num_false, self.CFG.random_state, target = self.CFG.target)

            valid_loss = 0
            valid_pred, valid_true = [], []
            n_batch = len(x_list)//batch_size + 1

            for mini_batch_number in range(n_batch):
                
                x_left, x_right, y = self.CFG.DataLoader_Categorisation(x_list, y_list, mini_batch_number, batch_size, self.CFG.ROOT)

                x_left, x_right, y = x_left.to(self.device), x_right.to(self.device), y.to(self.device)
                pred, true = self.model(x_left, x_right), y

                loss = self.criterion(pred, true)

                valid_loss += loss.detach().cpu().numpy()

                predicted_labels = torch.argmax(pred, dim=1)
                true_labels = torch.argmax(true, dim=1)

                valid_pred += predicted_labels.detach().cpu().tolist()
                valid_true += true_labels.detach().cpu().tolist() 


            valid_loss /= n_batch
            valid_accuracy = accuracy_score(valid_pred, valid_true)
            print(f"Validation Loss: {valid_loss:.3f}, Accuracy: {valid_accuracy:.3f}")

        if return_loss:
            return valid_loss
        else:
            return valid_pred, valid_true

from environment import *

class CNN_Triplet_Model(object):
    class Model():
        def __init__(self, CFG):
            pass

    def __init__(self, CFG, name="Model"):
        super().__init__()
        self.CFG   = CFG
        self.name  = name
        self.model = self.Model(self.CFG)

        torch.manual_seed(self.CFG.random_state)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.CFG.lr)
        self.criterion = self.CFG.loss
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CFG.device = self.device
        self.model.to(self.device)
        self.criterion.to(self.device)
        
    def __str__(self):
        return self.name
    
    def save(self, mark=''): # need to be changed
        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(self.CFG.rootpath, f'state/{self}{mark}.pt'))
    
    def load(self, mark=''):
        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(self.CFG.rootpath, f'state/{self}{mark}.pt'), map_location=self.device))
    
    def fit(self, train_list, val_list = None, batch_size=128, epochs=32, patience=8, scheduler=False, grad_clip=False, mark=''):
        self.model.train()
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=patience//2) if scheduler else None

        np.random.seed(self.CFG.random_state)
        seeds = [np.random.randint(1, 1000) for _ in range(epochs)]

        min_loss = math.inf
        for epoch in range(epochs):

            if not patience:
                break
            
            x_list = self.CFG.DataFactory_Triplet(train_list, self.CFG.num_false, self.CFG.random_state, target = self.CFG.target)

            epoch_loss = 0
            n_batch = len(x_list)//batch_size + 1

            np.random.seed(seeds[epoch])
            np.random.shuffle(x_list)

            for mini_batch_number in tqdm(range(n_batch)):
                
                x_anchor, x_positive, x_negative = self.CFG.DataLoader_Triplet(x_list, mini_batch_number, batch_size, self.CFG.ROOT)

                x_anchor, x_positive, x_negative = x_anchor.to(self.device), x_positive.to(self.device), x_negative.to(self.device)

                self.optimizer.zero_grad()
                anchor, positive, negative = self.model(x_anchor, x_positive, x_negative)

                loss = self.criterion(anchor, positive, negative)

                loss.backward()

                if grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()

            epoch_loss /= n_batch
            print(f"Epoch: {epoch + 1:>3} - Loss: {epoch_loss:.3f}")


            if val_list is not None:
                valid_loss = self.eval(val_list, batch_size=batch_size, return_loss=True)
                self.model.train()
                if valid_loss <= min_loss:
                    min_loss = valid_loss
                    self.save(mark)
                else:
                    patience -= 1
                if scheduler:
                    scheduler.step(valid_loss)
    
    
    def eval(self, val_list, batch_size=128, return_loss=False):
        self.model.eval()


        with torch.no_grad():
        

            x_list = self.CFG.DataFactory_Triplet(val_list, self.CFG.num_false, self.CFG.random_state, target = self.CFG.target)

            valid_loss = 0
            n_batch = len(x_list)//batch_size + 1

            for mini_batch_number in range(n_batch):
                
                x_anchor, x_positive, x_negative = self.CFG.DataLoader_Triplet(x_list, mini_batch_number, batch_size, self.CFG.ROOT)

                x_anchor, x_positive, x_negative = x_anchor.to(self.device), x_positive.to(self.device), x_negative.to(self.device)
                anchor, positive, negative = self.model(x_anchor, x_positive, x_negative)

                loss = self.criterion(anchor, positive, negative)

                valid_loss += loss.detach().cpu().numpy() 


            valid_loss /= n_batch
            print(f"Validation Loss: {valid_loss:.3f}")

        if return_loss:
            return valid_loss
