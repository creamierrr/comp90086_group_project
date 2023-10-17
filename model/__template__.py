from environment import *
from utils import *

class CNN_Categorisation_Model(object):
    class Model():
        def __init__(self, CFG):
            pass

    def __init__(self, CFG, name="Model"):
        super().__init__()
        self.CFG   = CFG
        self.name  = name
        self.model = self.Model(self.CFG) # initialise model

        torch.manual_seed(self.CFG.random_state) # set random seed
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.CFG.lr)
        self.criterion = self.CFG.loss
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CFG.device = self.device
        self.model.to(self.device)
        self.criterion.to(self.device)
        
    def __str__(self):
        return self.name
    
    def save(self, mark=''): # save model
        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(self.CFG.rootpath, f'state/{self}{mark}.pt'))
    
    def load(self, mark=''): # load model
        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(self.CFG.rootpath, f'state/{self}{mark}.pt'), map_location=self.device))
    
    def fit(self, train_list, val_list = None, batch_size=128, epochs=32, patience=8, scheduler=False, grad_clip=False, mark=''):
        self.model.train()
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=patience//2) if scheduler else None

        np.random.seed(self.CFG.random_state)
        seeds = [np.random.randint(1, 1000) for _ in range(epochs)] # get a seed for every round

        max_accu = -math.inf
        for epoch in range(epochs):

            if not patience:
                break
            
            x_list, y_list = self.CFG.DataFactory_Categorisation(train_list, self.CFG.num_false, self.CFG.random_state, target = self.CFG.target) # create the instance for this round

            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            n_batch = len(x_list)//batch_size + 1

            np.random.seed(seeds[epoch]) # shuffle the data (make sure that the two lists are shuffled in the same way)
            np.random.shuffle(x_list)
            np.random.seed(seeds[epoch])
            np.random.shuffle(y_list)

            for mini_batch_number in tqdm(range(n_batch)):
                
                x_left, x_right, y = self.CFG.DataLoader_Categorisation(x_list, y_list, mini_batch_number, batch_size, self.CFG) # load each mini batch

                if len(x_left) == 0: # last batch potential bug
                    continue

                x_left, x_right, y = x_left.to(self.device), x_right.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                pred, true = self.model(x_left, x_right), y # run through model to get prediction which also contains gradients

                predicted_labels = torch.argmax(pred, dim=1)
                true_labels = torch.argmax(true, dim=1)

                loss = self.criterion(pred, true) # find out the loss

                loss.backward() # calculate gradient 

                if grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step() # backpropagate
                epoch_loss += loss.detach().cpu().numpy()

                epoch_pred += predicted_labels.detach().cpu().tolist()
                epoch_true += true_labels.detach().cpu().tolist() 


            epoch_loss /= n_batch
            epoch_accuracy = accuracy_score(epoch_pred, epoch_true)
            print(f"Epoch: {epoch + 1:>3} - Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}")

            # do validation
            if val_list is not None:
                valid_loss, valid_accu = self.eval(val_list, batch_size=batch_size, return_loss=True)
                self.model.train()
                if valid_accu >= max_accu: # if validation improve, then save the model
                    max_accu = valid_accu
                    self.save(mark)
                else:
                    patience -= 1 # else contribute 1 to early stopping
                if scheduler:
                    scheduler.step(valid_loss) # adjust the scheduler
    
    
    def eval(self, val_list, batch_size=128, return_loss=False):
        self.model.eval()


        with torch.no_grad():
        

            x_list, y_list = self.CFG.DataFactory_Categorisation(val_list, self.CFG.num_false, self.CFG.random_state, target = self.CFG.target) # create the instance

            valid_loss = 0
            valid_pred, valid_true = [], []
            n_batch = len(x_list)//batch_size + 1

            for mini_batch_number in range(n_batch):
                
                x_left, x_right, y = self.CFG.DataLoader_Categorisation(x_list, y_list, mini_batch_number, batch_size, self.CFG) # load each mini batch

                if len(x_left) == 0:
                    continue

                x_left, x_right, y = x_left.to(self.device), x_right.to(self.device), y.to(self.device)
                pred, true = self.model(x_left, x_right), y

                loss = self.criterion(pred, true)

                valid_loss += loss.detach().cpu().numpy()

                predicted_labels = torch.argmax(pred, dim=1)
                true_labels = torch.argmax(true, dim=1)

                valid_pred += predicted_labels.detach().cpu().tolist()
                valid_true += true_labels.detach().cpu().tolist() 


            val_future_list = turn_val_into_future(val_list, self.CFG.random_state)
            
            out, real_val_accu = self.real_eval(val_future_list, self.CFG.real_eval_batch_size) # get top2 evaluation (i.e. real evaluation)

            valid_loss /= n_batch
            valid_accuracy = accuracy_score(valid_pred, valid_true)
            print(f"Validation Loss: {valid_loss:.3f}, Accuracy: {valid_accuracy:.3f}")

        if return_loss:
            return valid_loss, real_val_accu
        else:
            return valid_pred, valid_true
    

    def real_eval(self, future_list, row_batch_size):

        correct = 0
        total = 0

        with torch.no_grad():
            
            left_images = []
            right_images = []
            results = []
            for id, row in future_list.iterrows():
                sample = row.values
                left_images.extend([self.CFG.images[f'{sample[0]}.jpg'] for _ in range(20)])
                right_images.extend([self.CFG.images[f'{image}.jpg'] for image in sample[1:]])
                
                if (id+1) % row_batch_size == 0:
                    
                    left_images = torch.tensor(np.array(left_images)).to(self.device)
                    right_images = torch.tensor(np.array(right_images)).to(self.device)

                    scores = self.model(left_images, right_images)[:, 1]
                    scores = scores.cpu().numpy()
                    scores = scores.reshape(row_batch_size, int(len(right_images)/row_batch_size))
                    results.extend(scores)
                    second_largest_values = np.partition(scores, -2, axis=1)[:, -2]
                    correct += sum(scores[:, 0] >= second_largest_values)
                    total += len(second_largest_values)

                    left_images = []
                    right_images = []

            if (id+1) % row_batch_size != 0: # in case residual batch

                left_images = torch.tensor(np.array(left_images)).to(self.device)
                right_images = torch.tensor(np.array(right_images)).to(self.device)

                scores = self.model(left_images, right_images)[:, 1]
                scores = scores.cpu().numpy()
                scores = scores.reshape(((id+1)%row_batch_size), int(len(right_images)/((id+1)%row_batch_size)))
                results.extend(scores)
                second_largest_values = np.partition(scores, -2, axis=1)[:, -2]
                correct += sum(scores[:, 0] >= second_largest_values)
                total += len(second_largest_values)

        print('Nominal Correct:', correct/total)

        out = future_list[['left']] # create final dataframe

        results_df = pd.DataFrame(results, columns = [f'c{i}' for i in range(20)])
        results_df = results_df.apply(softmax, axis = 1) # apply softmax to get pseudo probs

        out = pd.concat([out, results_df], axis = 1)
        
        return out, correct/total


class CNN_Triplet_Model(object): # COMMENTS SAME AS CLASSIFICATION TEMPLATE, EXCEPT WHERE THERE ARE ADDITIONAL COMMENTS
    class Model():
        def __init__(self, CFG):
            pass

    def __init__(self, CFG, name="Model"):
        super().__init__()
        self.CFG   = CFG
        self.name  = name
        self.model = self.Model(self.CFG) 

        assert self.CFG.num_false >= self.CFG.num_random_sample_false, 'num_false must be greater than or equal to num_random_sample'

        torch.manual_seed(self.CFG.random_state)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.CFG.lr)
        self.criterion = self.CFG.loss
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CFG.device = self.device
        self.model.to(self.device)
        self.criterion.to(self.device)
        
    def __str__(self):
        return self.name
    
    def save(self, mark=''): 
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

        max_accu = -math.inf
        for epoch in range(epochs):

            if not patience:
                break

            if self.CFG.resample and epoch != 0:
                x_list = self.CFG.DataFactory_Triplet(train_list, self.CFG.num_false, self.CFG.random_state, most_similar_hard_negatives, self.CFG)
            else:
                x_list = self.CFG.DataFactory_Triplet(train_list, self.CFG.num_false, self.CFG.random_state)

            epoch_loss = 0
            n_batch = len(x_list)//batch_size + 1

            np.random.seed(seeds[epoch])
            np.random.shuffle(x_list)

            for mini_batch_number in tqdm(range(n_batch)):
                
                x_anchor, x_positive, x_negative = self.CFG.DataLoader_Triplet(x_list, mini_batch_number, batch_size, self.CFG)

                if len(x_anchor) == 0:
                    continue

                x_anchor, x_positive, x_negative = x_anchor.to(self.device), x_positive.to(self.device), x_negative.to(self.device)

                self.optimizer.zero_grad()
                anchor, positive, negative = self.model(x_anchor = x_anchor, x_positive = x_positive, x_negative = x_negative)

                loss = self.criterion(anchor, positive, negative)

                loss.backward()

                if grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
            
            epoch_loss /= n_batch
            print(f"Epoch: {epoch + 1:>3} - Loss: {epoch_loss:.3f}")

            # do validation
            if val_list is not None:
                valid_loss, valid_accu = self.eval(val_list, batch_size=batch_size, return_loss=True)
                self.model.train()
                if valid_accu >= max_accu:
                    max_accu = valid_accu
                    self.save(mark)
                else:
                    patience -= 1
                if scheduler:
                    scheduler.step(valid_loss)

            if self.CFG.resample: # re-rank the negative instances so each time we train on the hardest negatives
            
                most_similar_hard_negatives = self.re_rank(train_list, batch_size, CFG=self.CFG)
    
    
    def eval(self, val_list, batch_size=128, return_loss=False):
        self.model.eval()

        with torch.no_grad():

            x_list = self.CFG.DataFactory_Triplet(val_list, 19, self.CFG.random_state)

            valid_loss = 0
            n_batch = len(x_list)//batch_size + 1

            for mini_batch_number in range(n_batch):
                
                x_anchor, x_positive, x_negative = self.CFG.DataLoader_Triplet(x_list, mini_batch_number, batch_size, self.CFG)

                if len(x_anchor) == 0:
                    continue

                x_anchor, x_positive, x_negative = x_anchor.to(self.device), x_positive.to(self.device), x_negative.to(self.device)
                anchor, positive, negative = self.model(x_anchor = x_anchor, x_positive = x_positive, x_negative = x_negative)

                loss = self.criterion(anchor, positive, negative)

                valid_loss += loss.detach().cpu().numpy() 
            
            val_future_list = turn_val_into_future(val_list, self.CFG.random_state)

            out, real_val_accu = self.real_eval(val_future_list, self.CFG.real_eval_batch_size)

            valid_loss /= n_batch
            print(f"Validation Loss: {valid_loss:.3f}")

        if return_loss:
            return valid_loss, real_val_accu

    def real_eval(self, future_list, row_batch_size):

        correct = 0
        total = 0
    
    
        left_image = []
        right_images = []

        with torch.no_grad():

            results = []
            for id, row in future_list.iterrows():
                sample = row.values
                left_image.append(self.CFG.images[f'{sample[0]}.jpg'])
                right_images.extend([self.CFG.images[f'{image}.jpg'] for image in sample[1:]])

                if (id+1) % row_batch_size == 0:
                
                    left_image = torch.tensor(np.array(left_image)).to(self.device)
                    right_images = torch.tensor(np.array(right_images)).to(self.device)

                    left_embeddings = self.model(x_anchor = left_image)
                    right_embeddings = self.model(x_positive = right_images)

                    len_right = len(right_embeddings)

                    right_embeddings = right_embeddings.reshape(row_batch_size, int(len_right/row_batch_size), self.CFG.embed_dim)
                    scores = []
                    for i in range(len(left_embeddings)):
                        scores.extend([(1/(1e-5 + torch.sqrt(torch.sum(torch.pow(left_embeddings[i] - right_embed, 2))))).cpu().numpy() for right_embed in right_embeddings[i]])
                    
                    scores = np.array(scores)
                    scores = scores.reshape(row_batch_size, int(len_right/row_batch_size))
                    results.extend(scores)
                    second_largest_values = np.partition(scores, -2, axis=1)[:, -2]
                    correct += sum(scores[:, 0] >= second_largest_values)
                    total += len(second_largest_values)
                
                    left_image = []
                    right_images = []
                
            if (id+1) % row_batch_size != 0: 
            
                left_image = torch.tensor(np.array(left_image)).to(self.device)
                right_images = torch.tensor(np.array(right_images)).to(self.device)
                
                left_embeddings = self.model(x_anchor = left_image)
                right_embeddings = self.model(x_positive = right_images)
                len_right = len(right_embeddings)

                right_embeddings = right_embeddings.reshape(((id+1)%row_batch_size), int(len_right/((id+1)%row_batch_size)), self.CFG.embed_dim)
                scores = []
                for i in range(len(left_embeddings)):
                    scores.extend([(1/(1e-5 + torch.sqrt(torch.sum(torch.pow(left_embeddings[i] - right_embed, 2))))).cpu().numpy() for right_embed in right_embeddings[i]]) # take the inverse euclidean distance

                scores = np.array(scores)
                scores = scores.reshape(((id+1)%row_batch_size), int(len_right/((id+1)%row_batch_size)))
                results.extend(scores)
                second_largest_values = np.partition(scores, -2, axis=1)[:, -2]
                correct += sum(scores[:, 0] >= second_largest_values)
                total += len(second_largest_values)

            left_image = []
            right_images = []
        
        print('Nominal Correct:', correct/total)

        out = future_list[['left']]
        
        results_df = pd.DataFrame(results, columns = [f'c{i}' for i in range(20)])
        results_df = results_df.apply(softmax, axis = 1)

        out = pd.concat([out, results_df], axis = 1)

        return out, correct/total
    
    def re_rank(self, train_list, batch_size, CFG):

        anchor = list(train_list['left']) + list(train_list['right'])
        nonanchor = list(train_list['right']) + list(train_list['left'])


        embeddings_anchor = self.get_embeddings(anchor, batch_size, mode = 'anchor') # get the embeddings
        embeddings_nonanchor = self.get_embeddings(nonanchor, batch_size, mode = 'non-anchor')

        anchors = torch.stack(embeddings_anchor)
        nonanchors = torch.stack(embeddings_nonanchor)

        # calculate the inter-pair euclidean distance
        distances = list()

        for i in range(0, anchors.shape[0], batch_size):

            anchor_batch = anchors[i:i+batch_size, :].unsqueeze(0)

            squared_diff = (anchor_batch - nonanchors.unsqueeze(1))**2

            distance = squared_diff.sum(dim=-1).permute(1, 0)

            distances.extend(distance)

        distances = torch.stack(distances)

        # get the hard negatives
        most_similar_hard_negatives = {}

        sorted_indices_list = torch.argsort(distances, dim=1)

        for i in range(len(nonanchors)):
            sorted_indices = sorted_indices_list[i]

            switch = not self.CFG.semi_hard # if semihard, start with false and then becomes true after hitting semihard criteria
            similar_hard_negatives = []
            for j in range(len(sorted_indices)):
                nonanchor_name = nonanchor[sorted_indices[j]]
                if switch and nonanchor_name != anchor[i]: # seimihard sampling, and also avoid sampling anchor itself as nonannchor
                    similar_hard_negatives.append((nonanchor_name, distances[i, sorted_indices[j]])) 
                if not switch and nonanchor_name == nonanchor[i]: # ignore all negatives that are closer to anchor than the positive
                    switch = True
            most_similar_hard_negatives[anchor[i]] = similar_hard_negatives
        
        return most_similar_hard_negatives
            

    def get_embeddings(self, input_list, batch_size, mode):
        
        self.model.eval()

        with torch.no_grad():

            embeddings = []
            for i in range(len(input_list)//batch_size + 1):
                
                input_images = [self.CFG.images[f'{img}.jpg'] for img in input_list[i*batch_size:(i+1)*batch_size]]
                input_images = torch.tensor(np.array(input_images)).to(self.device)

                if len(input_images) == 0:
                    continue

                if mode == 'anchor': # put the model through the model to get embeddings (different modes in case of bi-encoder)
                    embedding = self.model(x_anchor = input_images)
                elif mode == 'non-anchor':
                    embedding = self.model(x_positive = input_images)
                
                embeddings.extend(embedding)
                
                
            return embeddings