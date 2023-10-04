from environment import *

# prepare image for the model
def prepare_image(filepath, resize_shape = 0):
    img = cv2.imread(filepath)
    if resize_shape:
        img = cv2.resize(img, (resize_shape, resize_shape))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype("float32")
    img /= 255.0
    return img

def softmax(row):
    e_x = np.exp(row - np.max(row))  # Subtracting the max value for numerical stability
    return e_x / e_x.sum()

def DataFactory_Categorisation(train_list, num_false, seed, target = 1):
    """ 
        Input:
            Num_False: number of false samples for each true sample
            Target: target value for positive samples    
        Output:
            x_list: list of tuples (left, right)
            y_list: list of target values
    """
    
    left = list(train_list['left'])
    right = list(train_list['right'])

    x_list = [(left[i], right[i]) for i in range(len(train_list))]
    y_list = [[target, 1-target] for _ in range(len(train_list))]

    np.random.seed(seed)

    for i in range(len(train_list)):
        
        right_tmp = [right[j] for j in range(len(right)) if j != i]
    
        sampled_right = np.random.choice(right_tmp, num_false, replace = False)

        for j in range(num_false):

            x_list.append((left[i], sampled_right[j]))
            y_list.append([1-target, target])

    return x_list, y_list

def DataFactory_Triplet(train_list, num_false, seed, most_similar_hard_negatives = None, CFG = None):
    """ 
        Input:
            Num_False: number of false samples for each true sample
            Ranking: ranking of negative samples
            Target: target value for positive samples    
        Output:
            x_list: list of tuples (left, right)
    """
    
    left = list(train_list['left'])
    right = list(train_list['right'])

    x_list = []

    if most_similar_hard_negatives is not None:

        for i in range(len(train_list)):

            ranked_right_tuples = most_similar_hard_negatives[left[i]][:num_false - CFG.num_random_sample_false]
            ranked_right = [right_tuple[0] for right_tuple in ranked_right_tuples]

            for j in range(len(ranked_right)):
                x_list.append((left[i], right[i], ranked_right[j]))
            
            right_tmp = [right_tuple[0] for right_tuple in most_similar_hard_negatives[left[i]][num_false - CFG.num_random_sample_false:]]
        
            sampled_right = np.random.choice(right_tmp, CFG.num_random_sample_false, replace = False)

            for j in range(len(sampled_right)):

                x_list.append((left[i], right[i], sampled_right[j]))

    else:

        np.random.seed(seed)

        for i in range(len(train_list)):
            
            right_tmp = [right[j] for j in range(len(right)) if j != i]
        
            sampled_right = np.random.choice(right_tmp, num_false, replace = False)

            for j in range(num_false):

                x_list.append((left[i], right[i], sampled_right[j]))

    return x_list

def DataLoader_Categorisation(x_list, y_list, batch_number, batch_size, CFG):
    
    x_left, x_right, y = [], [], []

    x_list = x_list[batch_number*batch_size:(batch_number+1)*batch_size]
    y_list = y_list[batch_number*batch_size:(batch_number+1)*batch_size]

    for i in range(len(x_list)):
        x_left.append(CFG.images[f'{x_list[i][0]}.jpg'])
        x_right.append(CFG.images[f'{x_list[i][1]}.jpg'])
        y.append(y_list[i])

    x_left = np.array(x_left)
    x_right = np.array(x_right)
    y = np.array(y)

    return torch.FloatTensor(x_left), torch.FloatTensor(x_right), torch.FloatTensor(y)


def DataLoader_Triplet(x_list, batch_number, batch_size, CFG):
    
    x_anchor, x_positive, x_negative = [], [], []

    x_list = x_list[batch_number*batch_size:(batch_number+1)*batch_size]

    for i in range(len(x_list)):
        x_anchor.append(CFG.images[f'{x_list[i][0]}.jpg'])
        x_positive.append(CFG.images[f'{x_list[i][1]}.jpg'])
        x_negative.append(CFG.images[f'{x_list[i][2]}.jpg'])

    x_anchor = np.array(x_anchor)
    x_positive = np.array(x_positive)
    x_negative = np.array(x_negative)

    return torch.FloatTensor(x_anchor), torch.FloatTensor(x_positive), torch.FloatTensor(x_negative)\


def turn_val_into_future(val_list, seed):
    
    left = list(val_list['left'])
    right = list(val_list['right'])

    np.random.seed(seed)

    new_list = []
    for i in range(len(right)):
        new_right = [left[i], right[i]]
        right_tmp = [right[j] for j in range(len(right)) if j != i]
        new_right.extend(np.random.choice(right_tmp, 19, replace = False))
        new_list.append(new_right)

    return pd.DataFrame(new_list, columns = ['left'] + [f'c{i}' for i in range(19+1)])
