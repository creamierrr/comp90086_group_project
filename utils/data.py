from environment import *

# prepare image for the model
def prepare_image(filepath, resize_shape = 0, normalize = 0):
    img = cv2.imread(filepath)
    if resize_shape:
        
        # padding if the resize shape is greater than the original image shape
        if resize_shape > img.shape[0] or resize_shape > img.shape[1]:
            # Get the center coordinates of the image
            height, width, _ = img.shape

            pad_width = max(0, resize_shape - width)
            pad_height = max(0, resize_shape - height)

            # Calculate padding for top, bottom, left, and right
            top = pad_height // 2
            bottom = pad_height - top
            left = pad_width // 2
            right = pad_width - left

            # Add zero-padding to the image
            padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img = padded_img

        # Get the center coordinates of the image
        height, width, _ = img.shape
        center_x, center_y = width // 2, height // 2

        # Calculate crop boundaries
        left = center_x - resize_shape // 2
        top = center_y - resize_shape // 2
        right = center_x + resize_shape // 2
        bottom = center_y + resize_shape // 2

        # Perform the center crop
        center_cropped_img = img[top:bottom, left:right]
        img = center_cropped_img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype("float32")

    if normalize:
        img /= 255.0  # Normalize pixel values to [0, 1]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std  # Apply mean and standard deviation normalization
    
    else:
        img /= 255.0

    return img.astype("float32")

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

    np.random.seed(seed)

    if most_similar_hard_negatives is not None: # first round or not using semihard


        for i in range(len(train_list)):
            # do left
            ranked_negative_tuples = most_similar_hard_negatives[left[i]][:num_false - CFG.num_random_sample_false]
            ranked_negative = [negative_tuple[0] for negative_tuple in ranked_negative_tuples]

            for j in range(len(ranked_negative)):
                x_list.append((left[i], right[i], ranked_negative[j]))
            
            to_be_sampled = [neg_tuple[0] for neg_tuple in most_similar_hard_negatives[left[i]][num_false - CFG.num_random_sample_false:]] # will only look at remaining images which are all worse than anchor-positive (all satisfy semihard condition)
        
            sampled_neg = np.random.choice(to_be_sampled, CFG.num_random_sample_false, replace = False)

            for j in range(len(sampled_neg)):

                x_list.append((left[i], right[i], sampled_neg[j]))

            # do right
            ranked_negative_tuples = most_similar_hard_negatives[right[i]][:num_false - CFG.num_random_sample_false]
            ranked_negative = [negative_tuple[0] for negative_tuple in ranked_negative_tuples]

            for j in range(len(ranked_negative)):
                x_list.append((right[i], left[i], ranked_negative[j]))
            
            to_be_sampled = [neg_tuple[0] for neg_tuple in most_similar_hard_negatives[right[i]][num_false - CFG.num_random_sample_false:]] # will only look at remaining images which are all worse than anchor-positive (all satisfy semihard condition)
        
            sampled_neg = np.random.choice(to_be_sampled, CFG.num_random_sample_false, replace = False)

            for j in range(len(sampled_neg)):

                x_list.append((right[i], left[i], sampled_neg[j]))


    else:

        for i in range(len(train_list)):

            # do left
            to_be_sampled = [left[j] for j in range(len(left)) if j != i] + [right[j] for j in range(len(right)) if j != i] # all images except the anchor and the positive
        
            sampled_neg = np.random.choice(to_be_sampled, num_false, replace = False)

            for j in range(num_false):

                x_list.append((left[i], right[i], sampled_neg[j]))
        
            # do right
            sampled_neg = np.random.choice(to_be_sampled, num_false, replace = False)

            for j in range(num_false):

                x_list.append((right[i], left[i], sampled_neg[j]))

    return x_list

def DataLoader_Categorisation(x_list, y_list, batch_number, batch_size, CFG):
    """ Load in categorisation data """
    
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
    """ Load in Triplets """
    
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
    """ Helper that randomly samples negatives to make the train val test dataframe look like the future dataframe """
    
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
