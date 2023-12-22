import numpy as np
import torch
import torch.optim as optim
import transformers
from editDataset import EditDataset
import json
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def batch_negative_bce_loss(edits,questions,nos):
    
    batch_size = len(edits)
    nos = np.array(nos)
    # num of negative samples
    negative_number = 20

    positive_dist = (edits-questions).norm(2,-1)
    positive_dist = -positive_dist**2          # logp

    negative_dist = torch.ones((batch_size)).to(device)
    for i in range(batch_size):
        negative_idx = np.random.randint(0, batch_size, size=negative_number)
        while nos[i] in nos[negative_idx]:
            negative_idx = np.random.randint(0, batch_size, size=negative_number)
        negative_dist_list = -(questions[negative_idx]-edits[i]).norm(2,-1)**2
        negative_likelihood = torch.log(1-negative_dist_list.exp())    # log(1-p)
        negative_dist[i] = negative_likelihood.mean()

    cls_loss = positive_dist + negative_dist

    return -cls_loss.mean()

def cal_accuracy(edits,questions):
    batch_size = len(edits)

    positive_dist = (edits-questions).norm(2,-1)
    positive_dist = -positive_dist**2          # logp
    prob = positive_dist.exp()
    positive_num = (prob>=0.5).sum()

    return positive_num 

def retrieval_metric(edits,questions,nos):
    
    # recall_rate  \  block_rate
    instance_num = len(questions)
    nos = np.array(nos)
    retrieval_num = 0
    block_num = 0

    for i in range(instance_num):
        idxs = nos != nos[i]
        dist = (edits-questions[i]).norm(2,-1)

        log_prob = -dist**2
        prob = log_prob.exp()
        if prob[i] > prob[idxs].max():
            retrieval_num += 1

        if prob[idxs].max()<0.5:
            block_num += 1
    
    return retrieval_num, block_num

def construct_dataset(no_list,input_dataset):
    
    input_edit = []
    input_questions = []
    input_no = []
    for i in range(len(input_dataset)):
        for question in input_dataset[i]["questions"]:
            input_edit.append(input_dataset[i]["edit"])
            input_questions.append(question)
            input_no.append(no_list[i])
        
    return input_edit,input_questions,input_no

def train_detector():
    
    model_name = "distilbert-base-cased"

    model = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    cache_dir = "detector-ckpt"
    
    with open('datasets/cls-filtered.json', 'r') as f:
        dataset = json.load(f)


    # pre-processing finetune dataset
    input_no = range(1,len(dataset)+1)

    dataset_train, dataset_test, no_train, no_test = train_test_split(dataset, input_no, test_size=0.2, random_state=42)
    # train: 5350 test:1338
    
    edit_train,questions_train,no_train = construct_dataset(no_train,dataset_train)
    edit_test,questions_test,no_test = construct_dataset(no_test,dataset_test)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    TrainSet = EditDataset(edit_train,questions_train,no_train)
    ValSet = EditDataset(edit_test,questions_test,no_test)

    test_size = len(ValSet)

    dataloader = DataLoader(TrainSet,batch_size=1024,shuffle=True)
    dataloader_val = DataLoader(ValSet,batch_size=1338,shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.to(device)

    epochs = 1000
    log_interval = 20
    bestYDI = 0.0
    early_stop_epoch = 0

    print("Scope detector training stage 1 : pre-detector")
    for iter in range(epochs):
        epoch_loss = 0

        for data in dataloader:
            edits , questions ,nos = data
            edits_input = tokenizer(edits, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            questions_input = tokenizer(questions, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)

            optimizer.zero_grad()

            edits_output = model(**edits_input).last_hidden_state[:,0]
            questions_output = model(**questions_input).last_hidden_state[:,0]

            bce_loss = batch_negative_bce_loss(edits_output,questions_output,nos)
            bce_loss.backward()
            epoch_loss += bce_loss.item()*len(edits)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        print('epoch: {}  loss: {}'.format(iter,epoch_loss))

        if iter % log_interval==0:  # validation
            with torch.no_grad():
                val_items = 0
                recall_items = 0
                block_items = 0
                
                for data in dataloader_val:
                    edits , questions, nos = data

                    edits_input = tokenizer(edits, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
                    questions_input = tokenizer(questions, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)

                    edits_output = model(**edits_input).last_hidden_state[:,0]
                    questions_output = model(**questions_input).last_hidden_state[:,0]

                    acc_nums  = cal_accuracy(edits_output,questions_output)
                    recall_nums,block_nums = retrieval_metric(edits_output,questions_output,nos)

                    val_items += acc_nums
                    recall_items += recall_nums
                    block_items += block_nums
                
                val_accuracy = val_items.item()/test_size
                recall_rate = recall_items/test_size
                block_rate = block_items/test_size

                YDIndex = recall_rate + block_rate
                print(f'validation - epoch:{iter}  YDIndex:{YDIndex}')

                # YDIndex serving as the indicator of early stopping
                if YDIndex > bestYDI:
                    early_stop_epoch = 0
                    bestYDI = YDIndex
                    model.save_pretrained("detector-checkpoint/"+cache_dir)
                    print(f'epoch:{iter} acc:{val_accuracy} recall_r:{recall_rate} block_r:{block_rate} YDI:{YDIndex} saving_to:{cache_dir}')
                else:
                    early_stop_epoch += 1
            
                if early_stop_epoch>=5:
                    print(f'early stopping !')
                    break
    
    del model
    return cache_dir

if __name__=='__main__':
    train_detector()




