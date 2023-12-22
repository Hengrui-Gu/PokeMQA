import numpy as np
import torch
import torch.optim as optim
import transformers
from editDataset import EditSeqDataset
import json
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as functional

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

def construct_dataset_embeddingcls(no_list,input_dataset):
    
    input_edit = []
    input_questions = []
    input_no = []
    for i in range(len(input_dataset)):
        for question in input_dataset[i]["questions"]:
            input_edit.append(input_dataset[i]["edit"])
            input_questions.append(question)
            input_no.append(no_list[i])
    
    return input_edit,input_questions,input_no

def one_hot_matrix(labels):

    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    labels_reshaped = [[label] for label in labels]
    final_labels = encoder.fit_transform(labels_reshaped)
    return final_labels

def construct_seqinput_one_negative(edits,questions,tokenizer):
    labels = []
    input_dataset = []
    size = len(edits)
    for i in range(size):
        for subquestions in questions:
            input_dataset.append(edits[i] + tokenizer.sep_token + subquestions[i])
            labels.append(0)  # positive
            negative_idx = np.random.randint(0,size)
            while negative_idx == i:
                negative_idx = np.random.randint(0,size)
            input_dataset.append(edits[i] + tokenizer.sep_token + questions[np.random.randint(0,4)][negative_idx])
            labels.append(1)  # negative sample

    return input_dataset,labels

def construct_valset(edits,question,no,tokenizer):
    val_set = []
    labels = []
    for i in range(len(edits)):
        val_set.append(edits[i] + tokenizer.sep_token + question)
        if i==no:
            labels.append(0)
        else:
            labels.append(1)

    return val_set , labels

def train_disambiguator():
    
    model_name = "distilbert-base-cased"

    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    cache_dir = "dis-ckpt"

    with open('datasets/cls-filtered.json', 'r') as f:
        dataset = json.load(f)

    # pre-processing finetune dataset
    input_no = np.ones((len(dataset)))

    # 按比例划分
    dataset_train, dataset_test, no_train, no_test = train_test_split(dataset, input_no, test_size=0.2, random_state=42)
    # train: 5350 test:1338

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

    TrainSet = EditSeqDataset(dataset_train)
    ValSet = EditSeqDataset(dataset_test)

    # self.toknizer.sep_token 
    test_size = len(ValSet)

    dataloader = DataLoader(TrainSet,batch_size=256,shuffle=True)
    dataloader_val = DataLoader(ValSet,batch_size=1338,shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.to(device)

    epochs = 500
    log_interval = 10
    bestYDI = 0.0
    early_stop_epoch = 0

    for iter in range(epochs):
        epoch_loss = 0
        
        for data in dataloader:
            batch_edit , batch_questions = data

            optimizer.zero_grad()
            inputs , labels = construct_seqinput_one_negative(batch_edit,batch_questions,tokenizer)
            labels = one_hot_matrix(labels)
            labels = torch.from_numpy(labels).to(device)
            
            batch_input = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            train_loss = model(**batch_input,labels = labels).loss

            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()*len(batch_edit)*8
            
        print('epoch: {}  loss: {}'.format(iter,epoch_loss))

        if iter % log_interval==0:  # validation
            with torch.no_grad():
                sum_test_sample = 0
                recall_num = 0
                block_num = 0
         
                for data in dataloader_val:
                    edits , questions = data
                    edit_size = len(edits)

                    for group in questions:
                        candidate_idxs = np.random.choice(range(edit_size),size=100,replace=False)
                        for i in candidate_idxs:
                            sum_test_sample += 1
                            val_input ,val_labels = construct_valset(edits,group[i],i,tokenizer)
                            val_batch_input = tokenizer(val_input,padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                            val_labels = one_hot_matrix(val_labels)
                            val_labels = torch.from_numpy(val_labels).to(device)

                            test_logits = model(**val_batch_input,labels = val_labels).logits
                            test_probs = functional.softmax(test_logits,dim=-1)[:,0]
            
                            positive_prob = test_probs[i].item()
                            test_probs[i] = 0.0
                            if positive_prob > test_probs.max():
                                recall_num += 1
                            if test_probs.max() < 0.5:
                                block_num += 1

                recall_rate = (recall_num/sum_test_sample)
                block_rate = (block_num/sum_test_sample)

                YDIndex = recall_rate + block_rate
                print(f'validation - epoch:{iter}  YDIndex:{YDIndex}')

                # YDIndex serving as the indicator of early stopping
                if YDIndex > bestYDI:
                    early_stop_epoch = 0
                    bestYDI = YDIndex
                    model.save_pretrained("detector-checkpoint/"+cache_dir)
                    print(f'epoch:{iter} recall:{recall_rate} block:{block_rate} YDIndex:{YDIndex} saving_to:{cache_dir}')
                else:
                    early_stop_epoch += 1
            
                if early_stop_epoch >= 10:
                    print(f'early stopping !')
                    break
    
    del model
    return cache_dir
               
if __name__=='__main__':
    train_disambiguator()




