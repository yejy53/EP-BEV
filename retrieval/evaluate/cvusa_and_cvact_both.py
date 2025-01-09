import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict


def evaluate(config,
             model_1,
             model_2,
             reference_dataloader1,
             reference_dataloader2,
             query_dataloader, 
             BEV_dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features1, reference_labels = predict(config, model_1, reference_dataloader1) 
    query_features, query_labels = predict(config, model_1, query_dataloader)
    reference_features2, reference_labels = predict(config, model_2, reference_dataloader2) 
    BEV_features, BEV_labels = predict(config, model_2, BEV_dataloader) 
    
    
    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features1, reference_features2, BEV_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
        
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features1, reference_features2, reference_labels, BEV_features, BEV_labels, reference_features2, query_features, query_labels
        gc.collect()
        
    return r1


def calc_sim(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader) 
    query_features, query_labels = predict(config, model, query_dataloader)
    
    print("Compute Scores Train:")
    r1 =  calculate_scores0(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, near_dict




def calculate_scores0(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        
    results = results/ Q * 100.
 
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
        
    print(' - '.join(string)) 

    return results[0]

def calculate_scores(query_features, reference_features1, reference_features2, BEV_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):
    
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features1)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    # Initialize similarity as an empty tensor with the same shape as the final similarity matrix
    similarity = torch.zeros(Q, R, device=query_features.device)  # Ensure similarity is on the same device
    
    # Compute similarity in steps
    for i in range(steps):
        
        start = step_size * i
        end = start + step_size
        
        # Calculate similarity for query features and reference features
        sim_tmp = query_features[start:end] @ reference_features1.T
        similarity[start:end] += sim_tmp  # No need to move sim_tmp to CPU
        
        # Optionally, also calculate similarity for BEV features and add to similarity matrix
        sim_tmp_BEV = BEV_features[start:end] @ reference_features2.T
        similarity[start:end] += sim_tmp_BEV  # No need to move sim_tmp_BEV to CPU
        # 

    # Append R//100 to topk for final comparison
    topk.append(R // 100)
    
    results = np.zeros([len(topk)])
    
    bar = tqdm(range(Q))
    
    for i in bar:
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        higher_sim = similarity[i,:] > gt_sim
        
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
    
    results = results / Q * 100.
    
    bar.close()
    
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
    
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
    
    print(' - '.join(string)) 

    return results[0]


    

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):


    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+1, dim=1)

    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

     
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        
        nearest = topk_references[i][mask[i]][:neighbour_range]
    
        nearest_dict[query_labels[i].item()] = list(nearest)
    

    return nearest_dict
