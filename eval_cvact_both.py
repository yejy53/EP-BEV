import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from congeo.dataset.cvact import CVACTDatasetEval
from congeo.transforms import get_transforms_val, get_transforms_val
from congeo.evaluate.cvusa_and_cvact_both import evaluate
from congeo.model import TimmModel

@dataclass
class Configuration:
    model_1: str = 'convnext_base.fb_in22k_ft_in1k_384'  # First model
    model_2: str = 'convnext_base.fb_in22k_ft_in1k_384'  # Second model (as an example)
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    
    # Dataset
    data_folder_1 = ""  # Dataset for first model(Same),CVUSA or CVACT
    data_folder_2 = ""  # Dataset for second model(Same),CVUSA or CVACT
    
    
    checkpoint_start_1 = ''# Checkpoint for SVI； Such：weights_e36_90.8149.pth from “https://huggingface.co/Yejy53/CVACT-Street/tree/main”
     
    checkpoint_start_2 = '' # Checkpoint for BEV； Such：weights_e40_89.0140.pth from “https://huggingface.co/Yejy53/EB-BEV-CVACT/tree/main”
  
    # Set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # Train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 



# Config
config = Configuration() 

if __name__ == '__main__':
    
    #-----------------------------------------------------------------------------#
    # Model 1
    #-----------------------------------------------------------------------------#
    print("\nModel 1: {}".format(config.model_1))
    model_1 = TimmModel_ConGeo(config.model_1,
                               pretrained=True,
                               img_size=config.img_size)
    
    data_config_1 = model_1.get_config()
    mean_1 = data_config_1["mean"]
    std_1 = data_config_1["std"]
    
    # Load pretrained Checkpoint for Model 1    
    if config.checkpoint_start_1 is not None:  
        print("Start from:", config.checkpoint_start_1)
        model_state_dict_1 = torch.load(config.checkpoint_start_1)  
        model_1.load_state_dict(model_state_dict_1, strict=False)     
    
    model_1 = model_1.to(config.device)

    #-----------------------------------------------------------------------------#
    # Model 2
    #-----------------------------------------------------------------------------#
    print("\nModel 2: {}".format(config.model_2))
    model_2 = TimmModel_ConGeo(config.model_2,
                               pretrained=True,
                               img_size=config.img_size)
    
    data_config_2 = model_2.get_config()
    mean_2 = data_config_2["mean"]
    std_2 = data_config_2["std"]
    
    # Load pretrained Checkpoint for Model 2    
    if config.checkpoint_start_2 is not None:  
        print("Start from:", config.checkpoint_start_2)
        model_state_dict_2 = torch.load(config.checkpoint_start_2)  
        model_2.load_state_dict(model_state_dict_2, strict=False)     
    
    model_2 = model_2.to(config.device)

    #-----------------------------------------------------------------------------#
    # DataLoader 1
    #-----------------------------------------------------------------------------#
    sat_transforms_val_1, ground_transforms_val_1 = get_transforms_val(
        image_size_sat=(config.img_size, config.img_size),
        img_size_ground=(384,768),
        mean=mean_1,
        std=std_1,
    )

    reference_dataset_test_1 = CVACTDatasetEval(data_folder=config.data_folder_1,
                                                split="val",
                                                img_type="reference",
                                                transforms=sat_transforms_val_1)

    reference_dataloader_test_1 = DataLoader(reference_dataset_test_1,
                                             batch_size=config.batch_size,
                                             num_workers=config.num_workers,
                                             shuffle=False,
                                             pin_memory=True)

    query_dataset_test_1 = CVACTDatasetEval(data_folder=config.data_folder_1,
                                            split="val",
                                            img_type="query",
                                            transforms=ground_transforms_val_1)

    query_dataloader_test_1 = DataLoader(query_dataset_test_1,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)

    #-----------------------------------------------------------------------------#
    # DataLoader 2
    #-----------------------------------------------------------------------------#
    sat_transforms_val_2, ground_transforms_val_2 = get_transforms_val(
        image_size_sat=(config.img_size, config.img_size),
        img_size_ground=(384,384),
        mean=mean_2,
        std=std_2,
    )

    reference_dataset_test_2 = CVACTDatasetEval(data_folder=config.data_folder_2,
                                                split="val",
                                                img_type="reference",
                                                transforms=sat_transforms_val_2)

    reference_dataloader_test_2 = DataLoader(reference_dataset_test_2,
                                             batch_size=config.batch_size,
                                             num_workers=config.num_workers,
                                             shuffle=False,
                                             pin_memory=True)

    query_dataset_test_2 = CVACTDatasetEval(data_folder=config.data_folder_2,
                                            split="val",
                                            img_type="query_BEV",
                                            transforms=sat_transforms_val_2)

    query_dataloader_test_2 = DataLoader(query_dataset_test_2,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)


    print("Reference Images Test 1:", len(reference_dataset_test_1))
    print("Query Images Test 1:", len(query_dataset_test_1))
    print("Reference Images Test 2:", len(reference_dataset_test_2))
    print("Query Images Test 2:", len(query_dataset_test_2))

    #-----------------------------------------------------------------------------#
    # Evaluate - For both models and datasets
    #-----------------------------------------------------------------------------#
    print("\n{}[{}]{}".format(30 * "-", "CVUSA", 30 * "-"))  

    r1_test = evaluate(config=config,
                       model_1=model_1, 
                       model_2=model_2,
                       reference_dataloader1=reference_dataloader_test_1, 
                       reference_dataloader2=reference_dataloader_test_2,
                       query_dataloader=query_dataloader_test_1, 
                       BEV_dataloader=query_dataloader_test_2,
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)

    # Output evaluation results
    print("Evaluation Results for Model 1 and Model 2:", r1_test)
