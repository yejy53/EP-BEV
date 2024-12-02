import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.cvact_c_bev import CVACTDatasetEval, CVACTDatasetTest
from sample4geo.transforms import get_transforms_val
from sample4geo.evaluate.cvact import evaluate,calc_sim
from sample4geo.model import TimmModel


@dataclass
class Configuration:

    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation 
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128    # pool size for selection
    # Dataset
    data_folder = "data_path"     
        
    # Checkpoint to start from
    checkpoint_start = 'check_point'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    image_size_sat = (img_size, img_size)
    
    # new_width = config.img_size * 2    
    # new_hight = round((224 / 1232) * new_width)
    # img_size_ground = (new_hight, new_width)

    img_size_ground = image_size_sat 
    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # Transforms                                                                  #
    #-----------------------------------------------------------------------------#

    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    #-----------------------------------------------------------------------------#
    # Validation                                                                  #
    #-----------------------------------------------------------------------------#

    # Reference Satellite Images
    reference_dataset_val = CVACTDatasetEval(data_folder=config.data_folder ,
                                             split="val",
                                             img_type="reference",
                                             transforms=sat_transforms_val,
                                             )
    
    reference_dataloader_val = DataLoader(reference_dataset_val,
                                          batch_size=config.batch_size,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_val = CVACTDatasetEval(data_folder=config.data_folder ,
                                         split="val",
                                         img_type="query",    
                                         transforms=ground_transforms_val,
                                         )
    
    query_dataloader_val = DataLoader(query_dataset_val,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      shuffle=False,
                                      pin_memory=True)
    
    
    print("Reference Images Val:", len(reference_dataset_val))
    print("Query Images Val:", len(query_dataset_val))
    
 
    print("\n{}[{}]{}".format(30*"-", "CVACT_VAL", 30*"-"))  

  
    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=reference_dataloader_val,
                       query_dataloader=query_dataloader_val, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
    