"""
Training script for FedDISC
dataset_name: Selecting dataset (mosi  mosei iemocap4 iemocap6)
"""
from run import FedDISC_run

 
if __name__ == "__main__":

    
    FedDISC_run(model_name='disc',
            dataset_name='iemocap4',
            seeds=[1111],
            mr=0.1)
  