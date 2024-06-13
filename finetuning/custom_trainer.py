from trl import RewardTrainer, DPOTrainer
from transformers.trainer_utils import get_last_checkpoint


class CustomRewardTrainer(RewardTrainer): 
    '''
    Create a custom trl reward trainer that supports loading from the last checkpoint
    '''
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
    def load(self, checkpoint = None): 
        if checkpoint is None: 
            checkpoint = get_last_checkpoint(self.args.output_dir)
        self._load_from_checkpoint(resume_from_checkpoint=checkpoint)
        
class CustomDPOTrainer(DPOTrainer): 
    '''
    Create a trl DPO trainer that supports loading from the last checkpoint
    '''
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

    def load(self, checkpoint = None): 
        if checkpoint is None: 
            checkpoint = get_last_checkpoint(self.args.output_dir)
        print(checkpoint)
        self._load_from_checkpoint(resume_from_checkpoint=checkpoint)