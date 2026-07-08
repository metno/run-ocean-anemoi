import os
import ruamel.yaml
from datetime import datetime, timedelta
import numpy as np
import sys
import logging
logger = logging.getLogger(__name__)

def set_date(input_yaml, date):
    '''
        Updates the date in yaml file
    Args:
        input_yaml  [str]   :   name of yaml file
        date        [str]   :   date string
    '''
    yaml = ruamel.yaml.YAML()
    with open(input_yaml) as f:
        data = yaml.load(f)
    
    if 'date' in data:
        data['date'] = date
    else:
        raise ValueError(f'No key date found in {input_yaml}')
    
    with open(input_yaml, 'w') as f:
        yaml.dump(data, f)

def set_checkpoint(input_yaml, checkpoint):

    yaml = ruamel.yaml.YAML()
    with open(input_yaml) as f:
        data = yaml.load(f)
    
    if 'checkpoint' in data:
        if checkpoint is not None:
            data['checkpoint'] = checkpoint
        logger.info(f'Running inference for checkpoint: {data["checkpoint"]}')
    else: 
        raise ValueError(f'No key checkpoint found in {input_yaml}')
    
    with open(input_yaml, 'w') as f:
        yaml.dump(data, f)
    
def inference(input_yaml, start, end, checkpoint_list=False):
    '''
        Runs inference for multiple dates.
    Args:
        input_yaml  [str]   :   name of yaml file   
        start       [str]   :   start time 
        end         [str]   :   end time
    '''
    logging.basicConfig(level=logging.INFO)

    start = datetime.strptime(start, '%Y-%m-%d')
    end = datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1) if end is not None else None #adding extra to include the last day

    times = np.arange(start, end, timedelta(days=1)).astype(datetime) if end is not None else np.array([start])
    
    # getting times
    if len(times) > 1:
        logger.info(f'Running inference for dates between {start} - {end}.')
    elif len(times) == 1:
        logger.info(f'Running inference for {start}.')
    
    # getting checkpoints
    checkpoints = []
    if checkpoint_list:
        logger.info('Running inference for checkpoints defined in checkpoint_list.csv')
        with open('checkpoint_list.csv', 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                logger.warning('No checkpoints found in checkpoint_list.csv')
            for ckpt in lines:
                checkpoints.append(ckpt.strip())

    if len(checkpoints) == 0:
        logger.info('Using checkpoint defined in yaml file')
        checkpoints = [None]

    for checkpoint in checkpoints:
        set_checkpoint(input_yaml, checkpoint)
        for time in times:
            date = f'{time.year}-{time.month:02d}-{time.day:02d}'
            logger.info(f'Starting inference for {date}')
            set_date(input_yaml, date)

            os.system(f'anemoi-inference run {input_yaml}')
            os.system(f'python ../postpro-inference.py {input_yaml}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='multi inference'
    )
    parser.add_argument(
        '-f', '--file', type=str, required=True, help='Input yaml file'   
    )
    parser.add_argument(
        '-s', '--start', type=str, required=True, help='Start date, Y-m-d'
    )
    parser.add_argument(
        '-e', '--end', type=str, required=False, help='End date, Y-m-d'
    )
    parser.add_argument(
        '-cl', '--checkpoint_list', action='store_true', help='Flag to run multiple inferences from checkpoints defined in checkpoint_list.csv'
    )
    args = parser.parse_args()

    inference(
        input_yaml=args.file,
        start=args.start,
        end=args.end,
        checkpoint_list=args.checkpoint_list
    )
