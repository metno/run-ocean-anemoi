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
    
def inference(input_yaml, start, end):
    '''
        Runs inference for multiple dates.
    Args:
        input_yaml  [str]   :   name of yaml file   
        start       [str]   :   start time 
        end         [str]   :   end time
    '''
    logging.basicConfig(level=logging.INFO)
    start =  datetime.strptime(start, '%Y-%m-%d')
    end = datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1) #adding extra to include the last day
    times = np.arange(start, end, timedelta(days=1)).astype(datetime)
    logger.info(f'Running inference for dates between {start} - {end}.')

    for time in times:
        date = f'{time.year}-{time.month:02d}-{time.day:02d}'
        logger.info(f'Starting inference for {date}')
        set_date(input_yaml, date)
        
        os.system(f'anemoi-inference run {input_yaml}')
        os.system(f'python ../postpro-inference.py {input_yaml}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='long inference'
    )
    parser.add_argument(
        '-f', '--file', type=str, required=True, help='Input yaml file'   
    )
    parser.add_argument(
        '-s', '--start', type=str, required=True, help='Start date, %Y-%m-%d'
    )
    parser.add_argument(
        '-e', '--end', type=str, required=True, help='End date, %Y-%m-%d'
    )
    args = parser.parse_args()

    inference(
        input_yaml=args.file,
        start=args.start,
        end=args.end
    )
