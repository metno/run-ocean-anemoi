import os
import ruamel.yaml
import sys


def set_date_in_yaml(input_yaml, date):
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


if __name__ == '__main__':
    import argparse
    set_date_in_yaml('infer.yaml', '---')

