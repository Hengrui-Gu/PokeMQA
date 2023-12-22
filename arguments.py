import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='PokeMQA Arguments.')

    parser.add_argument('--edited-num', dest='edited_num', type=int, default=1,
            help='Number of edited instances')
    parser.add_argument('--dataset', dest='dataset_name', type=str, default="MQuAKE-CF-3k",
            help='dataset name')

    parser.add_argument('--retraining_detector', action='store_true')
    parser.add_argument('--detector_name', dest='cls_name', type=str, default="detector-ckpt",
            help='classifier name')

    parser.add_argument('--retraining_disambiguator', action='store_true')
    parser.add_argument('--dis_name', dest='seq_name', type=str, default="dis-ckpt",
            help='Sequential classifier name')
    
    parser.add_argument('--activate_kgprompt', action='store_true',help='w or w/o knowledge prompt')

    return parser.parse_args()
