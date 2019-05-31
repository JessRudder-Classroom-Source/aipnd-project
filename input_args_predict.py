# Import python modules
import argparse

def get_input_args():
    # Create the argparser object
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path',
                        type=str,
                        action='store',
                       )
    
    parser.add_argument('checkpoint',
                        type=str,
                        action='store',
                       )

    parser.add_argument('--arch',
                        type=str,
                        action='store',
                        dest='arch',
                        default='densenet',
                        help='CNN model architecture to use'
                       )

    parser.add_argument('--hidden_units',
                        type=int,
                        action='store',
                        dest='hidden_units',
                        default=512,
                        help='integer for the hidden units'
                       )

    parser.add_argument('--top_k',
                        type=int,
                        action='store',
                        dest='top_k',
                        default=5,
                        help='integer for the number of proability estimates you want to see'
                       )

    parser.add_argument('--category_names',
                        type=str,
                        action='store',
                        dest='category_names',
                        default='cat_to_name.json',
                        help='integer for the number of proability estimates you want to see'
                       )

    parser.add_argument('--gpu',
                        action='store_true',
                        dest='gpu',
                        help='flag to indicate you want to run on the GPU'
                       )

    return parser.parse_args()
