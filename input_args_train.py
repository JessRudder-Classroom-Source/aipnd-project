# Import python modules
import argparse

def get_input_args():
    # Create the argparser object
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir',
                        action='store',
                       )
    
    parser.add_argument('--save_dir',
                        type=str,
                        action='store',
                        dest='save_dir',
                        default='checkpoints/',
                        help='path to the folder where we save the checkpoints'
                       )

    parser.add_argument('--arch',
                        type=str,
                        action='store',
                        dest='arch',
                        default='densenet',
                        help='CNN model architecture to use'
                       )

    parser.add_argument('--learning_rate',
                        type=float,
                        action='store',
                        dest='learning_rate',
                        default=0.003,
                        help='float to indicate the learning rate'
                       )

    parser.add_argument('--hidden_units',
                        type=int,
                        action='store',
                        dest='hidden_units',
                        default=512,
                        help='integer for the hidden units'
                       )

    parser.add_argument('--epochs',
                        type=int,
                        action='store',
                        dest='epochs',
                        default=10,
                        help='integer to set the number of epochs for the data to train'
                       )

    parser.add_argument('--gpu',
                        action='store_true',
                        dest='gpu',
                        help='flag to indicate you want to run on the GPU'
                       )

    parser.add_argument('--validate',
                        action='store_true',
                        dest='validate',
                        help='flag to indicate you want to run the validation pass'
                       )

    return parser.parse_args()
