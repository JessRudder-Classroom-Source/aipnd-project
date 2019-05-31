# Import python modules
import json
import numpy as np

# Imports functions created for this program
from input_args_predict import get_input_args
import utils

# Main program function defined below
def main():
    print("The predictions have begun...")

    in_arg = get_input_args()
    
    device = "cuda" if in_arg.gpu else "cpu"
    model = utils.load_checkpoint(in_arg.checkpoint, in_arg.arch, in_arg.hidden_units)
    probs, classes = utils.predict(in_arg.image_path, model, in_arg.top_k, device)

    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    labels = [cat_to_name[str(index + 1)] for index in np.array(classes)]
    probs = np.array(probs)
    
    i=0
    while i < in_arg.top_k:
        print("{} - probability of {}%".format(labels[i], probs[i] * 100))
        i += 1

    print("The predictions have ended...")

# Call to main function to run the program
if __name__ == "__main__":
    main()
