# Imports functions created for this program
from input_args_train import get_input_args
import utils

# Main program function defined below
def main():
    in_arg = get_input_args()
    device = "cuda" if in_arg.gpu else "cpu"

    trainloader, testloader, validloader, train_data = utils.create_loaders(in_arg.data_dir)
    model, device, criterion, optimizer = utils.set_up_model_params(in_arg.arch, in_arg.learning_rate, in_arg.hidden_units, device)
    utils.train_the_model(model, trainloader, validloader, criterion, optimizer, device, in_arg.epochs)
    if in_arg.validate:
        utils.validate_model(model, testloader, device)
    utils.save_model(model, optimizer, in_arg.save_dir, in_arg.arch, train_data)

# Call to main function to run the program
if __name__ == "__main__":
    main()
