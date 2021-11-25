import functions as f
import argparse
from argparse import RawTextHelpFormatter
from sys import argv
# f.synthetic_data_generator()
# train,test = f.data_loader()

# print("This is my file to test Python's execution methods.")
# print("The variable __name__ tells me which context this file is running in.")
# print("The value of __name__ is:", repr(__name__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Chose one of the three modes: Demo, Model, Data',formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-m','--mode',
        help = '''Specify the mode that the program will run (Default = Demo)
             \n\t Demo: Create the dataset and train a CNN classification model
             \n\t Data: Create the test Dataset for the project
             \n\t CNN : Create and Train a Deep Learning model and print the confusion matrix 
             \n\t HMM : Create and Train a Hiden Markov model and print the confusion matrix''',
        choices = ['Demo','CNN','HMM','Data'] ,
        default='Demo')
    args = parser.parse_args()

    if args.mode == 'Demo':
        print('Demo mode on')
        f.synthetic_data_generator()
        print('Example Dataset Created')
        model = f.Model()
        print(model.summary())
        history = model.train()
        print('Model is Trained')
    elif args.mode == 'Data':
        print('Data mode on')
        f.synthetic_data_generator()
        print('Example Dataset Created')
    elif args.mode == 'CNN':
        model = f.Model()
        print(model.summary())
        history = model.train()
    elif args.mode == 'HMM':
        print('HMM mode on')