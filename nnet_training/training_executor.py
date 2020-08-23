#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

from training_frameworks.mono_seg_trainer import MonoSegmentationTrainer, Init_Training_MonoFSCNN

if __name__ == "__main__":

    usr_input = int(input("1: Mono FSCNN\nUser Input:"))
    if (usr_input == 1):
        modeltrainer = Init_Training_MonoFSCNN()

    #   User Input Training loop
    while_cond = 1
    while(while_cond == 1):
        n_epochs = int(input("Number of Training Epochs: "))
        modeltrainer.train_model(n_epochs)

        usr_input = int(input("Show Training Statistics? (1/0): "))
        if (usr_input == 1):
            modeltrainer.plot_data()
    
        usr_input = int(input("Show Example Output? (1/0): "))
        if (usr_input == 1):
            modeltrainer.visualize_output()

        usr_input = int(input("Pass Specific Image? (1/0): "))
        if (usr_input == 1):
            usr_input = str(input("Enter Path: "))
            modeltrainer.custom_image(usr_input)
        
        print("Current Learning Rate: ", modeltrainer.get_learning_rate)
        
        usr_input = int(input("New Learning Rate? (1/0): "))
        if (usr_input == 1):
            usr_input = float(input("New Learning Rate Value: "))
            modeltrainer.set_learning_rate(usr_input)

        while_cond = int(input("Continue Training? (1/0): "))