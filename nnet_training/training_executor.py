#!/usr/bin/env python3

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

from training_frameworks.mono_seg_trainer import MonoSegmentationTrainer, Init_Training_MonoFSCNN

if __name__ == "__main__":
    if int(input("1: Mono FSCNN\nUser Input:")) == 1:
        MODELTRAINER = Init_Training_MonoFSCNN()

    #   User Input Training loop
    LOOP_COND = 1
    while LOOP_COND == 1:
        MODELTRAINER.train_model(int(input("Number of Training Epochs: ")))

        if int(input("Show Training Statistics? (1/0): ")) == 1:
            MODELTRAINER.plot_data()

        if int(input("Show Example Output? (1/0): ")) == 1:
            MODELTRAINER.visualize_output()

        if int(input("Pass Specific Image? (1/0): ")) == 1:
            MODELTRAINER.custom_image(str(input("Enter Path: ")))

        print("Current Learning Rate: ", MODELTRAINER.get_learning_rate)

        if int(input("New Learning Rate? (1/0): ")) == 1:
            MODELTRAINER.set_learning_rate(float(input("New Learning Rate Value: ")))

        LOOP_COND = int(input("Continue Training? (1/0): "))
