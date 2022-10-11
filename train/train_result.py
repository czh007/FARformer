# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib
import  pandas as pd
import csv

print(matplotlib.get_backend())


def plot_f_fill(epoch_f):
    ax=plt.gca()
    num_epochs=len(epoch_f)
    window = int(num_epochs / 50)
    print('window:', window)
    rolling_mean = pd.Series(epoch_f).rolling(window).mean()
    std = pd.Series(epoch_f).rolling(window).std()
    x = [i for i in range(len(epoch_f))]
    plt.plot(x,rolling_mean)
    ax.fill_between(range(len(epoch_f)), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2)
    ax.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('F1')
    ax.set_ylim(0, 1)
    # Example of ax as two coordinate axes
    # Set the main scale of the x-axis to a multiple of 1
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()

def plot_f(epoch_f):
    ax = plt.gca()
    x = [i for i in range(len(epoch_f))]
    plt.plot(x,epoch_f)
    ax.set_title('Performance on valid set')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('F1')
    ax.set_ylim(0, 1)
    # Example of ax as two coordinate axes
    # Set the main scale of the x-axis to a multiple of 1
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()

def plot_loss(losses):
    # Read the loss value from the file
    ax = plt.gca()
    x=[i for i in range(len(losses))]
    plt.plot(x,losses)
    ax.set_title('Performance on valid set')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Loss')
    # Example of ax as two coordinate axes
    # Set the main scale of the x-axis to a multiple of 1
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()


if __name__ == '__main__':
    # plot results
    # Drawings
    # Set the scale interval of x-axis to 1 and put it in the variable
    x_major_locator = MultipleLocator(1)
    # Set the scale interval of y-axis to 10 and put it in the variable
    y_major_locator = MultipleLocator(0.1)

    with open('data/train_losses.csv','r') as f:
        reader=csv.reader(f)
        loss=[float(row[0]) for row in reader]

    plot_loss(loss)

    with open('data/valid_f.csv','r') as f:
        reader=csv.reader(f)
        valid_f=[float(row[0]) for row in reader]
    plot_f(valid_f)
    plot_f_fill(valid_f)
