import re
import numpy as np
from matplotlib import pyplot as plt


def get_vals(path):
    file = open(path, 'r')
    filedata = file.read()
    epochs = []
    l = []
    ba = []
    mba = []
    steps = 14  # see the logs to set this value
    pattern = re.compile('Epoch:(.*) Step:(.*) Loss:(.*) Batch Acc:(.*) Mean Batch Acc:(.*)')
    for line in filedata.split("\n"):
        if len(pattern.findall(line)) == 0:
            continue
        res = pattern.findall(line)[0]
        epoch = res[0]
        loss = res[2]
        batch_acc = res[3]
        mean_batch_acc = res[4]
        if epoch not in epochs:
            epochs.append(epoch)
            l.append([float(loss)])
            ba.append([float(batch_acc)])
            mba.append([float(mean_batch_acc)])
        else:
            l[-1].append(float(loss))
            ba[-1].append(float(batch_acc))
            mba[-1].append(float(mean_batch_acc))

    mean_l = []
    mean_ba = []
    mean_mba = []
    for i in range(len(l)):
        mean_l.append(np.mean(l[i]))
        mean_ba.append(np.mean(ba[i]))
        mean_mba.append(np.mean(mba[i]))
    file.close()
    return mean_l, mean_ba


if __name__ == '__main__':
    path_gru = "log_files/log_gru.txt"
    path_gru_att = "log_files/log_gru_att.txt"
    path_lstm = "log_files/log_lstm.txt"
    path_lstm_att = "log_files/log_lstm_att.txt"

    mlg, mbg = get_vals(path_gru)
    mlga, mbga = get_vals(path_gru_att)

    mll, mbl = get_vals(path_lstm)
    mlla, mbla = get_vals(path_lstm_att)

    num_epochs = len(mlg)

    plt.plot(mlg, marker='*', label='GRU')
    plt.plot(mlga, marker='+', label='GRU with Attention')
    plt.xticks(np.arange(1, num_epochs + 1))
    plt.legend()
    # plt.title("Loss")
    # plt.show()
    #
    plt.plot(mll, marker='x', label='LSTM')
    plt.plot(mlla, marker='o', label='LSTM with Attention')
    plt.xticks(np.arange(1, num_epochs + 1))
    plt.legend()
    plt.title("Training Loss")
    plt.show()

    plt.plot(mbg, marker='*', label='GRU w/ Att.') #opp
    plt.plot(mbga, marker='+', label='GRU') #opp
    plt.xticks(np.arange(1, num_epochs + 1))
    # plt.legend()
    # plt.title("Batch Accuracy")
    # plt.show()

    plt.plot(mbl, marker='x', label='LSTM w/ Att.') #opp
    plt.plot(mbla, marker='o', label='LSTM') #opp
    plt.xticks(np.arange(1, num_epochs + 1))
    plt.legend()
    plt.title("Training Batch Accuracy")
    plt.show()

    # plt.plot(mlg, marker='*', label='GRU')
    # plt.plot(mll, marker='*', label='LSTM')
    # plt.xticks(np.arange(1, num_epochs + 1))
    # plt.legend()
    # plt.title("Loss")
    # plt.show()
    #
    # plt.plot(mlga, marker='*', label='GRU with Attention')
    # plt.plot(mlla, marker='*', label='LSTM with Attention')
    # plt.xticks(np.arange(1, num_epochs + 1))
    # plt.legend()
    # plt.title("Loss")
    # plt.show()
    #
    # plt.plot(mbg, marker='o', label='GRU')
    # plt.plot(mbl, marker='o', label='LSTM')
    # plt.xticks(np.arange(1, num_epochs + 1))
    # plt.legend()
    # plt.title("Batch Accuracy")
    # plt.show()
    #
    # plt.plot(mbga, marker='o', label='GRU with Attention')
    # plt.plot(mbla, marker='o', label='LSTM with Attention')
    # plt.xticks(np.arange(1, num_epochs + 1))
    # plt.legend()
    # plt.title("Batch Accuracy")
    # plt.show()



