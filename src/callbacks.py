from keras.callbacks import Callback
import os
import time
import matplotlib.pyplot as plt
from IPython import display

class PlotCurves(Callback):

    def __init__(self, model_name):
        self.model_name = model_name
        
    def on_train_begin(self, logs={}):
        self.epoch = 0
        self.best_epoch = 0
        self.best_f1_epoch = 0
        self.x = []
        self.losses = []
        self.acc = []
        self.f1 = []
        self.val_losses = []
        self.val_acc = []
        self.val_f1 = []
        self.best_val_acc = 0
        self.best_val_f1 = 0
        self.fig = plt.figure(figsize=(10, 5))

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.fig = plt.figure(figsize=(10, 5))
        self.logs.append(logs)
        self.x.append(self.epoch)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.f1.append(logs.get('f1_score'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.val_f1.append(logs.get('val_f1_score'))
        self.epoch += 1

        model_dir = './Model/' + self.model_name
        os.makedirs(model_dir, exist_ok=True)

        # Save at each epoch
#         self.model.save(os.path.join(model_dir, self.model_name + '_epoch_' + str(self.epoch) + '.h5'))
        
        # (Possibly) update best validation accuracy
        if self.val_acc[-1] > self.best_val_acc:
            self.best_val_acc = self.val_acc[-1]
            self.best_epoch = self.epoch

        # (Possibly) update best validation F1-score
        if self.val_f1[-1] > self.best_val_f1:
            with open(os.path.join(model_dir, 'best_f1_model.txt'), 'w') as f:
                f.write(str(self.epoch) + ' => f1:' + str(self.best_val_f1))
            self.best_val_f1 = self.val_f1[-1]
            self.best_f1_epoch = self.epoch

        display.clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.plot(self.x, self.f1, label="f1_score")
        plt.plot(self.x, self.val_f1, label="val_f1_score")
        plt.legend()
        plt.title('Best validation accuracy = {:.2f}% on epoch {} of {} \n' \
                  'Best validation F1-score = {:.2f}% on epoch {} of {}'.format(
                        100. * self.best_val_acc, self.best_epoch, self.epoch,
                        100. * self.best_val_f1, self.best_f1_epoch, self.epoch))
        plt.show();