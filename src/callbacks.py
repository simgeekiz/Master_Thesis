from keras.callbacks import Callback
import os
import time
import matplotlib.pyplot as plt
from IPython import display

class PlotCurves(Callback):

    def __init__(self, model_name, save=True):
        self.model_name = model_name
        self.save = save
        
        self.model_dir = './Model/' + self.model_name
        os.makedirs(self.model_dir, exist_ok=True)
        self.meta_file = os.path.join(self.model_dir, 'model_metadata.txt')
        
        with open(self.meta_file, 'w') as f:
            f.write('\n----------------\n')
        
    def on_train_begin(self, logs={}):
        self.epoch = 0
        
        self.best_acc_epoch = 0
        self.best_f1_macro_epoch = 0
        self.best_f1_micro_epoch = 0
        
        self.x = []
        self.losses = []
        
        self.acc = []
        self.f1_macro = []
        self.f1_micro = []
        
        self.val_losses = []
        self.val_acc = []
        self.val_f1_macro = []
        self.val_f1_micro = []
        
        self.best_val_acc = 0
        self.best_val_f1_macro = 0
        self.best_val_f1_micro = 0

        self.fig = plt.figure(figsize=(10, 5))
        self.logs = []
        
        with open(self.meta_file, 'a') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write('\n')

    def on_epoch_end(self, epoch, logs={}):

        self.fig = plt.figure(figsize=(10, 5))
        self.logs.append(logs)
        self.x.append(self.epoch)
        self.losses.append(logs.get('loss'))
        
        self.acc.append(logs.get('acc'))
        self.f1_macro.append(logs.get('f1_macro'))
        self.f1_micro.append(logs.get('f1_micro'))
        
        self.val_losses.append(logs.get('val_loss'))
        
        self.val_acc.append(logs.get('val_acc'))
        self.val_f1_macro.append(logs.get('val_f1_macro'))
        self.val_f1_micro.append(logs.get('val_f1_micro'))
        
        self.epoch += 1

        if self.save:
            # Save at each epoch
            self.model.save(os.path.join(self.model_dir, self.model_name + '_epoch_' + str(self.epoch) + '.h5'))
        
        # (Possibly) update best validation accuracy
        if self.val_acc[-1] > self.best_val_acc:
            self.best_val_acc = self.val_acc[-1]
            self.best_acc_epoch = self.epoch

        # (Possibly) update best validation F1-macro
        if self.val_f1_macro[-1] > self.best_val_f1_macro:
            self.best_val_f1_macro = self.val_f1_macro[-1]
            self.best_f1_macro_epoch = self.epoch
            self.model.save(os.path.join(self.model_dir, self.model_name + '_best_f1_macro_model.h5'))
            
        # (Possibly) update best validation F1-micro
        if self.val_f1_micro[-1] > self.best_val_f1_micro:
            self.best_val_f1_micro = self.val_f1_micro[-1]
            self.best_f1_micro_epoch = self.epoch
#             self.model.save(os.path.join(self.model_dir, self.model_name + '_best_f1_micro_model.h5'))
            
        with open(self.meta_file, 'a') as f:
            f.write(str(self.epoch) 
                    + ' => val_f1_macro: ' + str(self.val_f1_macro[-1]) 
                    + ' | val_f1_micro: ' + str(self.val_f1_micro[-1]) 
                    + ' | val_acc: ' + str(self.val_acc[-1]))
            f.write('\n')

        display.clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.plot(self.x, self.f1_macro, label="f1_macro")
        plt.plot(self.x, self.val_f1_macro, label="val_f1_macro")
        plt.plot(self.x, self.f1_micro, label="f1_micro")
        plt.plot(self.x, self.val_f1_micro, label="val_f1_micro")
        plt.legend()
        plt.title('Best validation accuracy = {:.2f}% on epoch {} of {} \n' \
                  'Best validation F1-macro = {:.2f}% on epoch {} of {} \n' \
                  'Best validation F1-micro = {:.2f}% on epoch {} of {} \n'.format(
                        100. * self.best_val_acc, self.best_acc_epoch, self.epoch,
                        100. * self.best_val_f1_macro, self.best_f1_macro_epoch, self.epoch,
                        100. * self.best_val_f1_micro, self.best_f1_micro_epoch, self.epoch))
        plt.savefig(os.path.join(self.model_dir, self.model_name + '.png'))
        plt.show();
