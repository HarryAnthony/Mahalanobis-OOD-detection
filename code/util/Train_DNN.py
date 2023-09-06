from torch.autograd import Variable
import numpy as np
import torch
from util.general_utils import print_progress
import sys
import time

class Train_DNN():
    """
    Train a neural network.

    Parameters
    ----------
    training_dict : dict
        A dictionary containing the training parameters.
    """

    def __init__(self, training_dict, **kwargs):
        self.net = training_dict['net']
        self.trainloader = training_dict['trainloader']
        self.validationloader = training_dict['validationloader']
        self.optimiser = training_dict['optimiser']
        self.criterion = training_dict['criterion']
        self.scheduler = training_dict['scheduler']
        self.scheduler_name = training_dict['scheduler_name']
        self.num_epochs = training_dict['num_epochs']
        self.use_cuda = training_dict['use_cuda']
        self.verbose = training_dict['verbose']
        self.epoch = 0
        self.training_mode = 0
        self.best_acc = 0
        self.save_model_mode = training_dict['save_model_mode']
        self.save_point = training_dict['save_point']
        self.filename = training_dict['filename']
        self.seed = training_dict['seed']

        np.random.seed(self.seed)


    def __call__(self,**kwargs):
        self.train_net(**kwargs)


    def train_net(self,**kwargs):
        """
        Train the neural network for the specified number of epochs, and validate after each epoch.
        """
        start_time = time.time()
        epoch =0 

        for epoch in range(0,self.num_epochs):
            self.epoch = epoch
            self.train_net_one_epoch(**kwargs)
            self.validate_net_one_epoch(**kwargs)

            if self.verbose:
                print('| Elapsed time : %d:%02d:%02d' % (self.get_hms(time.time() - start_time)))

            if self.save_model_mode == 'best_acc' and self.accuracy > self.best_acc:
                print('\n| Saving Best model...\tBest accuracy = %.2f%%, epoch = %d' % (self.accuracy, self.epoch))
                self.save_model()
                self.best_acc = self.accuracy

        if self.save_model_mode == 'last_epoch':
                print('\n| Saving Best model...\tBest accuracy = %.2f%%, epoch = %d' % (self.accuracy, self.epoch))
                self.save_model()
                self.best_acc = self.accuracy


    def apply_net(self,inputs,**kwargs):
        """
        Apply the model to the inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            The inputs of the model.

        Returns
        -------
        outputs : torch.Tensor
            The outputs of the model.
        """
        if self.use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        outputs = self.net(inputs)
        return outputs


    def calc_loss(self,outputs,targets,**kwargs):
        """
        Calculate the loss of the model.

        Parameters
        ----------
        outputs : torch.Tensor
            The outputs of the model.
        targets : torch.Tensor
            The targets of the model.

        Returns
        -------
        loss : torch.Tensor
            The loss of the model.
        """
        loss = self.criterion(outputs, targets)
        self.epoch_loss_list += loss.item()
        self.loss_mean = self.epoch_loss_list / (self.batch_idx + 1)

        return loss
    

    def classify_inputs(self,inputs,targets,**kwargs):
        """
        Classify the inputs of the model.

        Parameters
        ----------
        inputs : torch.Tensor
            The inputs of the model.
        targets : torch.Tensor
            The targets of the model.

        Returns
        -------
        outputs : torch.Tensor
            The outputs of the model.
        loss : torch.Tensor
            The loss of the model.
        targets : torch.Tensor
            The targets of the model.
        """
        outputs = self.apply_net(inputs)
        if self.use_cuda:
            targets = targets.cuda()
        loss = self.calc_loss(outputs,targets)
        return outputs, loss, targets
        

    def determine_accuracy(self,outputs,targets,**kwargs):
        """
        Determine the accuracy of the model on a batch of data.
        
        Parameters
        ----------
        outputs : torch.Tensor
            The outputs of the model.
        targets : torch.Tensor
            The targets of the model.
        """
        _, predicted = torch.max(outputs.data, 1)
        if self.use_cuda:
            predicted = predicted.cuda()
        batch_total = targets.size(0)
        batch_correct = (targets == predicted).sum()

        self.epoch_total += batch_total
        self.epoch_correct += batch_correct
        self.accuracy = 100.*self.epoch_correct/self.epoch_total

    
    def grad_descent(self,loss,gradient_clipping=False,**kwargs):
        """
        Perform gradient descent on the model.
        """
        self.optimiser.zero_grad()
        loss.backward()
        if gradient_clipping == True:
            self._grad_clip(**kwargs)
        self.optimiser.step()


    def _grad_clip(self,clip_value=0.5,**kwargs):
        """
        Perform gradient clipping on model parameters.
        """
        for param in self.net.parameters():
            if param.grad is not None:
                param.grad.data = torch.clamp(param.grad.data, -clip_value, clip_value)


    def initialise_progress_bar(self,**kwargs):
        """
        Initialise the progress bar.
        """
        print_progress(0, len(self.trainloader), prefix = 'Progress:', suffix = 'Complete', length = 50, verbose=self.verbose)


    def update_progress_bar(self,**kwargs):
        """
        Update the progress bar.
        """
        print_progress(self.batch_idx + 1, len(self.trainloader), prefix = 'Progress:', suffix = 'Complete', length = 50, verbose=self.verbose)


    def log_training_info(self,**kwargs):  
        """
        Output the training information (epoch number, learning rate)
        """
        if self.verbose:  
            print('\n=> Training Epoch [%3d/%3d], LR=%.4f' % (self.epoch, self.num_epochs, self.scheduler.get_last_lr()[0]))


    def log_batch_progress(self,**kwargs):
        """
        Output the progress of the batch.
        """
        if self.verbose:  
            sys.stdout.write('\r')
            sys.stdout.write('| Batch [%3d] \tLoss: %.4f \tAcc@1: %.3f%%'
                                % (self.batch_idx, self.loss_mean, self.accuracy))
        

    def log_training_epoch_progress(self,**kwargs):
        """
        Output the progress of the epoch.
        """
        if self.verbose:    
            sys.stdout.write('\r')
            sys.stdout.write('| Training \tLoss: %.4f \tAcc@1: %.3f%%'
                                % (self.loss_mean, self.accuracy))
            

    def log_validation_epoch_progress(self,**kwargs):
        """
        Output the progress of the epoch.
        """
        if self.verbose:    
            print('\n| Validation \tLoss: %.4f \tAcc@1: %.3f%%'
                                % (self.loss_mean, self.accuracy))
            

    def update_scheduler_batch_iter(self,**kwargs):
        """
        Update the learning rate scheduler after each batch.
        """
        if self.scheduler_name == 'OneCycleLR':
            self.scheduler.step()
        elif self.scheduler_name == 'CosineAnnealingLR':
            self.scheduler.step()

    def update_scheduler(self,**kwargs):
        """
        Update the learning rate scheduler.
        """
        if self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler.step(self.loss_mean)
        elif self.scheduler_name != 'OneCycleLR' and self.scheduler_name != 'CosineAnnealingLR':
            self.scheduler.step()


    def initialise_training_stats(self,**kwargs):
        """
        Initialise the training statistics.
        """
        self.initialise_eval_metrics()
        self.epoch_loss_list = torch.zeros(1)


    def initialise_net_for_training(self,**kwargs):
        """
        Initialise the model for training.
        """
        self.net.train()
        self.training_mode = 1


    def initialise_net_for_eval(self,**kwargs):
        """
        Initialise the model for evaluation.
        """
        self.net.eval()
        self.training_mode = 0


    def initialise_eval_metrics(self,**kwargs):
        """
        Initialise the evaluation metrics.
        """
        self.epoch_total = 0
        self.epoch_correct = torch.zeros(1)
        if self.use_cuda:
            self.epoch_correct = Variable(self.epoch_correct.cuda())
        self.batch_idx = 0


    def save_model(self,**kwargs):
        """
        Save the model.
        """
        torch.save(self.net.state_dict(), self.save_point + self.filename+'-'+str(self.seed)+'.pth')


    def get_hms(self,seconds):
        """
        Convert seconds to hours, minutes and seconds.
        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return h, m, s


    def train_net_one_epoch(self,**kwargs):
        """
        Train the model for one epoch.
        """
        self.initialise_net_for_training()
        self.log_training_info()
        self.initialise_progress_bar()
        self.initialise_training_stats()
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            self.batch_idx = batch_idx
            self.update_progress_bar()
            outputs, loss, targets = self.classify_inputs(inputs,targets)
            del inputs
            self.grad_descent(loss,**kwargs)
            self.determine_accuracy(outputs,targets)
            self.update_scheduler_batch_iter()

        self.log_training_epoch_progress()
        self.update_scheduler()


    def validate_net_one_epoch(self,**kwargs):
        """
        Validate the model for one epoch.
        """
        self.initialise_net_for_eval()
        self.initialise_training_stats()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.validationloader):
                self.batch_idx = batch_idx
                outputs, _, targets = self.classify_inputs(inputs,targets)
                self.determine_accuracy(outputs,targets)

        self.log_validation_epoch_progress()


    def __repr__(self):
        return str(self.net)

        
    def __str__(self):
        return str(self.net)
