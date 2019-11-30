class EarlyStopping(object):
    
    def __init__(self, best_performance, max_patience):
        super(EarlyStopping, self).__init__()
        
        self.best_performance = best_performance
        self.patience = 0
        self.max_patience = max_patience
        #self.epochs_interval = epochs_interval
        
    def lr_decay(self, optimizer, decay_factor):
        print("\nDecaying learning rate")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_factor
        print("New learning rate: %f\n" % (optimizer.param_groups[0]['lr'],))
        
    def check_improvement(self, recent_performance):
        if recent_performance > self.best_performance:
            self.best_performance = recent_performance
            self.patience = 0
            return True
        else:
            self.patience += 1
            print("\nEpochs since last best performance: %d\n" % (self.patience))
            return False
        
