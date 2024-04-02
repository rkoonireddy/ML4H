import pandas as pd


class Parameters:
    def __init__(self):
        # Number of epochs used for training
        self.num_of_epochs = 15
        # Batch size used
        self.batch_size = 64
        # Start value for learning rate
        # 0.001 lr is too big - not good results!
        self.learning_rate = 0.0005
        # Learning rate scheduler parameters
        self.scheduler_step_size = 5
        self.scheduler_gama = 0.33

        # Model name
        self.model = 'ResNet18'
        # model_parameters = '' - no starting parameters;
        # model_parameters = 'default' - pretrained network on ImageNet dataset;
        # model_parameters = './models/model1.pth' - starting parameters are taken from model1.pth
        self.model_parameters = 'default'
        # optionally untypical neural network can be used, nn configuration
        # should be defined as example:
        # {[<layer1_input_dim>, <layer1_output_dim>], <layer1_type>, <layer1_parameters>;
        # [<layer2_input_dim>, <layer2_output_dim>], <layer2_type>, <layer2_parameters>;
        # etc.}
        self.nn_config = ''

        # optim_fcn = 'adam' - ADAM optimization function is used;
        # optim_fcn = 'sgd' - SGD optimization function is used;
        # optim_fcn = 'adagrad' - ADAGRAD optimization function is used;
        self.optim_fcn = 'sgd'
        # loss_fcn = 'cross_entropy' - Cross Entropy loss is used;
        # loss_fcn = 'mse' - Mean Square Error loss is used;
        self.loss_fcn = 'cross_entropy'

        # Weight decay regularization parameter
        self.weight_decay = 0.0005
        # Images normalized
        self.images_normalized = True
        # Dataset split
        self.training_dataset = 0.7
        self.validation_dataset = 0.2
        self.test_dataset = 0.1

    def save(self, path, name):
        to_file = pd.DataFrame({
            'NumOfEpochs': self.num_of_epochs,
            'BatchSize': self.batch_size,
            'LearningRate': self.learning_rate,
            'LearningSchedulerStepSize': self.scheduler_step_size,
            'LearningSchedulerGama': self.scheduler_gama,
            'Model': self.model,
            'ModelParameters': self.model_parameters,
            'NNConfig': self.nn_config,
            'OptimFcn': self.optim_fcn,
            'LossFcn': self.loss_fcn,
            'WeightDecay': self.weight_decay,
            'ImagesNormalized': self.images_normalized,
            'TrainingDatasetSize': self.training_dataset,
            'ValidationDatasetSize': self.validation_dataset,
            'TestDatasetSize': [self.test_dataset],
        })
        to_file.T.to_excel('{}/{}.xlsx'.format(path, name),
                           sheet_name='Parameters',
                           index=True,
                           header=False)


if __name__ == '__main__':
    test_param = Parameters()
    test_param.save('./', 'test_param')
