# File that contains code that can train networks on data.

from .Network.GCNModel import DGCNN
from .Network.DataUtils import *

import torch
from tensorboardX import SummaryWriter
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Network():

    # A Network has a model and data.
    # The model weights are being trained.
    # The data is used for training.
    def __init__(self, model, data):
        if not isinstance(model, DGCNN):
            raise ValueError(f"A Network is instantiated with a model that is not a member of DGCNN.\nCurrent type: {type(model)}")
        if not isinstance(data, DatasetManager):
            raise ValueError(f"A Network is instantiated with a dataset that is not a member of DatasetManager.\nCurrent type: {type(data)}")
        self.model = model
        self.data = data

    # Save model weights to file.
    def saveModel(self, target_path):
        if not type(target_path) == str:
            raise ValueError("target_path is not a string..")
        if not target_path.endswith(".t7"):
            raise ValueError("target_path should end with .t7 to save the current weights with the correct file extension.")
        if os.path.isfile(target_path):
            raise ValueError("Attempting to write over an existing save file.")
        
        # Create non-existing parent directories if needed.
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), target_path)

    # Loads model weights into existing model.
    def loadModel(self, target_path):
        if not type(target_path) == str:
            raise ValueError("target_path should be a string.")
        if not os.path.isfile(target_path):
            raise ValueError("target_path should be a path to a file.")
        if not target_path.endswith(".t7"):
            raise ValueError("target_path should be a path to a .t7 file.")
        
        # This may throw an error if keys of the dictionary are missing or unexpected.
        # There is an option to partly load weights, but for now, I think I want an error to be thrown if the model and the save file don't have the save structure.
        self.model.load_state_dict(torch.load(target_path))
    
    # Train the current model on the given training set.
    def train(self, epochs, model_name, model_save_directory, summary_target_directory, save_epoch_models=False, Adam_learning_rate=0.0001, Adam_betas=(0.9, 0.999), loss_based_on_value_loss=1.):
        if type(epochs) != int or epochs < 1:
            raise ValueError(f"epochs should be an integers higher than zero being the amount of epochs you want to train. Currently it is {epochs}")
        if type(Adam_learning_rate) != float or Adam_learning_rate**2 > 1:
            raise ValueError(f"Adam_learning_rate should be a number between 0 and 1. Currently it is {Adam_learning_rate}")
        if type(Adam_betas) != tuple or len(Adam_betas) != 2 or any([type(x) != float for x in Adam_betas]) or any([x**2 > 1 for x in Adam_betas]) or Adam_betas[0] > Adam_betas[1]:
            raise ValueError(f"Adam_betas should be a tuple of 2 floats between 0 and 1 with the first value lower than the second value. Currently it is {Adam_betas}")
        if type(model_name) != str or not model_name.endswith(".t7"):
            raise ValueError(f"model_name should be a string with the .t7 extension. Currently it is {model_name}")
        if type(model_save_directory) != str or not os.path.isdir(model_save_directory):
            raise ValueError(f"model_save_directory should be a string targetting a file on the machine that does not exist yet with the .t7 extension. Currently it is {model_save_directory}")
        if os.path.isfile(model_save_directory + "/" + model_name):
            raise ValueError(f"model save path already contains a saved model! Make sure not to overwrite existing models!")
        if type(summary_target_directory) != str or not os.path.isdir(summary_target_directory):
            raise ValueError(f"summary_target_directory should be a string targetting an existing directory on the machine. Currently it is {summary_target_directory}")
        if type(save_epoch_models) != bool:
            raise ValueError(f"save_epoch_models should be True or False. Currently it is {save_epoch_models}")

        k_loss_writer = SummaryWriter(summary_target_directory)

        print('Loading data...')
        
        train_data_loader = self.data.getTrainingSet()
        num_train_batch = len(train_data_loader.dataset) / self.data.batch_size

        val_data_loader = self.data.getValidationSet()
        num_val_batch = len(val_data_loader.dataset) / self.data.batch_size

        # initialize Network structure etc.
        dgcnn = self.model
        # dgcnn = torch.nn.DataParallel(dgcnn)
        optimizer = torch.optim.Adam(dgcnn.parameters(), lr=Adam_learning_rate, betas=Adam_betas)
        dgcnn.cuda()

        cos_target = torch.tensor(np.ones((self.data.batch_size)))
        cos_target = cos_target.type(torch.FloatTensor).cuda()
        weight_alpha = 1 - loss_based_on_value_loss
        weight_beta = loss_based_on_value_loss

        print('Start training...')
        last_val_cos_loss = 999.
        last_val_value_loss = 999.
        for epoch in range(epochs):
            for i_train, data in enumerate(train_data_loader):
                inputs, gt_norm = data
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.permute(0, 2, 1)
                gt_norm = gt_norm.type(torch.FloatTensor)

                inputs = inputs.cuda()
                gt_norm = gt_norm.cuda()

                optimizer.zero_grad()
                dgcnn = dgcnn.train()

                output = dgcnn(inputs)

                cos_loss = torch.nn.functional.cosine_embedding_loss(output, gt_norm, cos_target)
                value_loss = torch.nn.functional.mse_loss(output, gt_norm)

                if(i_train % 100 == 0):
                    k_loss_writer.add_scalar('cos_loss', cos_loss, global_step=epoch * num_train_batch + i_train + 1)
                    k_loss_writer.add_scalar('value_loss', value_loss, global_step=epoch * num_train_batch + i_train + 1)

                loss = weight_alpha * cos_loss + weight_beta * value_loss
                loss.backward()
                optimizer.step()

                print("Epoch: %d/%d, || Batch: %d/%d, || cos loss: %.7f, || value loss: %.7f, || val cos loss: %.7f || val value loss: %.7f" % \
                    (epoch+1, epochs, i_train + 1, num_train_batch, cos_loss.data.item(), value_loss.data.item(), last_val_cos_loss, last_val_value_loss))
            
            #______Validation______
            
            val_cos_loss = []
            val_value_loss = []
            val_loss = []
            dgcnn.eval()
            for i_val, data in enumerate(val_data_loader):
                inputs, _, gt_norm, _ = data
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.permute(0, 2, 1)
                gt_norm = gt_norm.type(torch.FloatTensor)

                inputs = inputs.cuda()
                gt_norm = gt_norm.cuda()

                output = dgcnn(inputs)

                cos_loss = torch.nn.functional.cosine_embedding_loss(output, gt_norm, cos_target)
                value_loss = torch.nn.functional.mse_loss(output, gt_norm)

                loss = weight_alpha * cos_loss + weight_beta * value_loss

                val_loss.append(loss.data.item())
                val_cos_loss.append(cos_loss.data.item())
                val_value_loss.append(value_loss.data.item())

                print("Epoch: %d/%d, || Val Batch: %d/%d, || cos loss: %.7f, || value loss: %.7f" % \
                    (epoch+1, epochs, i_val + 1, num_val_batch, cos_loss.data.item(), value_loss.data.item()))
            
            val_cos_loss = np.array(val_cos_loss)
            val_value_loss = np.array(val_value_loss)
            val_loss = np.array(val_loss)

            last_val_cos_loss = np.mean(val_cos_loss)
            last_val_value_loss = np.mean(val_value_loss)
            
            if save_epoch_models:
                self.model = dgcnn
                self.saveModel(model_save_directory + f"/{int(time.time()) % 1000000}_{round(last_val_cos_loss, 4)}_{epoch}.t7")

            k_loss_writer.add_scalar('val_cos_loss', last_val_cos_loss, global_step=epoch + 1)
            k_loss_writer.add_scalar('val_value_loss', last_val_value_loss, global_step=epoch + 1)
        
        self.model = dgcnn
        self.saveModel(model_save_directory + "/" + model_name)

    # Validate the current model with the validation set.
    def test(self, loss_based_on_value_loss=1.):
        if not isinstance(loss_based_on_value_loss, (float, int)):
            raise ValueError(f"loss_based_on_value_error should be a float or integer between 0 and 1 representing a percentage. Currently it is {loss_based_on_value_loss}")
        
        weight_alpha = 1 - loss_based_on_value_loss
        weight_beta = loss_based_on_value_loss

        dgcnn = self.model
        dgcnn.cuda()
        dgcnn.eval()

        cos_target = torch.tensor(np.ones((self.data.batch_size)))
        cos_target = cos_target.type(torch.FloatTensor).cuda()

        test_data_loader = self.data.getValidationSet()
        
        val_cos_loss = []
        val_value_loss = []
        val_loss = []
        for i_test, data in enumerate(test_data_loader):
            inputs, gt_norm = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.permute(0, 2, 1)
            gt_norm = gt_norm.type(torch.FloatTensor)

            inputs = inputs.cuda()
            gt_norm = gt_norm.cuda()

            output = dgcnn(inputs)

            cos_loss = torch.nn.functional.cosine_embedding_loss(output, gt_norm, cos_target)
            value_loss = torch.nn.functional.mse_loss(output, gt_norm)

            loss = weight_alpha * cos_loss + weight_beta * value_loss

            val_loss.append(loss.data.item())
            val_cos_loss.append(cos_loss.data.item())
            val_value_loss.append(value_loss.data.item())

            print("Val Batch: %d/%d, || cos loss: %.7f, || value loss: %.7f" % \
                    (i_test + 1, len(test_data_loader.dataset) / self.data.batch_size, cos_loss.data.item(), value_loss.data.item()))
        
        # WIP Needs to forward input graphs through the given model, such that the output can be used for vertex updating.
        def forward(self, inputs):
            return

# A class with which you can create settings to configure training.
class NetworkSettings():
    def __init__(self):
        self.settings = None
