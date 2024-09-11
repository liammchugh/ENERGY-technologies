import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import math
import time
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if multiple GPUs, use cuda:0, cuda:1, etc.
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'Number of available GPUs: {torch.cuda.device_count()}')
    print(f'Current CUDA device: {torch.cuda.current_device()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(device)}')
    print(f'CUDA device properties: {torch.cuda.get_device_properties(device)}')
else:
    print("CUDA is not available. Using CPU.")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)

        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# class PositionalEncoding(nn.Module): # daylong sinusoid encoding
#     def __init__(self, d_model, max_seq_length):
#         super(PositionalEncoding, self).__init__()
#         self.d_model = d_model
#         self.max_seq_length = max_seq_length
        
#     def forward(self, x):
#         batch_size, seq_length, _ = x.size()
#         # Create positional encoding based on time of day
#         time_hours = x[:, :, 0]  # Extract the time in hours from the first column of the source data
#         daily_pe = torch.sin(2 * np.pi * time_hours / 24).unsqueeze(-1).repeat(1, 1, self.d_model // 2)
#         daily_pe = torch.cat((daily_pe, torch.cos(2 * np.pi * time_hours / 24).unsqueeze(-1).repeat(1, 1, self.d_model // 2)), dim=-1)
#         pe = daily_pe.to(x.device)

#         return x + pe

class PositionalEncoding(nn.Module): # varying freq. relative positional encoding
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # x = (x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        # x = (x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # attn_output = self.self_attn(x, x, x, tgt_mask)
        # x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.norm1(self.feed_forward(x))
        attn_output = self.cross_attn(ff_output, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, input_dim, output_dim):
        super(Transformer, self).__init__()
        self.encoder_input_linear = nn.Linear(input_dim, d_model)
        self.decoder_input_linear = nn.Linear(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        
        src_embedded = self.encoder_input_linear(src)
        src_embedded = self.dropout(self.positional_encoding(src_embedded))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # DECODER BLOCK
        tgt_embedded = self.decoder_input_linear(tgt)
        tgt_embedded = self.dropout((tgt_embedded))

        dec_output = tgt_embedded   
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output) #

        return output


def count_parameters(model: torch.nn.Module) -> int:
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """ Returns the number of learnable parameters for a PyTorch model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

## Padding and Masking - Training Data Prep ##
def pad_sequences(sequences, max_length):
    padded_sequences = torch.zeros(len(sequences), max_length, sequences[0].size(1))
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        padded_sequences[i, :length] = seq
    return padded_sequences

def generate_masks(src, tgt):
    src_mask = (src.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones((1, seq_length, seq_length), device=tgt.device), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask

# Custom Dataset to handle varying-length sequences
class TimeSeriesDataset(data.Dataset):
    def __init__(self, data, target):
        self.data = [torch.tensor(d.values, dtype=torch.float32) for d in data]
        self.target = [torch.tensor(t.values, dtype=torch.float32) for t in target]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    @staticmethod
    def collate_fn(batch):
        data, target = zip(*batch)
        max_length = max(seq.size(0) for seq in data)
        padded_data = pad_sequences(data, max_length)
        padded_target = pad_sequences(target, max_length)
        return padded_data, padded_target
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0.00003):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        self.best_loss = val_loss
        torch.save(model.state_dict(), 'Xfrmr_checkpoint.pt')

# Function to generate model predictions
def predict(model, src_batch, max_seq_length, start_token=None):
    model.eval()
    src_batch = src_batch.to(device)
    src_mask, _ = generate_masks(src_batch, src_batch)
    src_mask = src_mask.to(device)
    batch_size = src_batch.size(0)
    tgt_dim = model.fc_out.out_features

    # Initialize decoder input with zeros (or use a start token)
    if start_token == None:
        tgt_input = torch.zeros(batch_size, 1, tgt_dim).to(device)
    else:
        tgt_input = start_token.repeat(batch_size, 1, 1).to(device)
    
    outputs = []

    for _ in range(max_seq_length):
        tgt_mask = (tgt_input.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask & (1 - torch.triu(torch.ones((1, tgt_input.size(1), tgt_input.size(1)), device=tgt_input.device), diagonal=1)).bool()

        output = model(src_batch, tgt_input, src_mask, tgt_mask)
        outputs.append(output[:, -1:, :].cpu().numpy())

        # Append the latest output to tgt_input for the next step
        tgt_input = torch.cat([tgt_input, output[:, -1:, :]], dim=1)

    outputs = np.concatenate(outputs, axis=1)
    return outputs

# Convert DataFrames to lists of DataFrames
def convert_to_lists_of_dfs(sample_df_list, target_df_list, zero_indices):
    X_list = []
    y_list = [] 
    f = 0
    samples = len(zero_indices)
    print('#Samples:', samples)
    for k in range(samples-1):
        if k == samples-1:
            sample_x = sample_df_list.iloc[zero_indices[k]:, :]
            sample_y = target_df_list.iloc[zero_indices[k]:, :]
        else:
            sample_x = sample_df_list.iloc[zero_indices[k]:zero_indices[k+1], :]
            sample_y = target_df_list.iloc[zero_indices[k]:zero_indices[k+1], :]
        # Filter for infeasable data
        if sample_x.iloc[:, 7].min() < -2000 or sample_x.iloc[:, 7].max() > 325000:
            f+=1
        else:
            X_list.append(sample_x)
            y_list.append(sample_y) 
    print(f"{f} Samles Breach Feasibility Bounds")
    return X_list, y_list

def getloss(model, criterion, dataloader):
    model.eval() # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            src_mask, tgt_mask = generate_masks(src_batch, tgt_batch)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
            output = model(src_batch, tgt_batch[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
            loss = criterion(output, tgt_batch[:, 1:]) #FOR DECODER USE SHIFTED SEQ 1:
            val_loss += loss.item()
    return val_loss

def model_params_match(model, Xfrmr_path):
    checkpoint = torch.load(Xfrmr_path)
    model_state_dict = model.state_dict()
    # Compare the shapes of the parameters
    for name, param in checkpoint.items():
        if name not in model_state_dict:
            return False
        if model_state_dict[name].shape != param.shape:
            return False
    return True



if __name__ == "__main__":
    #---------- Data Initialization --------------------------------------------------------------------------
    import pandas as pd
    import os
    from matplotlib import pyplot as plt
    
    training_path = r"C:\\" # Path to the training data
    # make sure there are no commas in the data file
    data = pd.read_csv(training_path) # Read CSV file into pandas dataframes
    data.columns = data.columns.str.lower() # Convert column names to lowercase
    columns_to_drop = ['date', 'dateStr', 'zip_code'] # Drop columns that cant be used in math operations
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)
    data.fillna(data.mean(), inplace=True)
    # print(data.head())

    weather_inputs = ['Time', 'OutsideTemperature', 'Velocity', 'Cloudiness', 'TotalSolarRadiationHorizontal']
    energy_inputs = ['InletTemperature', 'OutletTemperature', 'QHeatPowerHEAT', 'HeatEnergy_Cumulat'] #removed VolumeFlowHEAT
    parameter_inputs = ['year', 'Latitude', 'Longitude', 'floors', 'height', 'length', 'width']
    window_inputs = ['heightwindows', 'widthwindows', 'N_WindNum', 'E_WindNum', 'W_WindNum', 'S_WindNum']
    boundary_inputs = ['N_Gamma', 'SharedWalls','N_wallshare', 'E_wallshare', 'W_wallshare', 'S_wallshare']
    data_inputs = [input.lower() for input in weather_inputs + energy_inputs + parameter_inputs + window_inputs + boundary_inputs]

    # Select the relevant columns from the automated data
    X = data[data_inputs]
    X_features = X.columns.tolist()
    # print(X.head())

    power = data['qheatpowerheat']
    print('POWERMAX:', max(power))
    print('POWERMIN', min(power))

    target_columns = ['gValueWindows', 
                      'ULambdaExWindows', 'ULambdaExWalls', 'ULambdaGrdFloor', 'ULambdaRoof', 
                      'LambdaExWalls',	'LambdaGrdFloor',	'LambdaRoof',	
                      'ThicknessExWalls',	'ThicknessGrdFloor',	'ThicknessRoof', 
                      'RadiatorSize', 'AirInfiltration'] 
    target_columns = [col.lower() for col in target_columns]
    y = data[target_columns]
    # print(y.head())

    times = data['time'].values
    zero_indices = np.where(times == 0)[0]
    X_list, y_list = convert_to_lists_of_dfs(X, y, zero_indices)

    # Create Dataset and DataLoader
    train_data, val_data, train_target, val_target = train_test_split(X_list, y_list, test_size=0.2, random_state=42) # random_state=42

    #working on normalized dataset. remember to denormalize with trained medians
    # import Data_PreProcess as prep
    # train_data, X_med = prep.medNormalizeData(train_data)
    # val_data, _ = prep.medNormalizeData(val_data, X_med)
    # train_target, y_med = prep.medNormalizeData(train_target)
    # val_target, _ = prep.medNormalizeData(val_target, y_med)
    
    train_dataset = TimeSeriesDataset(train_data, train_target)
    val_dataset = TimeSeriesDataset(val_data, val_target)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, collate_fn=TimeSeriesDataset.collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, collate_fn=TimeSeriesDataset.collate_fn)

    for i, (src_batch, tgt_batch) in enumerate(train_dataloader):
        print(f"Batch {i+1}")
        break  # Only inspect the first batch
        ## CREATE AND TRAIN MODEL ##


    ########## Model parameters ##########
    input_dim = len(data_inputs)  # Number of input features
    output_dim = len(target_columns)  # Number of output features
    print('Input Dimension:', input_dim, 'Output Dimension:', output_dim)
    
    d_model = 16 # Embedded information dimensionality (ex GPT-4: 4k-6k for token embedding. 64-128 for timeseries phys.; Foumani et al. 2021)
    num_heads = 2   # Number of attention heads. Must divide d_model evenly
    num_layers = 4  # Number of encoder/decoder layers
    d_ff = 32
    dropout = 0.1
    max_seq_length = 73*365  # GENERATE YEARS WORTH DATA FOR INPUT!! MAX should be large enough to handle maximum sequence length in the dataset
    model = Transformer(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, input_dim, output_dim)
    params = count_parameters(model)
    print(f"Total number of learnable parameters in the model: {params}")
    model.to(device)
    
    # Path to the pretrained model
    Xfrmr_path = 'transformer' + '_' + str(input_dim) + '_' + str(output_dim) + '.pth'
    proceed = 'y' # Initialize Training Procession
    
    # Load the pretrained model if parameters match
    if os.path.isfile(Xfrmr_path):
        if model_params_match(model, Xfrmr_path):
            print("Pretrained model found with matching I/O count..")
            new = input("Create New Model?(y/n): ")
            if new.lower() == 'n':
                model.load_state_dict(torch.load(Xfrmr_path))
                proceed = input("Proceed with Training?(y/n): ")
            else:
                id = time.time()
                print("Creating new model...")
                import datetime
                current_time = datetime.datetime.now()
                unique_id = current_time.strftime("%H%M")
                print("Unique ID:", unique_id)
        else:
            print("Different I/O count than pretrained model. Creating new model...")
    else:
        print("Pretrained model not found. Creating new model...")

    criterion = RMSELoss() # Can use nn.MSELoss(), but RMSE seems to have less of a zero-gradient problem with this dataset.
    optimizer = optim.Adam(model.parameters(), lr=0.09, betas=(0.9, 0.98), eps=1e-9)

    ##### Training Loop #####
    num_epochs = 6000
    estop_patience=50

    import threading
    def set_proceed():
        global proceed
        proceed = "y"
    timer = threading.Timer(60.0, set_proceed) # If no response, proceed with training
    timer.start()
    timer.cancel()

    if proceed.lower() == 'y':
        print("Training Model...")
        model.train() # Set model to training mode
        loss_history = []
        val_hist = []
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, threshold=0.001)
        early_stopping = EarlyStopping(patience=estop_patience, verbose=False)
        lr = optimizer.param_groups[0]['lr']
        for epoch in range(num_epochs):
            start_time = time.time()
            loaderloss = 0.0
            for src_batch, tgt_batch in train_dataloader:
                src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
                src_mask, tgt_mask = generate_masks(src_batch, tgt_batch)
                src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
                
                optimizer.zero_grad()
                output = model(src_batch, tgt_batch[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
                loss = criterion(output, tgt_batch[:, 1:])
                loss.backward()
                optimizer.step()
                loaderloss += loss.item()
            epoch_loss = loaderloss / len(train_dataloader) #avg loss over dataset
            loss_history.append(epoch_loss)
            step_time = time.time() - start_time
            print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, StepTime[s]: {step_time:.4f}")
            
            # Step the scheduler
            scheduler.step(epoch_loss)
            if optimizer.param_groups[0]['lr'] != lr:
                lr = optimizer.param_groups[0]['lr']
                print(f"LEARNING RATE: {lr}")
        
            # Validation loss comparator. Consider multi-thread instead of 1/5 epochs
            if val_dataloader is not None:
                val_loss = getloss(model, criterion, val_dataloader)
                val_hist.append(val_loss)
                if epoch % 15 == 0:
                    print(f"Validation Loss: {val_loss:.4f}")
                early_stopping(val_loss, model) # Check if model still improving
                if early_stopping.early_stop:
                    print("Early stopping")
                    torch.save(model.state_dict(), 'best_model.pt')
                    break

        val_loss = getloss(model, criterion, val_dataloader)
        print(f"Validation Loss: {val_loss}")

        # Plotting loss history
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        epoch = [i for i in range(len(loss_history))]
        scale = len(loss_history)/len(val_hist)
        valep = [i*scale for i in range(len(val_hist))]
        ax.plot(valep, val_hist)
        ax.plot(epoch, loss_history)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend(['Validation Loss', 'Training Loss'])
        ax.set_title('Training Loss History')
        print("Saving model...")
        model.to()
        torch.save(model.state_dict(), Xfrmr_path)
        model.eval() # Set model to evaluation mode for analysis
    else:
        print("Skipping Training.. using pretrained model.")
        model = Transformer(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, input_dim, output_dim)   # Define model architecture
        model.load_state_dict(torch.load(Xfrmr_path))     # Load the state dictionary into the model
        model.to(device)    # Move the model to the appropriate device
        # Set the model to evaluation mode
        model.eval()

        
    #---------- Model Evaluation --------------------------------------------------------------------------
    print("Model Evaluation...")
    
    # Function to plot predictions vs targets for each dimension
    def plot_predictions_vs_targets(predictions, targets, varnames):
        num_dimensions = predictions.shape[-1]
        plt.figure(figsize=(15, 10))
        for i in range(num_dimensions):
            plt.subplot(int(np.ceil(num_dimensions/3)), 3, i + 1)
            plt.plot(predictions[:, i], color='blue', label='Model Output')
            plt.plot(targets[:, i], color='red', label='Validation Target', linewidth=3.0)
            plt.xlabel('Look-Ahead Cycle')
            plt.ylabel(varnames[i])
            # Dynamically set the x-axis limits to the range of x plus some padding
            padding = 0.25
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            plt.ylim(min_val*padding, max_val*(1+padding))
            plt.legend()
        plt.tight_layout()

    # Preselect a single sample from the validation set
    index = 0  # Choose the index of the sample you want to use for prediction and plotting
    src_sample, tgt_sample = val_dataset[index]
    src_sample = src_sample.unsqueeze(0).to(device)  # Add batch dimension and move to device
    tgt_sample = tgt_sample.unsqueeze(0)  # Add batch dimension

    # Custom start token array (use your actual values instead of placeholder values)
    start_token = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # Match output_dim
    start_token = start_token.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions


    # Generate prediction for the single sample
    with torch.no_grad():
        max_seq_length = src_sample.size(1)
        predictions = predict(model, src_sample, max_seq_length, start_token=start_token)

    # Extract predictions and targets for the selected sample
    predictions = predictions.squeeze(0)  # Remove batch dimension
    targets = tgt_sample.squeeze(0).numpy()  # Remove batch dimension and convert to numpy

    # Plot predictions vs targets for each dimension
    plot_predictions_vs_targets(predictions, targets, target_columns)
    
    
    # Function to plot scatter charts for predictions vs targets for each dimension
    def plot_scatter_predictions_vs_targets(predictions, targets, varnames):
        num_dimensions = predictions.shape[-1]
        num_columns = 3
        num_rows = int(np.ceil(num_dimensions / num_columns))
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i in range(num_dimensions):
            ax = axes[i]
            ax.scatter(targets[:, i], predictions[:, i], color='blue', alpha=0.5)
            ax.set_xlabel('Validation Target')
            ax.set_ylabel('Model Prediction')
            ax.set_title(varnames[i])
            # Diagonal line indicating perfect prediction
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            ax.plot([targets[:, i].min(), targets[:, i].max()], [targets[:, i].min(), targets[:, i].max()], 'r--')  # Diagonal line
        # Remove any unused subplots
        for j in range(num_dimensions, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        

    # Assuming varnames is a list of names for each dimension
    varnames = target_columns  # or any other list of variable names you have

    # Generate predictions for the validation set
    val_outputs = []
    val_targets = []

    with torch.no_grad():
        for src_batch, tgt_batch in val_dataloader:  # We need both src_batch and tgt_batch for plotting
            src_batch = src_batch.to(device)
            max_seq_length = src_batch.size(1)  # Or set your own max sequence length for prediction
            output = predict(model, src_batch, max_seq_length)
            val_outputs.append(output)
            val_targets.append(tgt_batch.numpy())

    val_outputs = np.concatenate(val_outputs, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    # print('Validation outputs:', val_outputs)
    # print('Validation targets:', val_targets)

    # Plot scatter charts for each dimension
    plot_scatter_predictions_vs_targets(val_outputs, val_targets, varnames)
    plt.show()