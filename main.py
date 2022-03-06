import torch
from utils import DatasetLoader,init_weights,OneCycle
import random, math, os, pickle, sys
import numpy as np
from tqdm import tqdm
from models.CNNBasic import CNNBasic
from models.Resnet18 import ResNet18
from models.VGG import VGG
from torchvision.transforms import RandomRotation

# Load config file
params = {'weight_decay_rate' : 1e-7,
          'cyc_size' : 70,
          'max_lr' : 5e-3,
          'low_lr' : 1e-5,
          'dec_scale' : 1,
          'batch_size' : 32,
          'tot_epochs' : 90,
          'beta_weight' : 1,
          'heterogeneous' : False,
          'reflection' : True,
          'mode' : 'R4',
          'ratio' : [32,8,2,1],
          'mplc_kernel_size' : [8,4,2,1],
          'dataset' : 'rot_tim',
          'model' : 'Resnet18',
          'training' : True,
          'save_dir':'ien_training'}

# Check if cuda available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using",device,"for training...")

# Determining ChannelsMaxPool Layer characteristics
if params["heterogeneous"]:
    mplc_kernel_size = params["mplc_kernel_size"]
    if sum(mplc_kernel_size)==7 or sum(mplc_kernel_size)==14:
        ratio = [8,2,1]
    elif sum(mplc_kernel_size)==15:
        ratio = [32,8,2,1]
else:
    ratio = None
    if params["reflection"] or params["mode"]=="R8":
        mplc_kernel_size = 8
    elif params["mode"]=="R4":
        mplc_kernel_size = 4
    else:
        raise NameError("Only R4 and R8 modes are allowed")

# Loading dataset and model
print("loading",params["dataset"],"Dataset and",params["model"],"Model...")
if params["dataset"]=="rot_mnist":
    if params["model"]=="BasicCNN":
        model = CNNBasic(10,ratio,mplc_kernel_size).to(device)
        model.apply(init_weights)
        num_layers = 6
        dataset = DatasetLoader(params["dataset"],params["batch_size"],params["training"])
    else:
        raise RuntimeError("Only BasicCNN is allowed with",params["dataset"])
elif params["dataset"]=="rot_tim":
    if params["reflection"]:
        dataset_dir = "ImageNet-rot-ref-masked/"
    else:
        dataset_dir = "ImageNet-rot-masked/"
    if params["model"]=="VGG":
        model = VGG(100, ratio, mplc_kernel_size).to(device)
        num_layers = 8
        dataset = DatasetLoader(dataset_dir,params["batch_size"],params["training"])
    elif params["model"]=="Resnet18":
        model = ResNet18(100, ratio, mplc_kernel_size).to(device)
        num_layers = 4
        dataset = DatasetLoader(dataset_dir,params["batch_size"],params["training"],resize=65)
    else:
        raise RuntimeError("Only VGG and Resnet18 are allowed with",params["dataset"])
else:
    raise NameError("Dataset",params["dataset"],"is not available. Choose from: rot_mnist or rot_tim")

# setting up training parameters
classification_loss = torch.nn.CrossEntropyLoss()
equivariance_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['low_lr'], weight_decay=params['weight_decay_rate'])
onecycle = OneCycle(math.ceil(dataset['train_size']/params['batch_size'])*params['cyc_size'], params['max_lr'], params['low_lr'], params['dec_scale'])

all_rotations = None
if params["mode"]=="R8" or (params["heterogeneous"] and params["mplc_kernel_size"][0]==8):
    all_rotations = [(RandomRotation((x,x)),RandomRotation((-x,-x))) for x in range(45,360,45)]

# Training Loop
best_val_acc = 0
for epoch in range(params['tot_epochs']):
    model.train()
    running_loss1 = 0.0
    running_loss2 = [0.0]*num_layers
    running_corrects = 0
    print("Epoch",epoch,": Starting training...")
    for i,(inputs, labels) in enumerate(tqdm(dataset["train_loader"])):
        inputs = inputs.to(device)
        labels = labels.to(device)
        tbs = labels.shape[0]
        if all_rotations:
            aug_rot = random.choices(all_rotations,k=tbs)
        else:
            aug_rot = random.choices(range(0,4),k=tbs) #(1,4)
        if params["reflection"]:
            aug_ref = random.choices(range(0,3),k=tbs)
        for i in range(tbs):
            if all_rotations:
                augmented_input = aug_rot[i][0](inputs[i])
            else:
                augmented_input = torch.rot90(inputs[i], aug_rot[i], [-2, -1])
            if params["reflection"]:
                if aug_ref[i]==1:
                    augmented_input = torch.flip(augmented_input, (-2,))
                elif aug_ref[i]==2:
                    augmented_input = torch.flip(augmented_input, (-1,))
            inputs = torch.cat((inputs,augmented_input.unsqueeze(0)),0)

        optimizer.zero_grad()
        if epoch<params['cyc_size']:
            lr = onecycle.calc()
            for g in optimizer.param_groups:
                    g['lr'] = lr
        outputs, temparr = model(inputs)

        loss2 = []
        for j in range(num_layers):
            augmented_temp = temparr[j][tbs]
            if params["reflection"]:
                if aug_ref[0]==1:
                    augmented_temp = torch.flip(augmented_temp, (-2,))
                elif aug_ref[0]==2:
                    augmented_temp = torch.flip(augmented_temp, (-1,))
            if all_rotations:
                augmented_temp = aug_rot[0][1](augmented_temp)
            else:
                augmented_temp = torch.rot90(augmented_temp, -aug_rot[0], [-2,-1])
            temp_rot = augmented_temp.unsqueeze(0)
            for i in range(1,tbs):
                augmented_temp = temparr[j][tbs+i]
                if params["reflection"]:
                    if aug_ref[i]==1:
                        augmented_temp = torch.flip(augmented_temp, (-2,))
                    elif aug_ref[i]==2:
                        augmented_temp = torch.flip(augmented_temp, (-1,))
                if all_rotations:
                    augmented_temp = aug_rot[i][1](augmented_temp)
                else:
                    augmented_temp = torch.rot90(augmented_temp, -aug_rot[i], [-2,-1])
                temp_rot = torch.cat((temp_rot, augmented_temp.unsqueeze(0)),0)
            loss2.append(equivariance_loss(temparr[j][:tbs],temp_rot))

        labels = torch.cat((labels,labels))
        _, preds = torch.max(outputs, 1)
        loss1 = classification_loss(outputs, labels)

        for i in range(num_layers):
            loss1+=(params['beta_weight']*loss2[i])
        loss1.backward()
        optimizer.step()
        running_loss1 += loss1.item() * inputs.size(0)
        for i in range(num_layers):
            running_loss2[i] += loss2[i].item() * tbs
        running_corrects += torch.sum(preds == labels.data)
        sys.stdout.flush()

    train_class_loss = running_loss1 / (dataset['train_size']*2)
    train_equi_loss = [running_loss2[i] / (dataset['train_size']) for i in range(num_layers)]
    train_acc = running_corrects.double() / (dataset['train_size']*2)
    train_tot_loss = train_class_loss + sum(train_equi_loss)

    ### Validation
    print("Epoch",epoch,": Starting Validation...")
    model.eval()
    with torch.no_grad():
      val_running_loss1 = 0.0
      val_running_loss2 = [0.0]*num_layers
      val_running_corrects = 0

      for i,(inputs, labels) in enumerate(tqdm(dataset['test_loader'])):
          inputs = inputs.to(device)
          labels = labels.to(device)
          tbs = labels.shape[0]
          if all_rotations:
              aug_rot = random.choices(all_rotations,k=tbs)
          else:
              aug_rot = random.choices(range(0,4),k=tbs) #(1,4)
          if params["reflection"]:
              aug_ref = random.choices(range(0,3),k=tbs)
          for i in range(tbs):
              if all_rotations:
                  augmented_input = aug_rot[i][0](inputs[i])
              else:
                  augmented_input = torch.rot90(inputs[i], aug_rot[i], [-2, -1])
              if params["reflection"]:
                  if aug_ref[i]==1:
                      augmented_input = torch.flip(augmented_input, (-2,))
                  elif aug_ref[i]==2:
                      augmented_input = torch.flip(augmented_input, (-1,))
              inputs = torch.cat((inputs,augmented_input.unsqueeze(0)),0)
          outputs, temparr = model(inputs)
          _, preds = torch.max(outputs, 1)

          loss2 = []
          for j in range(num_layers):
              augmented_temp = temparr[j][tbs]
              if params["reflection"]:
                  if aug_ref[0]==1:
                      augmented_temp = torch.flip(augmented_temp, (-2,))
                  elif aug_ref[0]==2:
                      augmented_temp = torch.flip(augmented_temp, (-1,))
              if all_rotations:
                  augmented_temp = aug_rot[0][1](augmented_temp)
              else:
                  augmented_temp = torch.rot90(augmented_temp, -aug_rot[0], [-2,-1])
              temp_rot = augmented_temp.unsqueeze(0)
              for i in range(1,tbs):
                  augmented_temp = temparr[j][tbs+i]
                  if params["reflection"]:
                      if aug_ref[i]==1:
                          augmented_temp = torch.flip(augmented_temp, (-2,))
                      elif aug_ref[i]==2:
                          augmented_temp = torch.flip(augmented_temp, (-1,))
                  if all_rotations:
                      augmented_temp = aug_rot[i][1](augmented_temp)
                  else:
                      augmented_temp = torch.rot90(augmented_temp, -aug_rot[i], [-2,-1])
                  temp_rot = torch.cat((temp_rot, augmented_temp.unsqueeze(0)),0)
              loss2.append(equivariance_loss(temparr[j][:tbs],temp_rot).item())

          _, preds = torch.max(outputs, 1)
          labels = torch.cat((labels,labels))
          loss1 = classification_loss(outputs, labels)
          val_running_loss1 += loss1.item() * inputs.size(0)
          for i in range(num_layers):
              val_running_loss2[i] += (loss2[i] * tbs)
          val_running_corrects += torch.sum(preds == labels.data)

      val_class_loss = val_running_loss1 / (dataset['test_size']*2)
      val_equi_loss = [val_running_loss2[i] / dataset['test_size'] for i in range(num_layers)]
      val_acc = val_running_corrects.double() / (dataset['test_size']*2)
      val_tot_loss = val_class_loss + sum(val_equi_loss)
      if not os.path.exists(params['save_dir']):
        os.mkdir(params['save_dir'])
      print(f"Epoch {epoch}: Train Acc {train_acc}, Train Total Loss {train_tot_loss}, Class Loss {train_class_loss}, Equi Loss {train_equi_loss} | ; | Val Acc {val_acc}, Val Total Loss {val_tot_loss}, Class Loss {val_class_loss}, Equi Loss {val_equi_loss}")
      torch.save(model.state_dict(),params['save_dir']+"/last_model_weights.ckpt")
      torch.save(optimizer.state_dict(),params['save_dir']+"/last_adam_settings.ckpt")
      if val_acc>best_val_acc:
          best_val_acc = val_acc
          torch.save(model.state_dict(),params['save_dir']+"/best_model_weights.ckpt")
