# RF-GNN
RF-GNN: Random Forest Boosted Graph Neural Network for Social Bot Detection

# Environment Settings 
* python == 3.7   
* torch == 1.8.1+cu102	  
* numpy == 1.21.6  
* scipy == 1.7.2  
* pandas == 1.3.5	
* scikit-learn == 1.0.2	 
* torch-cluster == 1.5.9	
* torch-geometric == 2.0.4	
* torch-scatter == 2.0.8	
* torch-sparse ==	0.6.12	
* torch-spline-conv	== 1.2.1	


# Usage 

### RF-GNN

* **dataset**: including \[MGTAB, Twibot20, Cresci15\].  
* **model**: including \['GCN', 'GAT', 'SAGE', 'RGCN', 'SGC'\].  
* **labelrate**: parameter for labelrate. (default = 0.1)

e.g.
````
#run RF-GCN on MGTAB (label rate 0.05)
python RF-GNN.py -dataset MGTAB -model GCN --labelrate 0.05
#run RF-GAR on Twibot-20
python RF-GNN.py -dataset Twibot20 -model GAT -smote True
````


### RF-GNN-E and GNN

* **dataset**: including \[MGTAB, Twibot20, Cresci15\].  
* **model**: including \['GCN', 'GAT', 'SAGE', 'RGCN', 'SGC'\].  
* **ensemble**: including \[True, False\].  
* **labelrate**: parameter for labelrate. (default = 0.1)

e.g.
````
#run RF-GCN-E on MGTAB
python GNN.py -dataset MGTAB -model GCN -ensemble True
#run GCN on MGTAB
python GNN.py -dataset Cresci15 -model GCN -ensemble False
````


# Dataset

For TwiBot-20, please visit the [Twibot-20 github repository](https://github.com/BunsenFeng/TwiBot-20).
For MGTAB please visit the [MGTAB github repository](https://github.com/GraphDetec/MGTAB).
For Cresci-15 please visit the [Twibot-20 github repository](https://github.com/GraphDetec/MGTAB).


We also offer the processed data set: [Cresci-15](https://drive.google.com/uc?export=download&id=13J-UkHZ6tuZedOI0RUgEoHiMIJRGAdNC), [MGTAB](https://drive.google.com/uc?export=download&id=1XfLYIz4M3KPnVpsEUwRMddSs548y29a5), [Twibot-20](https://drive.google.com/uc?export=download&id=1VtpWZzzRyze_5xIy2f1T6jV5lzyj1Oc9).

