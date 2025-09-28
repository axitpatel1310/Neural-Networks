import joblib
from breast_cancer_FNN import model
import torch 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler

torch.save(model.state_dict(),"bc_model.pt")
joblib.dump(scaler,"bc_scaler.pkl")
print("Saved")
