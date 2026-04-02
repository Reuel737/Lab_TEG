import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = [
    # "historico/hist_kfold_exp08.csv",
    "historico/hist_kfold_exp12.csv",
    # "historico/hist_kfold_exp14.csv",
    "historico/hist_kfold_exp15.csv",
    "historico/hist_kfold_exp16.csv"
]

def analisar(file):
    print(f"\n===== {file} =====")
    
    df = pd.read_csv(file)
    
    # Selecionar colunas
    train = pd.to_numeric(df["train_loss_final"], errors='coerce').dropna()
    val = pd.to_numeric(df["val_loss_final"], errors='coerce').dropna()
    
    # Estatísticas
    print("\n--- TRAIN ---")
    print(f"Média: {np.mean(train):.4f}")
    print(f"Desvio: {np.std(train, ddof=1):.4f}")
    
    print("\n--- VAL ---")
    print(f"Média: {np.mean(val):.4f}")
    print(f"Desvio: {np.std(val, ddof=1):.4f}")
    
    # Overfitting (diferença média)
    diff = val - train
    print("\n--- OVERFITTING ---")
    print(f"Média (val - train): {np.mean(diff):.4f}")
    
    return train, val


all_train = []
all_val = []
nomes = ["exp12", "exp15", "exp16"]

for f in files:
    train, val = analisar(f)
    all_train.append(train)
    all_val.append(val)


# Boxplot comparação VAL
plt.figure()
plt.boxplot(all_val, labels=nomes)
plt.title("Comparação VAL LOSS")
plt.ylabel("Loss")
plt.show()

# Boxplot comparação TRAIN
plt.figure()
plt.boxplot(all_train, labels=nomes)
plt.title("Comparação TRAIN LOSS")
plt.ylabel("Loss")
plt.show()

# Scatter train vs val (visual de overfitting)
for i in range(len(files)):
    plt.scatter(all_train[i], all_val[i], label=nomes[i])

plt.xlabel("Train Loss")
plt.ylabel("Val Loss")
plt.title("Train vs Val")
plt.legend()
plt.show()