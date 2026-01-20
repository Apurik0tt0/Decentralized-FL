import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DONNÉES (Simulées pour l'exemple)
# ==========================================
# Dans votre projet, remplacez ces listes par les sorties de Fluke.
rounds = np.arange(1, 8)  # 7 rounds (de 1 à 7)

# Données simulées pour FedAvg (Global Accuracy & Loss)
fedavg_acc = [0.15, 0.28, 0.45, 0.55, 0.62, 0.68, 0.71]
fedavg_loss = [2.3, 1.9, 1.5, 1.1, 0.9, 0.75, 0.65]

# Données simulées pour la figure 4 (Précision locale des clients par round)
# Format : Une liste de listes. Chaque sous-liste contient les précisions des clients à ce round.
# Imaginons 10 clients.
client_accuracies_history = [
    np.random.normal(0.15, 0.05, 10), # Round 1 (moyenne 0.15, écart-type 0.05)
    np.random.normal(0.28, 0.06, 10), # Round 2
    np.random.normal(0.45, 0.08, 10), # Round 3
    np.random.normal(0.55, 0.10, 10), # Round 4
    np.random.normal(0.62, 0.11, 10), # Round 5
    np.random.normal(0.68, 0.09, 10), # Round 6
    np.random.normal(0.71, 0.05, 10)  # Round 7
]

# ==========================================
# 2. FONCTIONS DE TRACÉ
# ==========================================

def plot_accuracy_convergence(rounds, acc_data, algo_name="FedAvg"):
    """Figure 1 : Courbe de convergence (Précision vs Rounds)"""
    plt.figure(figsize=(10, 6))
    
    # Trace la courbe
    plt.plot(rounds, acc_data, marker='o', linestyle='-', linewidth=2, label=algo_name, color='b')
    
    # Mise en forme
    plt.title(f"Figure 1 : Convergence de la Précision ({algo_name})", fontsize=14)
    plt.xlabel("Rounds de Communication", fontsize=12)
    plt.ylabel("Précision Globale (Test Accuracy)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0, 1) # Précision entre 0 et 1
    
    # Pour ajouter d'autres algos plus tard, il suffira d'ajouter plt.plot(...) ici
    plt.show()

def plot_loss_stability(rounds, loss_data, algo_name="FedAvg"):
    """Figure 2 : Stabilité de l'apprentissage (Loss vs Rounds)"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(rounds, loss_data, marker='x', linestyle='--', linewidth=2, label=algo_name, color='r')
    
    plt.title(f"Figure 2 : Évolution de la Loss Globale ({algo_name})", fontsize=14)
    plt.xlabel("Rounds de Communication", fontsize=12)
    plt.ylabel("Loss (Perte)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.show()

def plot_final_comparison(algo_names, final_accuracies):
    """Figure 3 : Comparaison finale (Bar Chart) - Exemple avec 1 seul algo pour l'instant"""
    plt.figure(figsize=(8, 6))
    
    bars = plt.bar(algo_names, final_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    plt.title("Figure 3 : Précision Finale (Round 7)", fontsize=14)
    plt.ylabel("Précision", fontsize=12)
    plt.ylim(0, 1)
    
    # Ajout des valeurs au dessus des barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')
        
    plt.show()

def plot_client_distribution(client_data_history, rounds):
    """Figure 4 : Distribution des clients (Boxplot)"""
    plt.figure(figsize=(10, 6))
    
    # Création du boxplot
    plt.boxplot(client_data_history, labels=rounds, patch_artist=True, 
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="red"))
    
    plt.title("Figure 4 : Distribution des performances locales des clients (FedAvg)", fontsize=14)
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Précision Locale des Clients", fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

# ==========================================
# 3. EXÉCUTION DES TRACÉS
# ==========================================

# Tracer Figure 1
plot_accuracy_convergence(rounds, fedavg_acc)

# Tracer Figure 2
plot_loss_stability(rounds, fedavg_loss)

# Tracer Figure 3 (On simule d'autres algos à 0 pour l'instant pour montrer l'exemple)
algos = ['FedAvg', 'FedProx', 'SCAFFOLD', 'FedAdam', 'FedNova']
scores = [fedavg_acc[-1], 0, 0, 0, 0] # On prend la dernière valeur de FedAvg
plot_final_comparison(algos, scores)

# Tracer Figure 4
plot_client_distribution(client_accuracies_history, rounds)