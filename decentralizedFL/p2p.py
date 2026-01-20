import copy
from typing import Sequence
import numpy as np
import torch
from fluke.server import Server
from fluke.client import Client
from fluke.comm import Message
from fluke.utils.model import aggregate_models
from rich.table import Table
from rich.console import Console
from fluke import FlukeENV


class P2PServer(Server):
    def __init__(self, model, test_set, clients, **kwargs):
        """
        adjacency_matrix: dict[int, list[int]] 
                          Ex: {0: [1, 2], 1: [0, 3], ...}
        """
        super().__init__(model, test_set, clients, **kwargs)

        # modifier la matrice d'adjacence ici
        self.adjacency_matrix = {0: [1,2], 1: [2], 2: [1,3,5], 3: [2], 4: [6], 5: [2,6], 6: [5,4]}
        
        # Initialisation : On clone le state_dict initial pour chaque client
        initial_state = model.state_dict()
        self.client_states = {
            client.index: copy.deepcopy(model.state_dict()) # <--- CORRECT : On extrait les poids
            for client in clients
        }

    def broadcast_model(self, eligible):
        """
        Surcharge de la méthode de server.py.
        Au lieu de broadcaster le self.model global, on envoie à chaque client
        sa version spécifique stockée dans self.client_states.
        """
        for client in eligible:
            # 1. Charger l'état spécifique du client dans le conteneur du serveur
            my_state = self.client_states[client.index]
            self.model.load_state_dict(my_state)
            
            # 2. Créer une copie pour l'envoi (important pour éviter les conflits mémoire en simulation)
            # On envoie un objet modèle car client.py attend un objet ayant .state_dict()
            model_to_send = copy.deepcopy(self.model)
            
            # 3. Envoyer en unicast via le channel
            # self.channel.send(
            #     Message(model_to_send, "model", dst=client.index, src="server"), 
            #     dest=str(client.index) # Le channel attend souvent un str ou l'ID direct selon l'implémentation
            # )
            self.channel.broadcast(
                Message(model_to_send, "model", "server"),
                [client.index]
            )

    def aggregate(self, eligible, client_models_gen):
        """
        Logique de Gossip / Consensus.
        1. On récupère les mises à jour des clients qui ont travaillé.
        2. Pour chaque client, on agrège son nouveau modèle avec ceux de ses voisins.
        """

        # conversion du générateur en liste si nécessaire
        client_models = list(client_models_gen)

        # 1. Mise à jour des états LOCAUX
        # ATTENTION : server.fit() envoie des OBJETS modèles, il faut extraire le state_dict
        for client, model_obj in zip(eligible, client_models):
            # C'est ici qu'on évite l'erreur pour le prochain round
            self.client_states[client.index] = copy.deepcopy(model_obj.state_dict())

        # 2. Logique de mélange (Gossip)
        next_round_states = {}
        
        for client in self.clients:
            cid = client.index
            neighbors = self.adjacency_matrix.get(cid, [])
            
            # On récupère les dictionnaires (state_dicts) stockés
            states_to_avg = [self.client_states[cid]]
            for nid in neighbors:
                if nid in self.client_states:
                    states_to_avg.append(self.client_states[nid])
            
            # On fait la moyenne des dictionnaires
            next_round_states[cid] = self._average_state_dicts(states_to_avg)

        self.client_states = next_round_states
        
        # Optionnel : On met à jour self.model avec la moyenne globale juste pour l'évaluation "Server-side"
        # si vous voulez garder une trace de la performance globale du réseau.
        all_states = list(self.client_states.values())
        self.model.load_state_dict(self._average_state_dicts(all_states))

    def _average_state_dicts(self, state_dicts):
        """Fonction utilitaire pour moyenner une liste de state_dicts."""
        if not state_dicts:
            return None
        
        avg_state = copy.deepcopy(state_dicts[0])
        n = len(state_dicts)
        
        for key in avg_state.keys():
            # On gère les tenseurs float, on ignore les LongTensor (ex: num_batches_tracked) souvent non moyennables
            if avg_state[key].is_floating_point():
                avg_state[key] = sum(d[key] for d in state_dicts) / n
                
        return avg_state
    
    def _compute_evaluation(self, round: int, eligible: Sequence[Client]) -> None:
        """
        Surcharge pour afficher les performances individuelles.
        """
        # 1. On garde le comportement standard (Global eval) si désiré
        super()._compute_evaluation(round, eligible)

        # 2. Notre affichage personnalisé
        console = Console()
        table = Table(title=f"📊 Performances P2P - Round {round + 1}")

        table.add_column("Client ID", justify="center", style="cyan")
        table.add_column("Accuracy", justify="right", style="green")
        table.add_column("Loss", justify="right", style="red")

        # On évalue chaque client participant
        evaluator = FlukeENV().get_evaluator()
        
        # Note: server.py utilise eligible, mais pour le P2P on peut vouloir tout le monde
        # Si vous voulez évaluer tout le monde, remplacez 'eligible' par 'self.clients'
        # Attention : évaluer tout le monde est plus long.
        for client in eligible:
            # client.evaluate utilise le modèle local du client
            metrics = client.evaluate(evaluator, self.test_set)
            
            # Protection si l'évaluation échoue ou retourne vide
            acc = metrics.get('accuracy', 0.0)
            loss = metrics.get('loss', 0.0) # Si le loss est calculé par l'évaluateur

            table.add_row(
                str(client.index), 
                f"{acc:.4f}", 
                f"{loss:.4f}" if loss else "N/A"
            )

        console.print(table)

class P2PClient(Client):

    # we override the fit method to implement our training "strategy"
    def fit(self, override_local_epochs: int = 0) -> float:
        # we can override the number of local epochs and call the parent class method
        new_local_epochs = np.random.randint(1, self.hyper_params.local_epochs + 1)
        return super().fit(new_local_epochs)
    
from fluke.algorithms import CentralizedFL

class P2P(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return P2PClient

    def get_server_class(self) -> type[Server]:
        return P2PServer


