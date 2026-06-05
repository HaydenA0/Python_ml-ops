"""Entrypoint for training the stacked ensemble (base models + meta-learner)."""
from python.pcxp_mlops.training import train_ensemble

if __name__ == "__main__":
    train_ensemble()
