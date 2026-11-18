from pathlib import Path
from trainer import train_model

HERE = Path(__file__).parent

if __name__ == "__main__":
    train_model(
        csv=HERE / "csv" / "cyp3a4_veith.csv",
        out=HERE / "models" / "cyp3a4_gcn.pt",
        hidden=128,
        layers=3,
        dropout=0.2,
        lr=1e-3,
        weight_decay=0.3,
        batch_size=128,
        epochs=150,
        patience=25,
        frac_train=0.7,
        frac_val=0.2,
        seed=42,
        print_every=10,
        use_pos_weight=False,
    )
