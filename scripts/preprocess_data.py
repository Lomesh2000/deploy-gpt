from src.utils.config import load_config
from pathlib import Path

def main():
    config = load_config("configs/base.yaml")
    print("Loaded configuration:", config)

    val_split = config['training']['val_split']
    print(f"Validation split ratio: {val_split}")

    paths = config['paths']
    
    with open(paths['raw_data'], 'r') as f:
        text = f.read()

    split_index = int(len(text) * (1 - val_split))
    train_text = text[:split_index]
    val_text = text[split_index:]

    Path(paths['train_data']).parent.mkdir(parents=True, exist_ok=True)
    Path(paths['val_data']).parent.mkdir(parents=True, exist_ok=True)

    with open(paths['train_data'], 'w') as f:
        f.write(train_text)

    with open(paths['val_data'], 'w') as f:
        f.write(val_text)

    print("Train and Validation split completed and files created")

if __name__ == "__main__":
    main()



