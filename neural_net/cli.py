import sys
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

def print_banner():
    print("==========================================")
    print("   NEURAL NET CLI - RAP BATTLE ENGINE     ")
    print("==========================================")
    print(" Commands:")
    print("   train   - Train the model on dummy data")
    print("   test    - Run a sample prediction")
    print("   exit    - Quit the CLI")
    print("------------------------------------------")

def main():
    model = SimpleNN()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print_banner()

    while True:
        try:
            choice = input("neural-cli> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if choice == "train":
            print("[*] Training model for 100 epochs...")
            X = torch.randn(200, 5)
            y = torch.randint(0, 2, (200, 1)).float()

            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            print(f"[+] Training complete! Final Loss: {loss.item():.4f}")

        elif choice == "test":
            print("[*] Running sample inference...")
            sample_input = torch.randn(1, 5)
            with torch.no_grad():
                prediction = model(sample_input)
            print(f"[+] Input features : {sample_input.tolist()}")
            print(f"[+] Prediction score: {prediction.item():.4f}")

        elif choice == "exit" or choice == "quit":
            print("Peace out!")
            break
        elif choice == "":
            continue
        else:
            print(f"Unknown command: '{choice}'. Type 'train', 'test', or 'exit'.")

if __name__ == "__main__":
    main()
