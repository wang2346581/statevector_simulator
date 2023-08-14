import os

print("\nlsblk | grep nvme")
os.system("lsblk | grep nvme")

num_cards = int(input("How many cards to remove state?\n"))

for num_card in range(num_cards):
    card = str(input("[Card]: card number?\n"))

    print(f"sudo rm /mnt/nvme/card{card}/0/state*")
    print(f"sudo rm /mnt/nvme/card{card}/1/state*")
    print(f"sudo rm /mnt/nvme/card{card}/2/state*")
    print(f"sudo rm /mnt/nvme/card{card}/3/state*")
    print()

    os.system(f"sudo rm /mnt/nvme/card{card}/0/state*")
    os.system(f"sudo rm /mnt/nvme/card{card}/1/state*")
    os.system(f"sudo rm /mnt/nvme/card{card}/2/state*")
    os.system(f"sudo rm /mnt/nvme/card{card}/3/state*")