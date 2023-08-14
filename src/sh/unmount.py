import os

print("lsblk | grep nvme")
os.system("sudo lsblk | grep nvme")
print()

num_cards = int(input("How many cards to unmount?\n"))

for num_card in range(num_cards):
    card = str(input("card number?\n"))

    proceed = input("Unmount these SSDs?(Y/n)")
    if proceed != "Y" and proceed != "y" and proceed != "":
        print("skip")
        continue
    
    for j in range(4):
        os.system(f"sudo umount /mnt/nvme/card{card}/{j}")
