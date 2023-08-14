import os

print("ls -l /mnt/nvme/")
os.system("ls -l /mnt/nvme/")

num_cards = int(input("How many cards to mount?\n"))

for num_card in range(num_cards):
    print("ls -l /mnt/nvme/")
    os.system("ls -l /mnt/nvme/")
    print()
    card = str(input("[Card]: card number?\n"))

    proceed = input("create new directory? (y/N)\n")
    if proceed == "Y" or proceed == "y":
        os.system(f"sudo mkdir /mnt/nvme/card{card}")
        os.system(f"sudo mkdir /mnt/nvme/card{card}/0")
        os.system(f"sudo mkdir /mnt/nvme/card{card}/1")
        os.system(f"sudo mkdir /mnt/nvme/card{card}/2")
        os.system(f"sudo mkdir /mnt/nvme/card{card}/3")
        print("ls -l /mnt/nvme/")
        os.system("ls -l /mnt/nvme/")

    print(f"ls -l /mnt/nvme/card{card}")
    os.system(f"ls -l /mnt/nvme/card{card}")
    print("\nlsblk | grep nvme")
    os.system("sudo lsblk | grep nvme")
    print()

    for j in range(4):
        nvme = input(f"[NVMe]: nvme*n1 mount to /mnt/nvme/card{card}/{j}\n")

        os.system(f"sudo mount -t ext4 /dev/nvme{nvme}n1 /mnt/nvme/card{card}/{j}")            
        print("\nlsblk | grep nvme")
        os.system("sudo lsblk | grep nvme")
        print()
