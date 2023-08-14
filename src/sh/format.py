import os

print("lsblk | grep nvme")
os.system("sudo lsblk | grep nvme")

num_nvmes = int(input("How many M.2s to format?\n"))

for _ in range(num_nvmes):
    nvme = input(f"[NVMe]: nvme*n1 format to ext4?\n")
    
    # proceed = input(f"sudo mkfs.ext4 /dev/nvme{nvme}n1(Y/n)")    
    # if proceed != "Y" and proceed != "y" and proceed != "":
    #     continue
    # os.system(f"sudo mkfs.ext4 /dev/nvme{nvme}n1")

    # proceed = input(f"sudo mkfs.ext4 /dev/nvme{nvme}n1 -O sparse_super2,large_file -E dioread_nolock,lazy_itable_init=0 -m 0 -T largefile4 (Y/n)")    
    # if proceed != "Y" and proceed != "y" and proceed != "":
    #     continue
    # os.system(f"sudo mkfs.ext4 /dev/nvme{nvme}n1 -O sparse_super2,large_file -E dioread_nolock,lazy_itable_init=0 -m 0 -T largefile4")

    proceed = input(f"sudo mkfs.ext4 /dev/nvme{nvme}n1 -O sparse_super2,large_file -E lazy_itable_init=0 -m 0 -T largefile4 (Y/n)")    
    if proceed != "Y" and proceed != "y" and proceed != "":
        continue
    os.system(f"sudo mkfs.ext4 /dev/nvme{nvme}n1 -O sparse_super2,large_file -E lazy_itable_init=0 -m 0 -T largefile4")
    
    # proceed = input(f"sudo tune2fs -E mount_opts=max_batch_time=0 /dev/nvme{nvme}n1 (Y/n)")
    # if proceed != "Y" and proceed != "y" and proceed != "":
    #     continue
    # os.system(f"sudo tune2fs -E mount_opts=max_batch_time=0 /dev/nvme{nvme}n1")

    proceed = input(f"sudo tune2fs -E mount_opts=dioread_nolock /dev/nvme{nvme}n1 (Y/n)")
    if proceed != "Y" and proceed != "y" and proceed != "":
        continue
    os.system(f"sudo tune2fs -E mount_opts=dioread_nolock /dev/nvme{nvme}n1")
    
    print("\nlsblk | grep nvme")
    os.system("sudo lsblk | grep nvme")
    print()
