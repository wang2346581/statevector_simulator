import os

print("lsblk | grep nvme")
os.system("sudo lsblk | grep nvme")

num_nvmes = int(input("How many M.2s to set?\n"))
max_batch_time = input("How much time set to max_batch_time?(usec)\n")
# min_batch_time = input("How much time set to max_batch_time?(usec)\n")

if int(max_batch_time) < 0:
# if int(min_batch_time) < 0:
    print("Time Error")
    exit()

for _ in range(num_nvmes):
    nvme = input(f"[NVMe]: nvme*n1 for tuning?\n")

    proceed = input(f"sudo tune2fs -E mount_opts=max_batch_time={max_batch_time} /dev/nvme{nvme}n1 (Y/n)")
    if proceed != "Y" and proceed != "y" and proceed != "":
        continue
    os.system(f"sudo tune2fs -E mount_opts=max_batch_time={max_batch_time} /dev/nvme{nvme}n1")
    
    # proceed = input(f"sudo tune2fs -E mount_opts=min_batch_time={min_batch_time} /dev/nvme{nvme}n1 (Y/n)")
    # if proceed != "Y" and proceed != "y" and proceed != "":
    #     continue
    # os.system(f"sudo tune2fs -E mount_opts=min_batch_time={min_batch_time} /dev/nvme{nvme}n1")
    
    print("\nlsblk | grep nvme")
    os.system("sudo lsblk | grep nvme")
    print()
