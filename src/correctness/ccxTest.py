from circuit_generator import *
from ini_generator import *
from test_util import *

# Test for CCX gate

# N    = 12
# NGQB =  3
# NSQB =  6
# NLQB =  3

class ccxTest:
    def __init__(self):
        self.setting = {'total_qbit':'12',
                        'device_qbit':'3',
                        'file_qbit':'3',
                        'thread_qbit':'3',
                        'local_qbit':'3',
                        'max_qbit':'38',
                        'testing':'1',
                        'state_paths':'./state/path1,./state/path2,./state/path3,./state/path4,./state/path5,./state/path6,./state/path7,./state/path8'}
        self.ini_path='test.ini'
        self.cir_path='cir_test'
        
        self.N    =  int(self.setting['total_qbit'])
        self.NGQB =  int(self.setting['file_qbit'])
        self.NDQB =  int(self.setting['device_qbit'])
        self.world_size = 1<<self.NDQB
        self.NSQB =  int(self.setting['file_qbit'])
        self.setting['thread_qbit'] = self.setting['file_qbit']
        self.NLQB =  int(self.setting['local_qbit'])
        self.qubit_type = ["Device","File", "Middle", "Local"]
        self.range_type = [range(3), range(3, 6), range(6, 9), range(9,12)]

    def _test(self, name, x_range, y_range, z_range):
        flag = True
        for x in x_range:
            for y in y_range:
                if x==y: continue
                for z in z_range:
                    if z==x: continue
                    if z==y: continue

                    circuit=get_circuit()
                    for i in range(12):
                        if i == z: continue
                        H(circuit, i)
                    CCX(circuit, x, y, z)
                    create_circuit(circuit, self.cir_path)

                    os.system("rm -r state0/*")
                    os.system("rm -r state1/*")
                    os.system("rm -r state2/*")
                    os.system("rm -r state3/*")
                    os.system("rm -r state4/*")
                    os.system("rm -r state5/*")
                    os.system("rm -r state6/*")
                    os.system("rm -r state7/*")
                    # os.system('scp -q -r /home/paslab/Desktop/new_file/stateVector/src paslab@140.112.90.50:/home/paslab/Desktop/new_file/stateVector/')
                    os.system(f"mpirun.mpich -np {self.world_size} ../qSim.out -i {self.ini_path} -c {self.cir_path}")
                    # exit(1)
                    cnt=9
                    for qqq in range(1,8):
                        for www in range(1,9):
                            os.system(f"cp ./state{qqq}/path{www} ./state0/path{cnt}")
                            cnt=cnt+1



                    # os.system(f"../qSim.out -i {self.ini_path} -c {self.cir_path} >> /dev/null")
                    # os.system(f"../qSim.out -i {self.ini_path} -c {self.cir_path}")
                    # print("===========================", flush=True)

                    circuit=get_circuit()
                    for i in range(12):
                        if i == z: continue
                        H(circuit, i)
                    CCX_true(circuit, x, y, z)
                    create_circuit(circuit, self.cir_path)

                    flag = simple_test(name, False, self.cir_path, "../path/set10.txt", self.N, self.NGQB+self.NDQB, True)
                    if(not flag): break
                if(not flag): break
            if(not flag): break

        if(not flag):
            print("[X]", name, ": not pass under 1e-9", flush=True)
            print("===========================")
        return flag

    def test(self):
        create_ini(self.setting, self.ini_path)
        flag = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    q0_type = self.qubit_type[i]
                    q1_type = self.qubit_type[j]
                    q2_type = self.qubit_type[k]
                    test_name = f"{q0_type}-{q1_type}-{q2_type}"
                    flag = self._test(test_name, self.range_type[i], self.range_type[j], self.range_type[k])
                    if not flag:    break
                if not flag:    break
            if not flag:    break 

        if(flag):
            print("[PASS]", "ccxTest", ": match with qiskit under 1e-9", flush=True)
        else:
            print("[x]", "ccxTest", ": not pass under 1e-9", flush=True)
        print("===========================")
        return flag
