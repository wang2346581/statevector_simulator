from circuit_generator import *
from ini_generator import *
from test_util import *
from math import pi

# Test for Phase and CPhase gates

# N    =  8
# NGQB =  2
# NSQB =  4
# NLQB =  2

class phaseTest:
    def __init__(self):
        self.setting = {'total_qbit':'8',
                        'device_qbit':'2',
                        'file_qbit':'2',
                        'thread_qbit':'2',
                        'local_qbit':'2',
                        'max_qbit':'38',
                        'testing':'1',
                        'state_paths':'./state/path1,./state/path2,./state/path3,./state/path4'}
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
        self.range_type = [range(2), range(2, 4), range(4, 6), range(6, 8)]

    def _test_Phase(self, name, x_range, phase):
        flag = True
        for x in x_range:
            circuit=get_circuit()
            for i in range(self.N):
                H(circuit, i)
            Phase(circuit, x, phase)
            create_circuit(circuit, self.cir_path)
            flag=self.execute(name)
            # os.system(f"../qSim.out -i {self.ini_path} -c {self.cir_path} >> /dev/null")
            # print("===========================", flush=True)

            # flag = simple_test(name, False, self.cir_path, "../path/set7.txt", self.N, self.NGQB, True)
            if(not flag): break
        if(not flag):
            print("[X]", name, ": not pass under 1e-9", flush=True)
            print("===========================")
        return flag

    def _test_CPhase(self, name, x_range, y_range, phase):
        flag = True
        for x in x_range:
            for y in y_range:
                if x==y: continue
                circuit=get_circuit()
                for i in range(self.N):
                    H(circuit, i)
                CPhase(circuit, x, y, phase)
                create_circuit(circuit, self.cir_path)
                flag=self.execute(name)
                # os.system(f"../qSim.out -i {self.ini_path} -c {self.cir_path} >> /dev/null")
                # print("===========================", flush=True)

                # flag = simple_test(name, False, self.cir_path, "../path/set7.txt", self.N, self.NGQB, True)
                if(not flag): break

        if(not flag):
            print("[X]", name, ": not pass under 1e-9", flush=True)
            print("===========================")
        return flag
    def execute(self, name):
        if self.world_size!=1:
            # os.chdir("..")
            # os.system('make clean')
            # os.system('make')
            # print("[SCP End]")
            # os.chdir('correctness')
            # print('[Move End]')
            os.system("rm -r state0/*")
            os.system("rm -r state1/*")
            os.system("rm -r state2/*")
            os.system("rm -r state3/*")
            # os.system('scp -q -r /home/paslab/Desktop/new_file/stateVector/src paslab@140.112.90.50:/home/paslab/Desktop/new_file/stateVector/')
            os.system(f"mpirun.mpich -np {self.world_size} ../qSim.out -i {self.ini_path} -c {self.cir_path}")
            os.system("cp ./state1/path1 ./state0/path5")
            os.system("cp ./state1/path2 ./state0/path6")
            os.system("cp ./state1/path3 ./state0/path7")
            os.system("cp ./state1/path4 ./state0/path8")
            os.system("cp ./state2/path1 ./state0/path9")
            os.system("cp ./state2/path2 ./state0/path10")
            os.system("cp ./state2/path3 ./state0/path11")
            os.system("cp ./state2/path4 ./state0/path12")
            os.system("cp ./state3/path1 ./state0/path13")
            os.system("cp ./state3/path2 ./state0/path14")
            os.system("cp ./state3/path3 ./state0/path15")
            os.system("cp ./state3/path4 ./state0/path16")
        else:
            os.system(f"../qSim.out -i {self.ini_path} -c {self.cir_path} >> /dev/null")
        # os.system(f"../qSim.out -i {self.ini_path} -c {self.cir_path}")
        return simple_test(name, False, self.cir_path, "../path/set9.txt", self.N, self.NGQB+self.NDQB, True)
    def test(self):
        create_ini(self.setting, self.ini_path)
        
        # CPhase test
        flag = True
        for i in range(4):
            q0_type = self.qubit_type[i]
            test_name = f"{q0_type}"
            for j in range(8):
                angle = j * pi/8
                flag = self._test_Phase(test_name, self.range_type[i], angle)
                if not flag:    break
            if not flag:    break
        if(not flag):
            print("[x]", "Phase Test", ": not pass under 1e-9", flush=True)
            print("===========================")
            return False
        print("[PASS]", "Phase Test", ": match with qiskit under 1e-9", flush=True)
        print("===========================")

        # Phase test
        flag = True
        for i in range(4):
            for j in range(i,4):
                q0_type = self.qubit_type[i]
                q1_type = self.qubit_type[j]
                test_name = f"{q0_type}-{q1_type}"
                for k in range(8):
                    angle = k * pi/8
                    flag = self._test_CPhase(test_name, self.range_type[i], self.range_type[j], angle)
                    if not flag:    break
                if not flag:    break
            if not flag:    break
    
        if(flag):
            print("[PASS]", "CPhase Test", ": match with qiskit under 1e-9", flush=True)
            print("===========================")
            print("[PASS]", "Phase, CPhase Test", ": match with qiskit under 1e-9", flush=True)
        else:
            print("[x]", "CPhase Test", ": not pass under 1e-9", flush=True)
        print("===========================")
        return flag