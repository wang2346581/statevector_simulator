from circuit_generator import *
from ini_generator import *


setting =   {'total_qbit':'20',
            'file_qbit':'3',
            'thread_qbit':'3',
            'local_qbit':'12',
            'max_qbit':'38',
            'max_path':'260',
            'max_depth':'1000',
            'is_density':'0',
            'skip_init_state':'0',
            'set_of_save_state':'2',
            'state_paths':'./state/path1,./state/path2,./state/path3,./state/path4,./state/path5,./state/path6,./state/path7,./state/path8,./state/path9,./state/path10,./state/path11,./state/path12,./state/path13,./state/path14,./state/path15,./state/path16,./state/path17,./state/path18,./state/path19,./state/path20,./state/path21,./state/path22,./state/path23,./state/path24,./state/path25,./state/path26,./state/path27,./state/path28,./state/path29,./state/path30,./state/path31,./state/path32,./state/path33,./state/path34,./state/path35,./state/path36,./state/path37,./state/path38,./state/path39,./state/path40,./state/path41,./state/path42,./state/path43,./state/path44,./state/path45,./state/path46,./state/path47,./state/path48,./state/path49,./state/path50,./state/path51,./state/path52,./state/path53,./state/path54,./state/path55,./state/path56,./state/path57,./state/path58,./state/path59,./state/path60,./state/path61,./state/path62,./state/path63,./state/path64'}
            # 'state_paths':'./state/path1,./state/path2,./state/path3,./state/path4,./state/path5,./state/path6,./state/path7,./state/path8'}
ini_path='test.ini'
cir_path='cir_test'

mpi=1
# set_ini(setting,'total_qbit',20)
if mpi == 1:
    set_ini(setting,'device_qbit',2)

create_ini(setting,ini_path)

cir_path='cir_test'
circuit=get_circuit()

for i in range(20):
    H(circuit,i)


m_list = [i for i in range(20)]
measure_multi(circuit, m_list, 2)

# H(circuit, 29)

# H(circuit,0)
create_circuit(circuit,cir_path)

os.chdir("..")
os.system('make clean')
os.system('make')
if mpi==1:
    os.system('rm /home/paslab/Desktop/new_file/stateVector/src/data_test/state0/*')
    os.system('rm /home/paslab/Desktop/new_file/stateVector/src/data_test/state1/*')
    os.system('rm /home/paslab/Desktop/new_file/stateVector/src/data_test/state2/*')
    os.system('rm /home/paslab/Desktop/new_file/stateVector/src/data_test/state3/*')
    # os.system('scp -q -r /home/paslab/Desktop/new_file/stateVector paslab@140.112.90.50:/home/paslab/Desktop/new_file/')
    # print("[SCP End]")
    # print('[Move End]')
os.chdir('data_test')

if mpi == 1:
    # os.system(f"mpirun.mpich --hostfile hf -np 2 ../qSim.out -i {ini_path} -c {cir_path}")
    os.system(f"mpirun.mpich -np 4 ../qSim.out -i {ini_path} -c {cir_path}")
    # os.system(f"mpirun.mpich --host headnode worker1 -np 2 ../qSim.out -i {ini_path} -c {cir_path}")
else:
    os.system(f"../qSim.out -i {ini_path} -c {cir_path}")

#os.system(f"mpirun.mpich --host headnode worker1 -np 2 ../qSim.out -i {ini_path} -c {cir_path}")
#os.system(f"mpirun.mpich  --host localhost -np 1 ../qSim.out -i {ini_path} -c {cir_path}")


