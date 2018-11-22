#### config.yaml contains the two containers, they would be flagges as 'source'a nd 'destination'.
# Transfer would be initiated from the source to the destination.
# Only required changed files would be transfered.
# If the docker server is remote/local, suitable transfer will take place...

# Important things to do:-
# Pre-conditions...
# pre #1: Generate about 1,000,000 data each time generate data is called.
# pre #2: Generate teh models periodically (mini-batch to be used to generate the model). Transfer the model based on whole data and based on partial data.
# 1. Do a complete transfer of model and training data and check the transfer time....
# 2. Do a partial transfer of model and training data and check the transfer time....
# 3. Do a partial transfer and check the accuracy of the model.


import yaml
import os
config_file = yaml.load(open('config_containers.yaml'))
print("container name==",config_file['source_cont_name'])

source_docker_fs = config_file['source_docker_serv_fs']
print("source file sys present in : ", source_docker_fs)

destin_docker_fs = config_file['dest_docker_serv_fs']
print("destn files sys in: ", destin_docker_fs)

dest_docker_IP = config_file['dest_docker_serv_IP']
print("Destination server IP Address==",dest_docker_IP)

dummy1 = 'naveen.txt'
dummy2 = 'naveen_cpy.txt'
cmd = 'cp'
os.system(cmd+' '+dummy1+' '+dummy2)
#step2. Copy from souce docker_server --> destination_docker_server
#check if it's a localsystem, just do a cp.
if dest_docker_IP == "127.0.0.1":
    print("I need to do a Copy.. from source to destination directory")

# check if it's a remote system, then do a SCP.
else:
    print("I have reached here. Need to SCP the files")
