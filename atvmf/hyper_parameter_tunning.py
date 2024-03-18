import time 
import subprocess
import threading
import os
from datetime import datetime

models = ['fcb']
kappas = list(map(str, [16, 32, 64]))
lambdas = list(map(str, [16, 32, 64]))
args = []
for kappa in kappas:
    for lam in lambdas:
        for model in models:
            args.append({"model": model, "kappa": kappa, "lam":lam})

workers = []
while len(args)>0:
    a = os.popen("nvidia-smi --query-gpu=utilization.memory --format=csv").read()
    print(a)
    for i, ligne in enumerate(a.split("\n")[1:-1]):
        # print()
        if int(ligne[:-4]) < 70 :
            now = datetime.now()
            date_time = now.strftime("%m_%d_%Y__%H_%M_%S")
            arg = args.pop()
            command = f'python3 /home/infres/jajdenbaum/Adaptive_t-vMF_Dice_loss/CVC_ClinicDB/train.py --path "/home/infres/jajdenbaum/Adaptive_t-vMF_Dice_loss//Raidium/" -g {i} -o {date_time} -e 2 -b 16 -s 0 -mo {arg["model"]} -lo Atvmf -c 2 --kappa {arg["kappa"]} --lamda {arg["lam"]}'
            print(command)
            # a = subprocess.Popen([command])
            workers.append(threading.Thread(target= os.system(command)))
            workers[-1].start()
            time.sleep(10)
    time.sleep(1)

for worker in workers:
    worker.join()