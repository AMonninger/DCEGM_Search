# Python script to execute matlab programs
import os

# Set directory of replication folder
basedirectory = '/Users/johanneswieland/Dropbox/DurableElasticity/Matlab/Replication/'

# Empirical Results (Figure 2 and 3)
os.chdir(basedirectory + 'IRFs')

cmd = 'matlab -batch "main; exit"'
print(cmd)
os.system(cmd) 


# Model Results (Figures 1 and 4-11)
os.chdir(basedirectory + 'Model')

modelscripts = ["figure1",
                "solvesteady","computejacobians","constructirfs",
                "figure45", "figure6", "figure7", "figure811",
                ]        

for filename in modelscripts:

        cmd = 'matlab -batch "' + filename + '; exit"'
        print(cmd)
        os.system(cmd) 