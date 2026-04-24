import yaml
import subprocess as sp

exp_name = 'tas_to_Rg_all_trend'

with open("config/config_" + exp_name + "_hptesting.yaml") as f:
    config = yaml.safe_load(f)
with open("config/hpts_" + exp_name + ".yaml") as f:
    hpts = yaml.safe_load(f)

for hpt in hpts:
    config['arch'] = hpts[hpt]['arch']
    config['optimizer'] = hpts[hpt]['optimizer']
    config['expname'] = exp_name + "_" + hpt

    with open("config/config_" + config['expname'] + ".yaml", 'w') as f:
        yaml.safe_dump(config, f)

    sp.run("python -u train.py " + config['expname'], shell=True)
    sp.run("rm config/config_" + config['expname'] + ".yaml", shell=True)