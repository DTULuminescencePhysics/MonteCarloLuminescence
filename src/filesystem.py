from __future__ import annotations
import os
import shutil
from numpy import save as npsave 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_DIR = os.path.join(PROJECT_ROOT,"run")
CONFIG_DIR = os.path.join(PROJECT_ROOT,"conf")
RESULTS_DIR = os.path.join(PROJECT_ROOT,"results")


def make_run_folder(name,handler):
    folder = os.path.join(RUN_DIR,name)
    if os.path.exists(folder):
        delete = input(f"The folder {folder} already exists do you want to overwrite it? Yes/No?")
        if delete[0].lower() == 'n':
            handler.error_handler("Change job name and restart")
        elif delete[0].lower() == 'y':
            try:
                shutil.rmtree(folder)
            except Exception as e: 
                handler.error_handler(e)
        else:
            handler.error_handler("Enter either yes or no or change the job name")
    try:
        os.mkdir(folder)
    except Exception as e: 
        handler.error_handler(e)

    handler.error_check()
    handler.set_files(folder)

def make_results_folder(name,err):
    folder = os.path.join(RESULTS_DIR,name)
    if os.path.exists(folder):
        delete = input(f"The folder {folder} already exists do you want to overwrite it? Yes/No?")
        if delete[0].lower() == 'n':
            err.error_handler("Change results folder name and retry")
        elif delete[0].lower() == 'y':
            try:
                shutil.rmtree(folder)
            except Exception as e: 
                err.error_handler(e)
        else:
            err.error_handler("Enter either yes or no or change the job name")
    try:
        os.mkdir(folder)
    except Exception as e: 
        err.error_handler(e)

    err.error_check()

def output_monte_carlo_results(rep,output):#,err):
    filename  =f"rep_{rep}.npy"
    try:
        npsave(filename,output)
        return 0
    except:
        return 1



