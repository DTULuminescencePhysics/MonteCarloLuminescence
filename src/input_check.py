from __future__ import annotations
from src.errors import ErrorOutputHandler
from omegaconf import DictConfig, OmegaConf
from src.classes.dot_dict import ddict

def number_check(val, name: str, err: ErrorOutputHandler):
    if not isinstance(val,(int,float)):
        err.error(f"Value Error: {name} has value {val}",fatal=True)

def check_physics(physics: dict, err: ErrorOutputHandler):

    number_check(physics.E_loc,"E_loc",err)
    number_check(physics.b,"b",err)
    number_check(physics.s,"s",err)
    if physics.alpha is not None:
        number_check(physics.alpha,"alpha",err)
    if physics.E_cb is not None:
        number_check(physics.E_cb,"E_cb",err)
    if physics.D0 is not None:
        number_check(physics.D0,"D0",err)
        if physics.D_dot is not None:
            number_check(physics.D_dot,"D_dot",err)
        else: 
            err.error("Using D0 requires D_dot to also be set",fatal=True)
    

def check_monte_carlo(mc: dict, err: ErrorOutputHandler):

    if mc.exp_typ.upper() not in ["TL","OSL","BACKGROUND"]:
        err.error("Experiment type needs to be one of TL, OSL or BACKGROUND",fatal=True)
    else:
        mc.exp_typ = mc.exp_typ.upper() 

    number_check(mc.T_init,"T_init",err)
    if mc.duration is None:
        if mc.T_rate is None or mc.T_end is None:
            err.error("Duration of experiment cannot be determined either specify a duration or " \
                        "end temperature and heating rate",fatal=True)
        else:
            number_check(mc.T_end,"T_end",err)
            number_check(mc.T_rate,"T_rate",err)
            try:
                mc.duration = (mc.T_end-mc.T_init)/mc.T_rate
            except:
                err.error("Could not set simulation durration",fatal=True)
    else: 
        number_check(mc.duration,"duration",err)


    if mc.T_end is not None:
        number_check(mc.T_init,"T_init",err)

    number_check(mc.max_dt,"max_dt",err)
    if not isinstance(mc.max_dt,(int,float)):
        err.error(f"Value Error: mc.max_dt has value {mc.max_dt} but has been set to 1",fatal=False)
        mc.max_dt = 1

    number_check(mc.n_el,"n_el",err)
    number_check(mc.n_tr,"n_tr",err)
    number_check(mc.reps,"reps",err)
    try:
        mc.reps = int(mc.reps)
    except:
        err.error("Could not convert the number of repetitions to an integer",fatal=True)


def check_inputs(cfg: DictConfig, err: ErrorOutputHandler): 
    """Checks input paramters"""
    err.output("Checking input parameters")

    check_monte_carlo(cfg.mc,err)
    check_physics(cfg.physics,err)

    if cfg.physics.rho is None and cfg.physics.urho is None:
        if( not isinstance(cfg.mc.n_el,(int,float)) or 
            not isinstance(cfg.mc.l,(int,float)) or 
            not isinstance(cfg.mc.w,(int,float)) or 
            not isinstance(cfg.mc.h,(int,float))):
                err.error("A density paramter has not been specified so it either must be specified or the " \
                "crystal dimensions be corrected so it can be infered",fatal=True)
        else:
            cfg.physics.rho = cfg.mc.n_el/(cfg.mc.l*cfg.mc.w*cfg*cfg.mc.h)
    else:
        if cfg.physics.rho is not None:
            number_check(cfg.physics.rho,"rho",err)
        if cfg.physics.urho is not None:
            number_check(cfg.physics.urho,"urho",err)
        if not isinstance(cfg.mc.n_el,(int,float)): 
            if(not isinstance(cfg.mc.l,(int,float)) or 
               not isinstance(cfg.mc.w,(int,float)) or 
               not isinstance(cfg.mc.h,(int,float))): 
                err.error("The inital number of electrons and crystal size will be set automatically",fatal=False)
            else: 
                err.error("The inital number of electrons will be determined by the density and crystal volume",fatal=False)            
        
    err.checkpoint()
    err.clear_errors()

    if cfg.physics.E_cb is None:
        if cfg.physics.D0 is None:
            ther_type = "Thermal"
        else:
            ther_type = "ThermalD"
    else:
        if cfg.physics.D0 is None:
            ther_type = "ThermalC"
        else:
            ther_type = "ThermalCD"

    phys_in = ddict({
        "E_loc": cfg.physics.E_loc,
        "alpha": cfg.physics.alpha,
        "b": cfg.physics.b,
        "s": cfg.physics.s,
        "E_cb": cfg.physics.E_cb,
        "D0": cfg.physics.D0,
        "D_dot": cfg.physics.D_dot,
        "T_init": (cfg.mc.T_init+273.15),
        "dT": cfg.mc.T_rate,
        "rho": cfg.physics.rho,
        "urho": cfg.physics.urho
    })

    mc_in = ddict({
        "therm_type": ther_type,
        "duration": cfg.mc.duration,
        "n_el": cfg.mc.n_el, 
        "n_tr": cfg.mc.n_tr,
        "reps": cfg.mc.reps,
        "max_dt":cfg.mc.max_dt,
        "h": cfg.mc.h,
        "w": cfg.mc.w,
        "l": cfg.mc.l,
    })
    err.output("Input check complete")
    return phys_in, mc_in 

