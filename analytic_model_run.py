from __future__ import annotations
from src.AnalyticModel.r_unitless_r_plots import main, main2
from src.AnalyticModel.kinetic_model import KineticModel
import matplotlib.pyplot as plt
import pandas as pd

def plot_ng(ax,T,ng,line,color,lab,factor=1):
    if lab == 0:
        ax.plot(T,ng*factor,linestyle=line,color=color)
    elif lab == 1:
        ax.plot(T,ng*factor,linestyle=line,label="n$_g$",color=color)
    else:
        ax.plot(T,ng*factor,linestyle=line,label=f"{lab}",color=color)

def plot_ne(ax,T,ne,line,color,lab,factor=1):
    if lab == 0:
        ax.plot(T,ne*factor,linestyle=line,color=color)
    elif lab == 1:
        ax.plot(T,ne*factor,linestyle=line,label="n$_e$",color=color)
    else:
        ax.plot(T,ne*factor,linestyle=line,label=f"{lab}",color=color)

def plot_lum(ax,T,lum,line,color,lab,factor=1):
    if lab is None: 
        ax.plot(T,lum*factor,linestyle=line,color=color)
    else:
        ax.plot(T,lum*factor,linestyle=line,label=f"{lab}",color=color)

def pretty_exponent_maker(number):
    exp = int(f"{number:.0E}".split('E')[1])
    num2 = float(f"{number:.1E}".split('E')[0])
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exponent_str = str(exp).translate(superscripts)
    formatted = f"{num2}x10{exponent_str}"
    return formatted

def build_data_frame(result): 
    df = (
        pd.DataFrame({
            "Temperature":  result[0],
            "n_g":   result[1],
            "n_e":   result[2],
            "lum":   result[3],
        })
    )
    return df

def output_to_excel(analytic, semi_analytic, analytic2015,semi_analytic2015,file_name):

    DFA = build_data_frame(analytic)
    DFSA = build_data_frame(semi_analytic)
    DFA15 = build_data_frame(analytic2015)
    DFSA15 = build_data_frame(semi_analytic2015)

    with pd.ExcelWriter(file_name)as writer:
        DFA.to_excel(writer, sheet_name="Analytic", index=False)
        DFSA.to_excel(writer, sheet_name="Semi-Analytic", index=False)
        DFA15.to_excel(writer, sheet_name="Analytic 2015", index=False)
        DFSA15.to_excel(writer, sheet_name="Semi-Analyti 2015", index=False)


def paper_examples(): 
    time=160; steps=10000; N=100
    T_init = 273.15; dtdt = 5; n0=1e9
    input1 = {"E_loc": 0.8, 
             "s":1e10, 
             "rho":8e-4, 
             "factor":1e5, 
             "z":1.8, 
             "b":1e10,
             "c1":"black",
             "c2":"orange"
             }
    input2 = {"E_loc": 0.8, 
             "s":1e10, 
             "rho":3e-4, 
             "factor":1e4, 
             "z":1.8, 
             "b":1e10,
             "c1":"blue",
             "c2":"pink"}
    input3 = {"E_loc": 1.2, 
             "s":1e12, 
             "rho":3e-4, 
             "factor":2e6, 
             "z":1.8, 
             "b":1e12,
             "c1":"green",
             "c2":"grey"}
    inputs = [input1,input2,input3]
    
    conc_plot = plt.figure(figsize=(12,6))
    ax = conc_plot.add_subplot(1, 2, 1)
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp = str("-3").translate(superscripts)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(f"Concentration (cm{exp})")
    ax.set_title("n$_g$, n$_e$ Populations")
    ax.set_xlim(0, 800)
    ax2 = conc_plot.add_subplot(1,2,2)
    ax2.set_xlim(0, 800)
    exp = str("-1").translate(superscripts)
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel(f"Intensity (°C{exp})")
    ax2.set_title("Luminescence")
    z=1.8
    for input in inputs:
        
        km = KineticModel(E_loc=input["E_loc"],s=input["s"],b=input["b"],rho=input["rho"],
                          unitless=True,set_urho=True,T=T_init,dtdt=dtdt,n0=n0,steps=steps)
        km.set_ur_values(N)
        format_string = f"ρ'={pretty_exponent_maker(input["rho"])}; E={input["E_loc"]}; s={pretty_exponent_maker(input["s"])}"
        analytic_2012 = km.analytical_thermal2012(time)
        plot_ng(ax,analytic_2012[0],analytic_2012[1],"solid", input["c1"],format_string,factor=1)
        plot_ne(ax,analytic_2012[0],analytic_2012[2],"solid",input["c1"],0,factor=input["factor"])
        
        semi_analytic_2012 = km.semi_analytical_thermal2012(time,z)
        plot_ng(ax,semi_analytic_2012[0],semi_analytic_2012[1],"dashed", input["c1"],0,factor=1)
        plot_ne(ax,semi_analytic_2012[0],semi_analytic_2012[2],"dashed",input["c1"],0,factor=input["factor"])
    
        analytic_2015 = km.analytical_thermal2015(time)
        plot_ng(ax,analytic_2015[0],analytic_2015[1],"dotted", input["c1"],0,factor=1)
        plot_ne(ax,analytic_2015[0],analytic_2015[2],"dotted",input["c1"],0,factor=input["factor"])

        semi_analytic_2015 = km.semi_analytical_thermal2015(time,z)
        plot_ng(ax,semi_analytic_2015[0],semi_analytic_2015[1],"dashdot", input["c1"],0,factor=1)
        plot_ne(ax,semi_analytic_2015[0],semi_analytic_2015[2],"dashdot",input["c1"],0,factor=input["factor"])
       
        plot_lum(ax2,analytic_2012[0],analytic_2012[3]/dtdt,"solid", input["c1"],format_string,factor=1)
        plot_lum(ax2,semi_analytic_2012[0],semi_analytic_2012[3]/dtdt,"dashed", input["c1"],None,factor=1)
        plot_lum(ax2,analytic_2015[0],analytic_2015[3]/dtdt,"dotted", input["c1"],None,factor=1)
        plot_lum(ax2,semi_analytic_2015[0],semi_analytic_2015[3]/dtdt,"dashdot", input["c1"],None,factor=1)


        file_name = f"E_{input["E_loc"]}_s_{'{:.1E}'.format(input["s"])}_rho_{'{:.1E}'.format(input["rho"])}_result.xlsx"
        output_to_excel(analytic_2012,semi_analytic_2012,analytic_2015,semi_analytic_2015,file_name) 
         
    # factors = [1e6,1e4,2e6]
    # i=0
    # for input in inputs:
    #     km = KineticModel(E_loc=input["E_loc"],s=input["s"],b=input["b"],rho=input["rho"],
    #                       unitless=True,set_urho=True,T=T_init,dtdt=dtdt,n0=n0,steps=steps)

    #     truncated = km.truncated_model_2015(time,z)
    #     format_string = f"ρ'={pretty_exponent_maker(input["rho"])}; E={input["E_loc"]}; s={pretty_exponent_maker(input["s"])}"
    #     plot_ng(ax,truncated[0],truncated[1],"solid", input["c1"],0,factor=1)
    #     plot_ne(ax,truncated[0],truncated[2],"dotted",input["c1"],0,factor=factors[i])
    #     plot_lum(ax2,truncated[0],truncated[3],"dotted", input["c1"],format_string,factor=1)
    #     i+=1

    ax.text(50,0.03e9,"n$_e$")    
    ax.text(50,0.97e9,"n$_g$")    
    ax.legend() 
    ax2.legend()
    plt.show()

if __name__ == "__main__":
    paper_examples()
    main()
    main2()