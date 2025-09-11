import numpy as np
import pandas as pd
from classes.constants import cnst
from scipy.integrate import solve_ivp 
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

class KineticModel: 
    def __init__(self, E_loc: float, s: float, b:float, rho: float, unitless: bool = False, set_urho = False, 
                T: float =  273.15, dtdt: float = 5, n0:float = 1e9, steps=10000, sigma= None, I=None):
        self.E_loc = E_loc  # Energy barrier height (eV)
        self.s = s  # escape frequency (s⁻¹)
        self.B = self.s
        self.b = b # attempt to tunnel frequency
        self.alpha = self.set_alpha() # tunneling rate constant (1/m)
        self.unitless = unitless
        self.z = 1.8
        if set_urho: 
            self.urho = rho 
            self.rho = self.rho_def(rho,self.alpha)
        else: 
            self.rho = rho 
            self.urho = self.urho_def(rho,self.alpha)

        self.n0 = n0
        self.T_init = T 
        self.dt_dt = dtdt
        self.steps = steps
        
    def set_alpha(self):
        """Square tunneling potential"""
        alpha = 2*np.sqrt(2*cnst.m_e*self.E_loc*cnst.ev_to_j)/ cnst.h_bar
        return alpha
    
    def urho_def(self, rho:float, alpha: float) -> float:
        """Calculate the unitless density urho given alpha (in 1/m)"""
        urho = (4*np.pi* rho/3)/np.power(alpha,3)
        return urho
    
    def rho_def(self,urho:float,alpha: float) -> float:
        rho = urho*np.power(alpha,3)*(3/(np.pi*4))
        return rho
    
    def ur(self, r: float) -> float:
        """Convert distance r (in m) to unitless form"""
        return (np.cbrt((4*np.pi*self.rho)/3)*r)

    def Pr(self, r):
        """Probability density function for nearest neighbour distance r (in m)"""
        if self.unitless: 
            return 3*np.power(r,2)*np.exp(-np.power(r,3))
        else:
            return np.exp((-4*np.pi*self.rho*np.power(r,3))/3)*4*np.pi*np.power(self.rho,2)
        
    def Tau_r(self,r):
        """Calculate the tunneling rate for a given distance r"""
        return np.exp(self.alpha*r)/self.s 
    
    def Tau_ur(self,ur):
        """Calculate the unitless tunneling rate for a given distance r"""
        return(np.exp(ur/np.cbrt(self.urho))/(self.s))
    
    def Tau_ur2015(self,ur,p):
        """Calculate the unitless tunneling rate for a given distance r using 
        the updated equation from 2015 paper"""
        return ((np.exp(ur/np.cbrt(self.urho))/p)/(self.b))
    
    def r_critical(self,n):
        """Calculate critical r where tunnelling is occuring"""
        return np.cbrt(np.log(self.n0)-np.log(n))
    
    def Tau_rc(self,n):
        """Calculate the tunneling rate at critical r"""
        return np.exp(np.cbrt((np.log(self.n0)-np.log(n))/self.urho))/self.s
    
    def Tau_rc2015(self,n,p):
        """Calculate the tunneling rate at critical r using the updated equation 
        from 2015 paper"""
        return ((np.exp(np.cbrt((np.log(self.n0)-np.log(n))/self.urho))/p)/self.b)

    
    def lum_tau_c(self, ne, n, tauc,z=None):
        """Luminescence using critical tau"""
        if z is None:
            z_use=self.z
        else:
            z_use = z
        return (3*ne*np.cbrt(self.urho)/tauc)*(np.power(np.cbrt(np.log(self.n0)-np.log(n)),2)*z_use)
    
    def set_ur_values(self,N):
        """Generates N unitless values of r and sets 
        the initial population weighting"""
        self.ur_list = np.linspace(0,2,N)
        prob = self.Pr(self.ur_list) 
        area = np.sum(prob*2)
        self.weight = prob/area

    def set_thermal_A(self,T):
        """Calculate the thermal prefactor A (in s⁻¹)"""
        return self.s*np.exp(-self.E_loc/(cnst.k_b_ev*T))
       
    def set_thermal_xi(self,T):
        """Calcualte xi for a given temperature"""
        return np.exp(self.E_loc/(cnst.k_b_ev*T))
    
    def set_optical_A(self, sigma: float, I: float) -> float:
        return sigma * I
    

    def analytical_thermal2012(self,time): 
        def system(t,vars,tau):
            ng, ne = vars 
            T = self.dt_dt*t + self.T_init
            A = self.set_thermal_A(T)
            dngdt = self.B*ne -A*ng
            dnedt = A*ng -self.B*ne -ne/tau
            return [dngdt, dnedt]

        t_span = (0,time)
        t_eval = np.linspace(t_span[0], t_span[1], self.steps)

        self.Tau_ur_list = self.Tau_ur(self.ur_list)
        lum = None
        ng = None
        ne = None
        for tau, w in zip(self.Tau_ur_list, self.weight):
            ng0 =  self.n0*w*2 
            ne0 = 0
            y0 = [ng0,ne0]
            solution = solve_ivp(system, t_span, y0, args=(tau,), t_eval=t_eval,method='Radau',dense_output=True)
            ng_solution = solution.y[0]
            ne_solution = solution.y[1]
            if lum is None:
                lum = ne_solution/tau 
            else:
                lum += ne_solution/tau
            if ng is None:
                ng = ng_solution
            else:
                ng += ng_solution 
            if ne is None:
                ne = ne_solution
            else:
                ne += ne_solution 
            
            T_solution = (self.dt_dt*solution.t+self.T_init)-273.15
        
        return [T_solution, ng, ne, lum]
    
    def semi_analytical_thermal2012(self,time,z):
        def system(t,vars):
            ng, ne = vars
            n = ng+ne
            T = self.dt_dt*t + self.T_init
            A = self.set_thermal_A(T)
            tauc = self.Tau_rc(n)
            L = self.lum_tau_c(ne,n,tauc)
            dngdt = self.B*ne -A*ng
            dnedt = A*ng -self.B*ne - L
            return [dngdt, dnedt]
        
        self.z =z
        t_span = (0,time)
        t_eval = np.linspace(t_span[0], t_span[1], self.steps)
        n0=[self.n0,0]
        try:
            solution = solve_ivp(system, t_span, n0, t_eval=t_eval,method='BDF',dense_output=True)
            ng = solution.y[0]
            ne = solution.y[1]
        except:
            solution = solve_ivp(system, t_span, n0, t_eval=t_eval,method='Radau',dense_output=True)
            ng = solution.y[0]
            ne = solution.y[1]
        
        n = ng + ne
        tauc = self.Tau_rc(n)
        lum = self.lum_tau_c(ne,n,tauc)
        T_solution = (self.dt_dt*solution.t+self.T_init)-273.15

        return [T_solution, ng, ne, lum]
    
    def analytical_thermal2015(self,time): 
        def system(t,Dt,ur,ng,ne):
            T = self.dt_dt*t + self.T_init
            A = self.set_thermal_A(T)
            p= A/self.s
            tau = self.Tau_ur2015(ur,p)
            dndt = (ng+ne)/tau
            new_n = (ng+ne)-dndt*Dt
            if new_n < 0:
                new_n = 0
            ng = new_n/(p+1) 
            ne = (new_n*p)/(p+1)
            return ng, ne, T
            
        t_span = (0,time)
        t_eval = np.linspace(t_span[0], t_span[1], self.steps)
        Dt=t_eval[1]
        lum = np.zeros(self.steps)
        ng = np.zeros(self.steps)
        ne = np.zeros(self.steps)
    
        T_solution = np.zeros(self.steps)
        for ur, w in zip(self.ur_list, self.weight):
            ng1 =  self.n0*w*2 
            ne1 = 0
            lum_temp = np.zeros(self.steps)
            for t in range(self.steps):
                ng_step, ne_step, T_step = system(t_eval[t],Dt,ur,ng1,ne1)
                ng[t] += ng_step
                ne[t] += ne_step
                lum_temp[t] = ne_step
                T_solution[t] = T_step
                ng1 = ng_step 
                ne1 = ne_step
           
            lum += lum_temp/self.Tau_ur(ur)
            
        T_solution = T_solution-273.15
        return [T_solution, ng, ne, lum]
    
    def semi_analytical_thermal2015(self,time,z):
        def system(t,Dt,ng,ne): 
            n = ng+ne
            T = self.dt_dt*t + self.T_init
            A = self.set_thermal_A(T)
            p= A/self.s
            tauc = self.Tau_rc2015(n,p)
            dndt = -3*n*self.z*np.cbrt(self.urho)*np.power(np.cbrt(np.log(self.n0)-np.log(n)),2)/tauc
            new_n = n + dndt*Dt 
            if new_n <0:
                new_n=0
            ng = new_n
            ne = (new_n*p)
            return ng, ne, T
        
        self.z = z
        t_span = (0,time)
        t_eval = np.linspace(t_span[0], t_span[1], self.steps)
        Dt=t_eval[1]
        lum = np.zeros(self.steps)
        ng = np.zeros(self.steps)
        ne = np.zeros(self.steps)
        T_solution = np.zeros(self.steps)
        ng1=self.n0 
        ne1=0 
        for t in range(self.steps):
            ng_step, ne_step, T_step = system(t_eval[t],Dt,ng1,ne1)
            ng[t] = ng_step
            ne[t] = ne_step
            T_solution[t] = T_step
            ng1 = ng_step 
            ne1 = ne_step
        
        n = ng + ne
        tauc = self.Tau_rc(n)
        lum = self.lum_tau_c(ne,n,tauc)
        T_solution = T_solution-273.15
        return [T_solution, ng, ne, lum]

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

def output_to_excel(analytic, semi_analytic, analytic2015,semi_analytic2015,file_name):
    
    analyticr = np.column_stack((analytic[0],analytic[1],analytic[2],analytic[3]))
    semianalyticr = np.column_stack((semi_analytic[0],semi_analytic[1],semi_analytic[2],semi_analytic[3]))
    analyticr15 = np.column_stack((analytic2015[0],analytic2015[1],analytic2015[2],analytic2015[3]))
    semianalyticr15 = np.column_stack((semi_analytic2015[0],semi_analytic2015[1],semi_analytic2015[2],semi_analytic2015[3]))

    DFA = pd.DataFrame(analyticr)
    DFSA = pd.DataFrame(semianalyticr)
    DFA15 =pd.DataFrame(analyticr15)
    DFSA15 = pd.DataFrame(semianalyticr15)
    with pd.ExcelWriter(file_name)as writer:
        DFA.to_excel(writer, sheet_name="Analytic", index=False)
        DFSA.to_excel(writer, sheet_name="Semi-Analytic", index=False)
        DFA15.to_excel(writer, sheet_name="Analytic 2015", index=False)
        DFSA15.to_excel(writer, sheet_name="Semi-Analyti 2015", index=False)


def main(): 
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
       
        plot_lum(ax2,analytic_2012[0],analytic_2012[3],"solid", input["c1"],format_string,factor=1)
        plot_lum(ax2,semi_analytic_2012[0],semi_analytic_2012[3],"dashed", input["c1"],None,factor=1)
        plot_lum(ax2,analytic_2015[0],analytic_2015[3],"dotted", input["c1"],None,factor=1)
        plot_lum(ax2,semi_analytic_2015[0],semi_analytic_2015[3],"dashdot", input["c1"],None,factor=1)


        file_name = f"E_{input["E_loc"]}_s_{'{:.1E}'.format(input["s"])}_rho_{'{:.1E}'.format(input["rho"])}_result.xlsx"
        # output_to_excel(analytic_2012,semi_analytic_2012,analytic_2015,semi_analytic_2015,file_name) 
         
    ax.text(50,0.03e9,"n$_e$")    
    ax.text(50,0.97e9,"n$_g$")    
    ax.legend() 
    ax2.legend()
    plt.show()

  


if __name__ == "__main__":
    main()


    # def cost_function(param,ang,ane,km):
    #     t, ng,ne,lum = km.semi_analytical_thermal(param)
    #     # error = np.linalg.norm(ang-ng)+np.linalg.norm(ane-ne)
    #     error = np.linalg.norm(ane-ne)
    #     return error

    # def cost_function2(param,ang,ane,km,i):
    #     t, ng,ne,lum = km.semi_analytical_thermal(param)
    #     error = np.mean((ang[i]-ng[i])**2)#+np.mean((ane[i]-ne[i])**2)
    #     return error

    # def semi_analytical_thermal2(self,z):
    #     def system(t,vars):
    #         ng, ne = vars
    #         n = ng+ne
    #         T = self.dt_dt*t + self.T_init
    #         A = self.set_thermal_A(T)
    #         tauc = self.Tau_rc(n)
    #         L = self.lumin_intensity(ne,n,tauc,self.z_list[int(t*160/self.steps)])
    #         dngdt = self.B*ne -A*ng
    #         dnedt = A*ng -self.B*ne - L
    #         return [dngdt, dnedt]
    #     self.z_list=z
    #     self.n0 = 1e9
    #     t_span = (0,160)
    #     t_eval = np.linspace(t_span[0], t_span[1], self.steps)
    #     n0=self.n0
    #     y0 = [n0,0]
    #     solution = solve_ivp(system, t_span, y0, t_eval=t_eval,method='Radau',dense_output=True)
    #     ng_solution = solution.y[0]
    #     ne_solution = solution.y[1]
    #     n = ng_solution +ne_solution
    #     tauc = self.Tau_rc(n)
    #     lum = self.lumin_intensity(ne_solution,n,tauc,self.z_list)
    #     T_solution = (self.dt_dt*solution.t+self.T_init)-273.15
    #     return T_solution, ng_solution, ne_solution,lum

    # AT, Ang, Ane, Alum = km.analytical_thermal()
    # minimise = minimize_scalar(cost_function,bounds=(0.01,3),args=(Ang,Ane,km),method='bounded')
    # z=minimise.x
    # SAT, SAng, SAne, SAlum = km.semi_analytical_thermal(z)
    # for i in range(len(SAT)):
    #     if np.isnan(SAng[i]):
    #         print(i,'ng')
    #     if np.isnan(SAne[i]):
    #         print(i,'ne')
    #     if np.isnan(SAlum[i]):
    #         print(i,'lum')
    # z_list = np.zeros(161)
    # for i in range(161):
    #     print(i)
    #     minimise = minimize_scalar(cost_function,bounds=(0,3),args=(Ang,Ane,km,i),method='bounded')
    #     z_list[i] = minimise.x
    
    # print(z_list)
    # SAT, SAng, SAne, SAlum = km.semi_analytical_thermal2(z_list)