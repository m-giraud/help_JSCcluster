
import sys

CPBdir = "../.."
sys.path.append(CPBdir+"/src");
sys.path.append(CPBdir);
sys.path.append("../../..");sys.path.append(".."); 
sys.path.append(CPBdir+"/src/python_modules");
sys.path.append("../build-cmake/cpp/python_binding/") # dumux python binding
sys.path.append("../../build-cmake/cpp/python_binding/")
sys.path.append("../modules/") # python wrappers 

#from rosi_richards import RichardsSP  # C++ part (Dumux binding)
#from richards import RichardsWrapper  # Python part
from phloem_flux import PhloemFluxPython  # Python hybrid solver
#from Leuning_speedup import Leuning #about 0.7 for both
#from photosynthesis_cython import Leuning
import plantbox as pb
#from plantbox import Photosynthesis as ph
#import vtk_plot as vp
import math
import os
import numpy as np
#import vtk_plot as vp
#import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def theta2H(vg,theta):#(-) to cm
    thetar =vg[0]# 0.059
    thetas = vg[1]#0.445
    alpha = vg[2]#0.00644
    n = vg[3]#1.503
    nrev = 1/(1-1/n)
    H =-(((( (thetas - thetar)/(theta - thetar))**nrev) - 1)**(1/n))/alpha
    return(H)#cm

def sinusoidal(t):
    return 1#(np.sin(np.pi*t*2)+1)/2 #( (t%1) < 0.5)#

def qair2rh(qair, es_, press):
    e =qair * press / (0.378 * qair + 0.622)
    rh = e / es_
    rh=max(min(rh, 1.),0.)
    return rh



def div0(a, b, c):        
    return np.divide(a, b, out=np.full(len(a), c), where=b!=0)
    
def div0f(a, b, c):    
    if b != 0:
        return a/b
    else:
        return a/c
        



def setKrKx_xylem(TairC, RH,r,DictVal): #inC
    #mg/cm3
    hPa2cm = 1.0197
    dEauPure = (999.83952 + TairC * (16.952577 + TairC * 
        (- 0.0079905127 + TairC * (- 0.000046241757 + TairC * 
        (0.00000010584601 + TairC * (- 0.00000000028103006)))))) /  (1 + 0.016887236 * TairC)
    siPhi = (30 - TairC) / (91 + TairC)
    siEnne=0
    mu =  pow(10, (- 0.114 + (siPhi * (1.1 + 43.1 * pow(siEnne, 1.25) )))) 
    mu = mu /(24*60*60)/100/1000; #//mPa s to hPa d, 1.11837e-10 hPa d for pure water at 293.15K
    mu = mu * hPa2cm #hPa d to cmh2o d 

    #number of vascular bundles
    VascBundle_leaf = 32
    VascBundle_stem = 52
    VascBundle_root = 1 #valid for all root type
            
    #radius of xylem type^4 * number per bundle
    rad_x_l_1   = (0.0015 **4) * 2; rad_x_l_2   = (0.0005 **4) * 2   
    rad_x_s_1   = (0.0017 **4) * 3; rad_x_s_2   = (0.0008 **4) * 1     
    rad_x_r0_1  = (0.0015 **4) * 4    
    rad_x_r12_1 = (0.00041**4) * 4; rad_x_r12_2 = (0.00087**4) * 1
    rad_x_r3_1  = (0.00068**4) * 1      

    # axial conductivity [cm^3/day]        
    kz_l  = VascBundle_leaf *(rad_x_l_1 + rad_x_l_2)    *np.pi /(mu * 8)  * DictVal['kx_x']
    kz_s  = VascBundle_stem *(rad_x_s_1 + rad_x_s_2)    *np.pi /(mu * 8) * DictVal['kx_x']
    kz_r0 = VascBundle_root * rad_x_r0_1                *np.pi /(mu * 8) * DictVal['kx_x'] 
    kz_r1 = VascBundle_root *(rad_x_r12_1 + rad_x_r12_2)*np.pi /(mu * 8) * DictVal['kx_x']
    kz_r2 = VascBundle_root *(rad_x_r12_1 + rad_x_r12_2)*np.pi /(mu * 8)  * DictVal['kx_x']
    kz_r3 = VascBundle_root * rad_x_r3_1                *np.pi /(mu * 8) * DictVal['kx_x']

    #radial conductivity [1/day],
    kr_l  = 3.83e-3 * hPa2cm * DictVal['kr_x']
    kr_s  = 0.#1.e-20  * hPa2cm # set to almost 0
    kr_r0 = 6.37e-5 * hPa2cm * DictVal['kr_x']
    kr_r1 = 7.9e-5  * hPa2cm * DictVal['kr_x']
    kr_r2 = 7.9e-5  * hPa2cm  * DictVal['kr_x']
    kr_r3 = 6.8e-5  * hPa2cm * DictVal['kr_x']
    l_kr = 0.8 #cm
    r.setKr([[kr_r0,kr_r1,kr_r2,kr_r0],[kr_s,kr_s ],[kr_l]], kr_length_=l_kr) 
    r.setKx([[kz_r0,kz_r1,kz_r2,kz_r0],[kz_s,kz_s ],[kz_l]])
    
    
    Rgaz=8.314 #J K-1 mol-1 = cm^3*MPa/K/mol
    rho_h2o = dEauPure/1000#g/cm3
    Mh2o = 18.05 #g/mol
    MPa2hPa = 10000
    hPa2cm = 1/0.9806806
    #log(-) * (cm^3*MPa/K/mol) * (K) *(g/cm3)/ (g/mol) * (hPa/MPa) * (cm/hPa) =  cm                      
    p_a = np.log(RH) * Rgaz * rho_h2o * (TairC + 273.15)/Mh2o * MPa2hPa * hPa2cm

    r.psi_air = p_a #*MPa2hPa #used only with xylem
    return r

    
    
def setKrKx_phloem(r,DictVal): #inC

    #number of vascular bundles
    VascBundle_leaf = 32
    VascBundle_stem = 52
    VascBundle_root = 1 #valid for all root type
            
    #numPerBundle
    numL = 18
    numS = 21
    numr0 = 33
    numr1 = 25
    numr2 = 25
    numr3 = 1
        
    #radius of phloem type^4 * number per bundle
    rad_s_l   = numL* (0.00025 **4)# * 2; rad_x_l_2   = (0.0005 **4) * 2   
    rad_s_s   = numS *(0.00019 **4) #* 3; rad_x_s_2   = (0.0008 **4) * 1     
    rad_s_r0  = numr0 *(0.00039 **4) #* 4    
    rad_s_r12 = numr1*(0.00035**4) #* 4; rad_x_r12_2 = (0.00087**4) * 1
    rad_s_r3  = numr3 *(0.00068**4) #* 1      

    # axial conductivity [cm^3/day] , mu is added later as it evolves with CST  
    beta = 0.9 * DictVal['kx_st'] #Thompson 2003a
    kz_l   = VascBundle_leaf * rad_s_l   * np.pi /8 * beta  
    kz_s   = VascBundle_stem * rad_s_s   * np.pi /8 * beta
    kz_r0  = VascBundle_root * rad_s_r0  * np.pi /8 * beta
    kz_r12 = VascBundle_root * rad_s_r12 * np.pi /8 * beta
    kz_r3  = VascBundle_root * rad_s_r3  * np.pi /8 * beta
    
    #print([[kz_r0,kz_r12,kz_r12,kz_r3],[kz_s,kz_s ],[kz_l]])
    #raise Exception
    #radial conductivity [1/day],
    kr_l  = 0.#3.83e-4 * hPa2cm# init: 3.83e-4 cm/d/hPa
    kr_s  = 0.#1.e-20  * hPa2cm # set to almost 0
    # kr_r0 = 1e-1
    # kr_r1 = 1e-1
    # kr_r2 = 1e-1
    # kr_r3 = 1e-1
    kr_r0 = 5e-2 * DictVal['kr_st']
    kr_r1 = 5e-2 * DictVal['kr_st']
    kr_r2 = 5e-2 * DictVal['kr_st']
    kr_r3 = 5e-2 * DictVal['kr_st']
    l_kr = 0.8 #cm
    
    r.setKr_st([[kr_r0,kr_r1 ,kr_r2 ,kr_r0],[kr_s,kr_s ],[kr_l]] , kr_length_= l_kr)
    r.setKx_st([[kz_r0,kz_r12,kz_r12,kz_r0],[kz_s,kz_s ],[kz_l]])
    
    a_ST = [[0.00039,0.00035,0.00035,0.00039 ],[0.00019,0.00019],[0.00025]]
    Across_s_l   = numL*VascBundle_leaf *(a_ST[2][0]**2)*np.pi   * DictVal['Across']
    Across_s_s   = numS *VascBundle_stem * (a_ST[1][0]**2)*np.pi   * DictVal['Across']   
    Across_s_r0  = numr0 *VascBundle_root * (a_ST[0][0]**2)*np.pi   * DictVal['Across']
    Across_s_r12 = numr1*VascBundle_root * (a_ST[0][1]**2)*np.pi   * DictVal['Across']
    Across_s_r3  =  numr3 *VascBundle_root *(a_ST[0][2]**2)*np.pi   * DictVal['Across']
    
    Perimeter_s_l   = numL*VascBundle_leaf *(a_ST[2][0])* 2 * np.pi# (0.00025 **2)# * 2; rad_x_l_2   = (0.0005 **4) * 2   
    Perimeter_s_s   = numS *VascBundle_stem * (a_ST[1][0])* 2 * np.pi#(0.00019 **2) #* 3; rad_x_s_2   = (0.0008 **4) * 1     
    Perimeter_s_r0  = numr0 *VascBundle_root * (a_ST[0][0])* 2 * np.pi#(0.00039 **2) #* 4    
    Perimeter_s_r12 = numr1*VascBundle_root * (a_ST[0][1])* 2 * np.pi#(0.00035**2) #* 4; rad_x_r12_2 = (0.00087**4) * 1
    Perimeter_s_r3  =  numr3 *VascBundle_root *(a_ST[0][2])* 2 * np.pi# (0.00068**2) #* 1  
    
    r.setAcross_st([[Across_s_r0,Across_s_r12,Across_s_r12,Across_s_r0],[Across_s_s,Across_s_s],[Across_s_l]])
    return r

def weather(simDuration,condition, hp):
    vgSoil = [0.059, 0.45, 0.00644, 1.503, 1]
    loam = [0.08, 0.43, 0.04, 1.6, 50]
    Qnigh = 0; Qday = 960e-6 
    cs = 350e-6
    if condition == "wet":
        Tnigh = 15.8; Tday = 22
        RHday = 0.6; RHnigh = 0.88
        Pair = 1010.00 #hPa
        thetaInit = 0.4#
        #thetaInit = 10.47/100
    elif condition == "dry":
        Tnigh = 20.7; Tday = 30.27
        RHday = 0.44; RHnigh = 0.78
        Pair = 1070.00 #hPa
        thetaInit = 28/100 
        #thetaInit = 10.47/100
    else:
        print("condition",condition)
        raise Exception("condition not recognised")

    coefhours = 1#sinusoidal(simDuration)
    RH_ = RHnigh + (RHday - RHnigh) * coefhours
    TairC_ = Tnigh + (Tday - Tnigh) * coefhours
    Q_ = Qnigh + (Qday - Qnigh) * coefhours
     #co2 paartial pressure at leaf surface (mol mol-1)
    es =  6.112 * np.exp((17.67 * TairC_)/(TairC_ + 243.5))
    ea = es*RH_#qair2ea(specificHumidity,  Pair)
    assert ea < es
    
    assert ((RH_ > 0) and(RH_ < 1))
    bl_thickness = 1/1000 #1mm * m_per_mm
    diffusivity= 2.5e-5#m2/sfor 25*C
    rbl =bl_thickness/diffusivity #s/m 13
    #cs = 350e-6
    Kcanopymean = 1e-1 # m2/s
    meanCanopyL = (2/3) * hp /2
    rcanopy = meanCanopyL/Kcanopymean
    windSpeed = 2 #m/s
    zmzh = 2 #m
    karman = 0.41 #[-]

    rair = 1
    if hp > 0:
        rair = np.log((zmzh - (2/3)*hp)/(0.123*hp)) * np.log((zmzh - (2/3)*hp)/(0.1*hp)) / (karman*karman*windSpeed)
        #print()
        #raise Exception


    pmean = theta2H(vgSoil, thetaInit)

    weatherVar = {'TairC' : TairC_,'TairK' : TairC_ + 273.15,'Pair':Pair,"es":es,
                    'Qlight': Q_,'rbl':rbl,'rcanopy':rcanopy,'rair':rair,"ea":ea,
                    'cs':cs, 'RH':RH_, 'p_mean':pmean, 'vg':loam}
    #print("Env variables at", round(simDuration//1),"d",round((simDuration%1)*24),"hrs :\n", weatherVar)
    return weatherVar

def setDefaultVals(r,weatherX,DictVal):
    #picker = lambda x, y, z: s.pick([x, y, z])    
    #r.plant.setSoilGrid(picker)  # maps segment

    """ Parameters phloem and photosynthesis """
    
    r = setKrKx_xylem(weatherX['TairC'], weatherX['RH'],r,DictVal)
    r = setKrKx_phloem(r,DictVal)
    r.Rd_ref = 0
    r.g0 = 8e-3 # * DictVal['g0']
    r.gm = 0.025 * DictVal['gm']
    r.VcmaxrefChl1 =1.28  * DictVal['kchl1']
    r.VcmaxrefChl2 = 8.33  * DictVal['kchl2']
    r.a1 = 6  * DictVal['kg1']
    r.a3 = 1.5  * DictVal['kjmax']
    r.alpha = 0.4 * DictVal['alpha']
    r.theta = 0.6 * DictVal['omega']
    
    r.setKrm2([[2e-5 * DictVal['krm2']]])#]])
    r.setKrm1([[10e-2  * DictVal['krm1']]])#]])
    r.leafGrowthZone = 2 # cm
    r.StemGrowthPerPhytomer = True # 
    #r.psi_osmo_proto = -10000*1.0197 #schopfer2006
    
    r.Q10 *= DictVal['Q10']# X[0]
    
    delta_osmo_min = (-r.psi_osmo_proto-r.psiMin)*DictVal['delta_osmo_min']
    
    r.psiMin *= DictVal['psiMin']
    r.psi_osmo_proto = -(delta_osmo_min + r.psiMin)
    assert r.psi_osmo_proto < 0
    assert -r.psi_osmo_proto > r.psiMin
    r.KMfu *= DictVal['KMfu']
    r.Vmaxloading = 0.05 #mmol/d, needed mean loading rate:  0.3788921068507634
    r.Vmaxloading *= DictVal['Vmaxloading']
    r.CSTimin =0.4* DictVal['CSTimin']
    r.beta_loading = 0.6
    r.beta_loading *= DictVal['beta_loading']
    r.Mloading = 0.2
    r.Mloading *= DictVal['Mloading']
    r.Gr_Y = 0.8*DictVal['Gr_Y']
    r.fwr = 0#0.1
    r.sh =4e-4 * DictVal['k_fw1']
    r.p_lcrit *= DictVal['psi_t,crit,1']
    r.limMaxErr = 1/100
    r.maxLoop = 10000
    r.minLoop=100
    r.gamma0 =r.gamma0 * DictVal['gamma0']
    r.gamma1 =r.gamma1 * DictVal['gamma1']
    r.gamma2 =r.gamma2 * DictVal['gamma2']
    
    r.setRhoSucrose([[0.51 * DictVal['rhoSucrose']],[0.65 * DictVal['rhoSucrose']],[0.56 * DictVal['rhoSucrose']]])
    #r.setRhoSucrose([[0.51],[0.65 ],[0.56]])
    r.setRmax_st([[14.4 * DictVal['rr0']* DictVal['rmax'],
                   9.0 * DictVal['rr1']* DictVal['rmax'],
                   6.0 * DictVal['rr2']* DictVal['rmax'],
                   14.4 * DictVal['rr0']* DictVal['rmax']],
                  [5. * DictVal['rst']* DictVal['rmax'],
                   5. * DictVal['rst']* DictVal['rmax']],
                  [15. * DictVal['rle']* DictVal['rmax']]])
    
    
    r.sameVolume_meso_st = False
    r.sameVolume_meso_seg = True
    r.withInitVal =True
    r.initValST = r.CSTimin
    r.initValMeso =r.CSTimin
    

    r.cs = weatherX["cs"]
    r.oi = r.oi * DictVal['oi']

    r.expression = 6
    r.update_viscosity = True
    r.solver = 1
    r.atol = 1e-12
    r.rtol = 1e-8
    #r.doNewtonRaphson = False;r.doOldEq = False
    SPAD= 41.0
    chl_ = (0.114 *(SPAD**2)+ 7.39 *SPAD+ 10.6)/10
    r.Chl = np.array( [chl_ * DictVal['Chl']]) #])
    r.Csoil = 1e-4
    return r

def resistance2conductance(resistance,weatherX,r):
    resistance = resistance* (1/100) #[s/m] * [m/cm] = [s/cm]
    resistance = resistance * r.R_ph * weatherX["TairK"] / r.Patm # [s/cm] * [K] * [hPa cm3 K−1 mmol−1] * [hPa] = [s] * [cm2 mmol−1]
    resistance = resistance * (1000) * (1/10000)# [s cm2 mmol−1] * [mmol/mol] * [m2/cm2] = [s m2 mol−1]
    return 1/resistance

def SobolTest(myid,weatherX, varNames, varLims, simDuration,condition, testType,power2,testtype):
    
    
    assert testType=="Xylem" or testType == "Phloem" or testType == "SEB"
    test_values = varLims.copy()
    DictVal = {}
    for key in varNames:
        for value in test_values:
            DictVal[key] = value
            test_values.remove(value)
            break
    
    assert len(varLims) == len(varNames)
    pl = pb.MappedPlant(seednum = 2) 
    path = CPBdir+"/modelparameter/plant/"
    name = "Triticum_aestivum_adapted_2023"
    #print(os.getcwd())
    #print(path + name + ".xml")

    pl.readParameters(path + name + ".xml")
    depth = 60
    sdf = pb.SDF_PlantBox(np.Inf, np.Inf, depth )
    pl.setGeometry(sdf) # creates soil space to stop roots from growing out of the soil
    
    
        
    for p in pl.getOrganRandomParameter(pb.leaf):
        p.lmax *= DictVal['lmaxle']
        p.tropismN *= DictVal['tropismNle']
        p.tropismS *= DictVal['tropismSle']
        p.r *= DictVal["rle"]
        p.a *= DictVal["ale"]
        p.theta *= DictVal["thetale"]
        p.Width_petiole *= DictVal["Width_petiole"]
        p.Width_blade *= DictVal["Width_blade"]

    for p in pl.getOrganRandomParameter(pb.stem):
        p.lmax *= DictVal['lmaxst']
        p.tropismN *= DictVal['tropismNst']
        p.tropismS *= DictVal['tropismSst']
        p.r *= DictVal["rst"]
        p.a *= DictVal["ast"]
        p.la *= DictVal["last"]
        p.lb *= DictVal["lbst"]
        p.ln *= DictVal["lnst"]
        p.delayLat *= DictVal["delayLat"]
        p.delayNGStart *= DictVal["delayNGStart"]
        p.delayNGEnd *= DictVal["delayNGEnd"]

    for p in pl.getOrganRandomParameter(pb.root):
        if (p.subType ==0):
            pass
        elif (p.subType ==1)or(p.subType >3):
            p.lmax *= DictVal['lmaxr0']
            p.tropismN *= DictVal['tropismNr0']
            p.tropismS *= DictVal['tropismSr0']
            p.r *= DictVal["rr0"]
            p.a *= DictVal["ar0"]
            p.la *= DictVal["lar0"]
            p.lb *= DictVal["lbr0"]
            p.ln *= DictVal["lnr0"]
            p.theta *= DictVal["thetar0"]
        elif (p.subType ==2):
            p.lmax *= DictVal['lmaxr1']
            p.tropismN *= DictVal['tropismNr1']
            p.tropismS *= DictVal['tropismSr1']
            p.r *= DictVal["rr1"]
            p.a *= DictVal["ar1"]
            p.la *= DictVal["lar1"]
            p.lb *= DictVal["lbr1"]
            p.ln *= DictVal["lnr1"]
            p.theta *= DictVal["thetar1"]
        elif (p.subType ==3):
            p.lmax *= DictVal['lmaxr2']
            p.tropismN *= DictVal['tropismNr2']
            p.tropismS *= DictVal['tropismSr2']
            p.r *= DictVal["rr2"]
            p.a *= DictVal["ar2"]
            p.theta *= DictVal["thetar2"]
        else:
            print("root subtype not recognized",p.subType)
            raise Exception("root subtype not recognized")
            
    for p in pl.getOrganRandomParameter(pb.seed):
        p.maxTil =int(round(p.maxTil* DictVal['maxTil']))
        p.firstTil *= DictVal['firstTil']
        p.delayTil *= DictVal['delayTil']
        p.firstB *= DictVal['firstB']
        p.delayB *= DictVal['delayB']
        p.maxB = int(round(p.maxB*DictVal["maxB"]))
    
    pl.initialize(verbose = True)#, stochastic = False)
    pl.simulate(simDuration, False)#, "outputpm15.txt")

    picker = lambda x,y,z : max(int(np.floor(-z)),-1)   
    pl.setSoilGrid(picker)  # maps segment


    r = PhloemFluxPython(pl,psiXylInit =-1000,ciInit = 0.5 *weatherX['cs'])


    hp = max([tempnode[2] for tempnode in r.get_nodes()]) /100
    weatherX = weather(simDuration, condition, hp)

    plant =setDefaultVals(r,weatherX,DictVal)

    #xXx = np.full((len(r.plant.nodes)),weatherX["p_mean"])
    p_bot = weatherX["p_mean"] + depth/2
    p_top = weatherX["p_mean"] - depth/2
    xXx = np.linspace(p_top, p_bot, depth)
    


    plant.Patm = weatherX["Pair"]
    ##resistances
    plant.g_bl = resistance2conductance(weatherX["rbl"],weatherX,r) / r.a2_bl
    plant.g_canopy = resistance2conductance(weatherX["rcanopy"],weatherX,r) / r.a2_canopy
    plant.g_air = resistance2conductance(weatherX["rair"],weatherX,r) / r.a2_air

    verbose_phloem = True
    #plant.doTroubleshooting=True
    if testType == "Phloem":
        directoryN = "/sobolST/"
    elif testType == "Xylem":
        directoryN = "/sobolX/"
    elif testType == "SEB":
        directoryN = "./results/sobolSEB/"+str(power2)+condition+str(simDuration)+testtype+"/"
    filename =directoryN+ "inPM"+repr(myid)+".txt"
    try:

        plant.solve_photosynthesis(sim_time_ = simDuration, sxx_=xXx, 
                                   cells_ = True,ea_ = weatherX["ea"],es_ = weatherX["es"],
                verbose_ = False, doLog_ = False,TairC_= weatherX["TairC"] ,outputDir_= "./results"+directoryN)
        
            
            
        plant.startPM(simDuration, simDuration + 3/(24), 1, ( weatherX["TairC"]  +273.15) , verbose_phloem, filename)
        Nt = len(plant.plant.nodes) 
        Q_Rm    = np.array(plant.Q_out[(Nt*2):(Nt*3)])
        Q_Exud  = np.array(plant.Q_out[(Nt*3):(Nt*4)])
        Q_Gr    = np.array(plant.Q_out[(Nt*4):(Nt*5)])
        
        Ev = sum(plant.Ev)#*1/(24)
        Ag = np.sum(plant.Ag4Phloem)#*1/(24)
        
        try:
            os.remove(filename)
        except OSError:
            pass
        loopsdone = plant.loop 

        #del plant
        #del pl
        Yexud = sum(Q_Exud)
        Ygr = sum(Q_Gr)*r.Gr_Y
        Yrm = sum(Q_Rm)
        if Ev >0:
            WUE = Ygr/Ev
            IWUE = Ag/Ev
        else:
            WUE = 0
            IWUE =0
            name2 =directoryN+'IDnoEv.txt'
            print(repr(simDuration)+" "+repr( myid) +" "+repr( plant.loop) +" no Ev"+'\n')
            #with open(name2, 'a') as log:
             #   log.write(repr(simDuration)+" "+repr( myid) +" "+repr( plant.loop) +" no Ev"+'\n')
              #  log.write(repr(DictVal) +'\n')
        del plant 
        del pl
        print("finished thourgh",myid)
        return Yexud,Ygr,Yrm, Ev,Ag,WUE,IWUE, loopsdone,myid
    
    except Exception as e:
        #print(repr(simDuration)+" "+repr( myid) +" "+repr( plant.loop) +" "+str(e) +'\n')
        #print(repr(DictVal) +'\n')
        #raise Exception
        name2 =directoryN+'IDError.txt'
        with open(name2, 'a') as log:
            log.write(repr(simDuration)+" "+repr( myid) +" "+repr( plant.loop) +" "+str(e) +'\n')
            #log.write(repr(DictVal) +'\n')
        loops = plant.loop
        del plant
        del pl
        return 0,0,0,0,0,0,0,loops,myid #np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan
        
        #raise Exception()
            
        