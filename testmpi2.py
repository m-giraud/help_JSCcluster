from mpi4py import MPI
import numpy as np
np.random.seed(25)
import pickle
import time
from helpUqr_seb import *
import os


                 
def runallSobol(start,rank,simDuration, condition,size, params_,testtype,namesVarsAll,namesLim_same):
    weatherX = weather(simDuration, condition, 1)
    
    result = np.array([SobolTest(start+i,weatherX, 
             namesVarsAll,  list(params)+namesLim_same, #all the var values
             simDuration,condition,"SEB",size,testtype) for i, params in enumerate(params_) ])
    return result
             
def testmpi(simDuration, condition,nrun,testtype):
    # get number of processors and processor rank
    print(testtype)
    if testtype =='functional' :
        
        namesVars = ['Chl','oi','k_fw1','psi_t,crit,1',
        'kchl1', 'kchl2','kg1','kjmax', 'alpha',
         'omega','gamma0', 'gamma1', 'gamma2','gm',
        'kx_x','kr_x',
        'Q10','delta_osmo_min','psiMin','KMfu','Vmaxloading','CSTimin',
        'beta_loading', 'Mloading','Gr_Y','rhoSucrose',
        'kx_st','kr_st','Across','krm2','krm1','rmax'] 
        
        namesVars_same = ["maxTil" ,"firstTil","delayTil" ,"maxB","firstB","delayB",
        "delayLat","delayNGStart","delayNGEnd",
        "lar0", "lbr0", "lnr0", "lmaxr0", "rr0", "ar0", "tropismNr0", "tropismSr0", "thetar0",
        "lar1", "lbr1", "lnr1", "lmaxr1", "rr1", "ar1", "tropismNr1", "tropismSr1", "thetar1",
        "lmaxr2", "rr2", "ar2", "tropismNr2", "tropismSr2", "thetar2",
        "last", "lbst", "lnst", "lmaxst", "rst", "ast", "tropismNst", "tropismSst",
        "lmaxle", "rle", "ale", "tropismNle", "tropismSle", "thetale","Width_petiole","Width_blade"]
        namesLim_same = [1 for ii in namesVars_same]
    elif testtype =='All' : 
        namesVars = ["maxTil" ,"firstTil","delayTil" ,"maxB","firstB","delayB",
        "delayLat","delayNGStart","delayNGEnd",
        "lar0", "lbr0", "lnr0", "lmaxr0", "rr0", "ar0", "tropismNr0", "tropismSr0", "thetar0",
        "lar1", "lbr1", "lnr1", "lmaxr1", "rr1", "ar1", "tropismNr1", "tropismSr1", "thetar1",
        "lmaxr2", "rr2", "ar2", "tropismNr2", "tropismSr2", "thetar2",
        "last", "lbst", "lnst", "lmaxst", "rst", "ast", "tropismNst", "tropismSst",
        "lmaxle", "rle", "ale", "tropismNle", "tropismSle", "thetale","Width_petiole","Width_blade",
        'Chl','oi','k_fw1','psi_t,crit,1',
        'kchl1', 'kchl2','kg1','kjmax', 'alpha',
        'omega','gamma0', 'gamma1', 'gamma2','gm',
        'kx_x','kr_x',
        'Q10','delta_osmo_min','psiMin','KMfu','Vmaxloading','CSTimin',
        'beta_loading', 'Mloading','Gr_Y','rhoSucrose',
        'kx_st','kr_st','Across','krm2','krm1'] 
        namesVars_same = ['rmax']
        namesLim_same = [1 for ii in namesVars_same]
        
    namesVarsAll =namesVars+namesVars_same
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    main_dir=os.getcwd()
    results_dir = main_dir +"/results/sobolSEB/"+str(nrun)+condition+str(simDuration)+testtype+"/"

    with open('param_values'+testtype+str(nrun)+'.pkl','rb') as f: # Use this to load temporary or final results                                  
        params=pickle.load(f)              # together with every used variable
        
    n = len(params) #number of runs to do
    count = n // size  # number of catchments for each process to analyze
    remainder = n % size  # extra catchments if n is not a multiple of size

    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (count + 1)  # index of first catchment to analyze
        stop = start + count + 1  # index of last catchment to analyze
    else:
        start = rank * count + remainder
        stop = start + count

    local_params = params[start:stop, :]  # get the portion of the array to be analyzed by each rank
    if(len(local_params)>0):
        outputsize=9
        print(len(local_params[0]), len(namesVars))
        assert len(local_params[0])== len(namesVars)
        assert len(namesLim_same)==len(namesVars_same)
        local_results = np.empty((local_params.shape[0], outputsize))  # create result array
        #local_results[:, :local_params.shape[1]] = local_params  # write parameter values to result array
        #SobolTest(myid,weatherX, varNames, varLims, simDuration,condition, testType,power2):
        local_results[:, :] = runallSobol(start,rank,simDuration, condition,nrun, local_params,testtype,namesVarsAll,namesLim_same)  

        Yall = [np.array([]) for i in range(7)]#Yexuds,Ygrs,Yrms,Ytrans,Yassi,Ywue]
        namesY = ["Yexuds","Ygrs","Yrms","Ytrans","Yassi","Ywue","YIwue"]
        allLoops = np.array([])
        allIds = np.array([])

        # send results to rank 0
        if rank > 0:
            comm.Send(local_results, dest=0, tag=14)  # send results to process 0
        else:
            final_results = np.copy(local_results)  # initialize final results with results from process 0
            for i in range(1, min(size, len(params))):  # determine the size of the array to be received from each process
                if i < remainder:
                    rank_size = count + 1
                else:
                    rank_size = count

                tmp = np.empty((rank_size, final_results.shape[1]), dtype=float)  # create empty array to receive results
                comm.Recv(tmp, source=i, tag=14)  # receive results from the process
                final_results = np.vstack((final_results, tmp))  # add the received results to the final results

            print("results")

            for j in range(len(Yall)):
                Yall[j] = np.concatenate((Yall[j] ,[smalldic[j] for smalldic in final_results ]))
            allLoops   = np.concatenate((allLoops,[smalldic[7] for smalldic in final_results ])) 
            allIds   = np.concatenate((allIds,[smalldic[8] for smalldic in final_results ])) 
            #print(Yall)

            addToName = repr(nrun)+"_"+repr(simDuration)+"_"+condition+testtype

            test_values = Yall.copy()
            DictVal = {}
            for key in namesY:
                for value in test_values:
                    DictVal[key] = value
                    test_values.remove(value)
                    break

            print("finished sobol",'./results/sobolSEB/'+str(nrun)+condition+str(simDuration)+testtype+"/"+'YAlls_R'+addToName+'.pkl')
            print("length y'all",len(Yall),len(Yall[0]))
            with open('results/sobolSEB/'+str(nrun)+condition+str(simDuration)+testtype+"/"+'YAlls_R'+addToName+'.pkl','wb') as f:
                 pickle.dump(DictVal,f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('results/sobolSEB/'+str(nrun)+condition+str(simDuration)+testtype+'/loops.txt', 'w') as log:
                log.write(','.join([num for num in map(str, allLoops)])  +'\n')
            with open('results/sobolSEB/'+str(nrun)+condition+str(simDuration)+testtype+'/threadIds.txt', 'w') as log:
                log.write(','.join([num for num in map(str, allIds)])  +'\n')
    

simDuration=int(sys.argv[3]); condition =sys.argv[2];reps=int(sys.argv[1])
testtype = sys.argv[4]
testmpi(simDuration, condition,reps,testtype)
