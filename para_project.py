import numpy as np
import time
import functions

def read_cadences(labels):
    """
    Function for reading OpSims in different filters.
    
    Parameters:
    -----------
    labels: list
        List of OpSim filenames as stored in ./data/filter_name/
    
    Returns:
    --------
    mjds_g: np.array
        Sorted OpSim cadences (days of the survey when there was a telescope visit).
    """
    
    # Initialize a list for opsim cadence arrays (for g and r filters)
    mjds_g = []
    #mjds_r = []
    
    # Load OpSim cadences
    for lb in labels:
        mjd_g = np.loadtxt('data/g/'+lb+'.dat')
        #mjd_r = np.loadtxt('data/r/'+lb+'.dat')
        mjds_g.append(np.sort(mjd_g))
        #mjds_r.append(np.sort(mjd_r))
    print("Data reading finished")

    return mjds_g


def SF_calc(mjd, nlc=50, frame='observed', redshift_bins=(0.5,7.5)):
    """
    Main function for performing Structure Function Metric calculations.
    By default, it generates 50 densely sampled artificial light curves based 
    on Damped Random Walk process for 9 redshift bins. Each of those light curves
    is used to evaluate flux in days when the survey visits a given location,
    providing sparsely sampled LSST OpSim light curves. Then, we compare ideal
    and LSST OpSim light curves (i.e. their structure functions) to evaluate the
    error (metric). In this way, we can compare how well different OpSim strategies
    are able to retreive good light curve parameters. Final result is stored in 
    matrix form. 
    
    Parameters:
    -----------
    mjd: np.array
        Days with observations in a given OpSim realization.
    nlc: int, default=50
        Number of light curves generated for each redshift bin.
    frame: {'observed', 'rest'}, default='observed'
        Frame of reference.
    redshift_bins: tuple
    	Redshift range for defining bins.
        
    Returns:
    -----------
    raz2: np.array (dim: 9 x 99)
        Matrix containing SF metric values for 9 redshift bins.
    """
    
    # Define redshift bins
    (z_min, z_max) = redshift_bins
    num_bins = int((z_max - z_min) + 1)  # number of bins
    zbin = np.linspace(z_min, z_max, num_bins)
    
    # Add z=0
    #zbin = np.insert(zbin,0,0)
    
    # Converting MJD to survey days
    long=int(mjd.max()-mjd.min()+1)
    swop=[]
    wedgeop=[]
    scop=[]
    edgecop=[]
    
    np.random.seed(0)
    # Generate a number (nlc) of light curves for each redshift bin
    for z in zbin:
        for w in range(nlc):
            # Generating continuous light curve (cadence = 1 day)
            tt, yy = functions.LC_conti(long, z=z, frame=frame)
            # Calculating structure function for the current continuous light curve
            sn, edgesn = functions.sf(tt,yy,z=z)
            scop.append(sn)
            edgecop.append(edgesn)
            # Generating OpSim light curve evaluated on the current continuous light curve
            top,yop = functions.LC_opsim(mjd,tt,yy)
            # Calculating structure function for the current OpSim light curve
            srol,edgesrol = functions.sf(top,yop,z=z)
            swop.append(srol)
            wedgeop.append(edgesrol)

    swop=np.asarray(swop)
    swop=swop.reshape(num_bins,nlc,99)
    scop=np.asarray(scop)
    scop=scop.reshape(num_bins,nlc,99)
    razrol=[]
    
    # SF metric calculation
    for z in range(num_bins):
        for r in range(nlc):
            razrol.append((np.nan_to_num(np.sqrt(scop[z,r,:]))-np.nan_to_num(np.sqrt(swop[z,r,:]))))
    
    razrol=np.asarray(razrol)
    razrol=razrol.reshape(num_bins,nlc,99)
    # We take the mean light curve for each redshift bin.
    raz2=np.nanmean(razrol[:,:,:],axis=1)    
     
    return raz2


############################# Parallel code ###################################


from mpi4py import MPI
comm = MPI.COMM_WORLD  # Initialize MPI communicator
size = comm.Get_size() # Number of processes
rank = comm.Get_rank() # Process rank (id)

start_time = time.time()

# Note: Test is performed using reduced number of OpSims, light curves and redshift bins.

#--------#
# INPUT: #
#--------#

# --> 12 OpSims used for testing
labels = ['agnddf_v1.5_10yrs', 'alt_dust_v1.5_10yrs','baseline_samefilt_v1.5_10yrs'
      ,'bulges_cadence_bs_v1.5_10yrs', 'daily_ddf_v1.5_10yrs','dcr_nham2_ug_v1.5_10yrs'
      ,'filterdist_indx5_v1.5_10yrs', 'footprint_gp_smoothv1.5_10yrs'
      ,'goodseeing_gri_v1.5_10yrs', 'greedy_footprint_v1.5_10yrs'
      ,'roll_mod2_dust_sdf_0.20_v1.5_10yrs','rolling_mod2_sdf_0.20_v1.5_10yrs'
      #,'short_exp_2ns_5expt_v1.5_10yrs', 'spiders_v1.5_10yrs','third_obs_pt45v1.5_10yrs'
      #,'twilight_neo_mod2_v1.5_10yrs','u60_v1.5_10yrs', 'var_expt_v1.5_10yrs',
      #'wfd_depth_scale0.90_v1.5_10yrs'
      ]

# --> number of simulated light curves per redshift bin
nlc = 2 

# --> redshift interval
redshift_bins = (0.5,2.5)

#-------------------------------------------------------------------------------

# Define amount of work for each process 
num_per_rank = len(labels) // size  # number of opsims to evaluate in each process
remainder = len(labels) % size  # remainder, num. of opsims to evaluate in sequential mode

# Data reading on root (sequential mode)
if rank == 0:
    mjds_g = read_cadences(labels)
    opsims_to_share = [] # list of opsim groups for sharing
    labels_to_share = [] # list of associated labels (for printing purposes)
    for p in range(size):
        lim1 = p * num_per_rank
        lim2 = p * num_per_rank + num_per_rank
        opsims_to_share.append(mjds_g[lim1:lim2])
        labels_to_share.append(labels[lim1:lim2])

# Usually, variables that have something to do with all processes are explicitly
# defined for every one of them, even if they are empty at the moment.
else:
    opsims_to_share = None
    labels_to_share = None
  
# Wait for all processes to finish
comm.Barrier()

# Root process scatters data to other processes
recvbuf_data = comm.scatter(opsims_to_share, root=0)
recvbuf_label = comm.scatter(labels_to_share, root=0)
print("Worker %d" %rank + " received data for %s" %recvbuf_label)
#print("Data on %d" %rank + " is", recvbuf_data)

# Calculation performed by each process on its share of data
temp = []
for i in range(num_per_rank):
    # We will generate only 2 light curves per redshift bin (instead of 50), 
    # for the sake of testing.
    raz2 =  SF_calc(recvbuf_data[i], nlc=nlc, redshift_bins=redshift_bins)
    temp.append(raz2)
print("Worker %d" %rank + " has finished calculations")

comm.Barrier()

# Root process starts to gather processed data from other porcesses
R_fin_g = comm.gather(temp, root=0)

if rank==0:
    print("Worker %s" %rank + " finished gathering the data")
    
    # Convert 3D list to 2D list    
    R_fin_g = [e for sl in R_fin_g for e in sl]
    
    # Saving
    for pic_g, lb in zip(R_fin_g, labels):
        np.savetxt('results/' + lb + '_g_para.csv', pic_g, delimiter=",")
    
    print("Finished saving first {} opsims".format(num_per_rank*size))
    
    # Calculate and save the results for remaining opsims
    if remainder > 0:
        print("Begin calculation for the remaining {} opsims:".format(remainder), labels[-remainder:])
        for i in range(remainder):
            tail_idx = i+1
            raz2 = SF_calc(mjds_g[-tail_idx], nlc=nlc, redshift_bins=redshift_bins)
            lb = labels[-tail_idx]
            np.savetxt('results/' + lb + '_g_para.csv', raz2, delimiter=",")
    
    print("Time spent with ", size, " processes in seconds:")
    print("-----", np.round((time.time() - start_time),2), "-----")
    
    print("Finished!")
    





