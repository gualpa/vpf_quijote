def read_quijote(dir, cosm, snapnum, axis, space):
    """
    Returns array with Halo positions in Mpc/h
    """
    #import numpy as np
    import readgadget
    import readfof

    #-----------
    # Read data from Quijote
    #-----------

    # get the name of the corresponding snapshot
    snapshot = '/home/fdavilakurban/Proyectos/VPF_Quijote/data/quijote/Snapshots/%s/0/snapdir_%03d/snap_%03d'%(cosm,snapnum,snapnum)
    
    # read the redshift, boxsize, cosmology...etc in the header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    #Nall     = header.nall         #Total number of particles
    #Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
    Omega_m  = header.omega_m      #value of Omega_m
    Omega_l  = header.omega_l      #value of Omega_l
    h        = header.hubble       #value of h
    redshift = header.redshift     #redshift of the snapshot
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l) #Value of H(z) in km/s/(Mpc/h)

    print(f'BoxSize = {BoxSize} Mpc/h')
    #print('Number of particles in the snapshot:',Nall)
    print(f'Omega_m = {Omega_m}')
    #print('Omega_l = %.3f'%Omega_l)
    print(f'h = {h}')
    print(f'redshift = {redshift:.1f}')
    #print(f'Omega_b = {header.omega_b}')
    #print(f'sigma_8 = {header.sigma_8}')
    #print(f'ns = {header.ns}')

    # read the halo catalogue
    #path = dir+'/'+cosm+'/'+simnum
    FoF = readfof.FoF_catalog(filepath, snapnum, long_ids=False,
                            swap=False, SFR=False, read_IDs=False)
    pos_h  = FoF.GroupPos/1e3            #Halo positions in Mpc/h
    vel_h  = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s

    if space == 'zspace':
    	RSL.pos_redshift_space(pos_h, vel_h, BoxSize, Hubble, redshift, axis)

    return pos_h, vel_h, BoxSize  #Halo positions in Mpc/h

def delta_P0(P0,Nran):
    """Calculates error for P0 as derived in Colombi et al 1995
    Args:

        P0(numpy array): value(s) of P0
        Nran(numpy array): number of volume samples in the data

    Returns:
        array: uncertainty of P0
    """

    import numpy as np
    return np.sqrt(P0*(1-P0)/Nran)

def delta_chi(chi,P0,P0err,N_mean,N_mean_std):
    """Calculates error for chi as derived in Colombi et al 1995 (Fry et al 2013)
    Args:

        chi(numpy array): value(s) of chi
        P0(numpy array): value(s) of P0
        P0err (numpy array): delta P0
        N_mean(numpy array): mean number of objects in volume(r)
        N_mean_std(numpy array): uncertainty of N_mean calculated with JK resampling

    Returns:
        array: uncertainty of chi
    """
    import numpy as np

    return chi*abs(P0err/(P0*abs(np.log(P0)))-N_mean_std/N_mean)

def vpf_new(rmin,rmax,rbins,njk,nsph,BoxSize,gxs,verbose=True):

    from scipy import spatial
    from cicTools import delta_chi, delta_P0, perrep_array

    # Basic Parameters
    #rmin, rmax = 3., 20. #Mpc
    #rbins = 10
    rs = np.geomspace(rmin,rmax,rbins)
    lbox = BoxSize

    # Jackknife Parameters
    #njk = 10 #N. of Jackknifes
    jackk_bins = np.linspace(0,lbox,njk+1)

    newgxs = perrep_array(gxs,lbox,rmax)

    data_tree = spatial.cKDTree(newgxs)

    # Probing Spheres
    nsph = 100000
    sph = lbox*np.random.rand(nsph,3)

    # Calculations
    N_mean_jk = np.zeros((njk,rbins))
    #N_var_jk = np.zeros((njk,rbins))
    P0_jk = np.zeros((njk,rbins))
    xi_jk = np.zeros((njk,rbins))

    sph = lbox*np.random.rand(njk,nsph,3)


    for i in range(njk):
        if verbose==True: print(i,'/',njk,'jk')
        mask2 = (sph[i][:,0] < jackk_bins[i+1])
        mask1 = (sph[i][:,0] > jackk_bins[i])        
        mask_inv = np.logical_and(mask1,mask2)

        mask = np.invert(mask_inv)
        sph_yes = sph[i][mask,:]
        
        for j in range(rbins):
            ngal = data_tree.query_ball_point(sph_yes,rs[j],return_length=True)

            P0_jk[i][j] = len(np.where(ngal==0)[0])/len(sph_yes)
            #print(np.mean(ngal))
            N_mean_jk[i][j] = np.mean(ngal)
            #N_var_jk[i][j] = np.var(ngal,ddof=1)/njk
            xi_jk[i][j] = (np.mean((ngal-N_mean_jk[i][j])**2)-N_mean_jk[i][j])/N_mean_jk[i][j]**2



    #<N>
    N_mean = np.mean(N_mean_jk,axis=0)
    N_mean_var = np.var(N_mean_jk,axis=0,ddof=1)/njk
    #P0
    P0 = np.mean(P0_jk,axis=0)
    #P0_var = np.var(P0_jk,axis=0,ddof=1)/njk
    #chi
    chi = -np.log(P0, out=np.zeros_like(P0), where=(P0!=0))/N_mean
    chi[chi==0] = np.nan
    chi_std = delta_chi(chi,P0,delta_P0(P0,nsph),N_mean,np.sqrt(N_mean_var))
    #xi
    xi = np.mean(xi_jk,axis=0)
    xi_var = np.var(xi_jk,axis=0,ddof=1)/njk

    return N_mean, N_mean_var, P0, chi, chi_std, xi, xi_var

###########################################################
###########################################################
###########################################################



import multiprocessing
import os
import yaml


def process_file(file_path):
    """Function to process a single file"""
    # Simulate file processing (replace this with actual data analysis code)
    # print(f"Processing {file_path}")
    # # Here, you could read and process the file
    # with open(file_path, 'r') as f:
    #     data = f.read()
    # # Return some result (e.g., analysis result)
    # return f"Processed {file_path}: {len(data)} characters"


    import os
    import numpy as np
    from astropy.io import ascii
    from astropy.table import Table
    import redshift_space_library as RSL
    
    # Load YAML configuration
    with open('/home/fdavilakurban/Proyectos/VPF_Quijote/codes/config_cosm.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Access the env variables
    cosm = str(os.getenv("cosm"))
    #simbatch = str(os.getenv("simbatch"))
    
    # Access the config variables
    halodir = config["settings"]["snapdir"]
    snapnum = config["settings"]["snapnum"]
    ns = config["settings"]["ns"]
    rbin = config["settings"]["rbin"]
    rmin = config["settings"]["rmin"]
    rmax = config["settings"]["rmax"]
    njk = config["settings"]["njk"]
    space = config["settings"]["space"]
    axis = config["settings"]["axis"]
    
    #simnums = os.listdir(snapdir+cosm+'/')
    
    #for simnum in simnums:
        
    print(filepath)

    gxs, vel, BoxSize = read_quijote(filepath,cosm,snapnum,axis,space)
    
    N_mean = np.zeros(rbin)
    N_mean_var = np.zeros(rbin)
    P0 = np.zeros(rbin)
    chi = np.zeros(rbin)
    chi_std = np.zeros(rbin)
    xi = np.zeros(rbin)
    xi_var = np.zeros(rbin)
    
    N_mean, N_mean_var, P0, chi, chi_std, xi, xi_var = vpf_new(rmin,rmax,rbin,njk,ns,BoxSize,gxs,verbose=False)
    vpfdata = Table()
    vpfdata['N_mean'] = N_mean
    vpfdata['N_mean_var'] = N_mean_var
    vpfdata['P0'] = P0
    vpfdata['chi'] = chi
    vpfdata['chi_std'] = chi_std
    vpfdata['xi'] = xi
    vpfdata['xi_var'] = xi_var

    return vpfdata
    #datafilename = f'../data/output/vpfdata_{rmin}-{rmax}-{rbin}-{njk}-{ns}-{cosm}-{space}{axis}-{simnum}.dat'
    #ascii.write(vpfdata,datafilename,overwrite=True)
    #print('File created:',datafilename)

def main():

    # Load YAML configuration
    with open('/home/fdavilakurban/Proyectos/VPF_Quijote/codes/config_cosm.yml', 'r') as file:
        config = yaml.safe_load(file)
    # Access the env variables
    cosm = str(os.getenv("cosm"))
    # Access the config variables
    halodir = config["settings"]["snapdir"]
    snapnum = config["settings"]["snapnum"]
    ns = config["settings"]["ns"]
    rbin = config["settings"]["rbin"]
    rmin = config["settings"]["rmin"]
    rmax = config["settings"]["rmax"]
    njk = config["settings"]["njk"]
    space = config["settings"]["space"]
    axis = config["settings"]["axis"]
    
    """Main function to parallelize file processing"""
    # Get a list of files in the directory
    directory = halodir+cosm
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Use multiprocessing to process files in parallel
    num_cpus = multiprocessing.cpu_count()  # Use all available CPUs
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(process_file, files)
    
    # Print or save results
    for result in results:
        datafilename = f'../data/output/vpfdata_{rmin}-{rmax}-{rbin}-{njk}-{ns}-{cosm}-{space}{axis}-{simnum}.dat'
        ascii.write(vpfdata,datafilename,overwrite=True)
        print('File created:',datafilename)
        
if __name__ == "__main__":
    main()

