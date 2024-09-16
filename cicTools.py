
#%%

def cic_stats(tree, n, r, lbox):
    """Returns Counts in Cells statistics

    Args:
        tree (ckdtree): coordinates
        n (int): Num of spheres
        r (float): Radius of the spheres
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        float: VPF
        float: Mean number of points in spheres of radius r
        float: Averaged 2pcf (variance of counts in cells)
    """
    import numpy as np
    from scipy import spatial
    
    a = 0
    b = lbox    
    # (b - a) * random_sample() + a
    spheres = (b-a)*np.random.rand(n,3) + a
    #spheres_tree = spatial.cKDTree(spheres)


    #ngal: Num de gxs en cada esfera de radio r
    #ngal = [len(a) for a in spheres_tree.query_ball_tree(tree,r)] 

    #Otra forma de obtener ngal:
    ngal = np.zeros(n)
    for k in range(n):
        ngal[k] = len(tree.query_ball_point(spheres[k],r))


    #VPF
    P0 = len(np.where(ngal==0)[0])/n

    N_mean = np.mean(ngal)

    #xi_mean
    xi_mean = (np.mean((ngal-N_mean)**2)-N_mean)/N_mean**2

    chi = -np.log(P0)/N_mean

    NXi = N_mean*xi_mean
    
    del ngal
    
    return chi, NXi, P0, N_mean, xi_mean

#%%
def readTNG(snap=99,minmass=-1.,maxmass=3.):
    """
    Read subhalos/galaxies in the TNG300-1 simulation 

    Args:
        disk (string): where to read TNG. 'local' or 'mounted'
        snap (int): snapshot number
        minmass, maxmass (float): log10 of min/max mass thresholds (e.g.: -1 means 1E-9Mdot, 3 means 1E13Mdot)
            
    Returns:
        ascii Table: gxs (Position, Mass, Velocity)

    """
    import os
    import sys
    illustrisPath = '/home/fdavilakurban/'
    #if disk=='local':
    basePath = '../../../TNG300-1/output/'
    #elif disk=='mounted':

    sys.path.append(illustrisPath)
    import illustris_python as il
    import numpy as np
    from astropy.table import Table
    import random 
    
    try:
        mass = il.groupcat.loadSubhalos(basePath,snap,fields=['SubhaloMass'])
    except FileNotFoundError: 
        os.system(f'mkdir {basePath}groups_{snap:03}')
        os.system(f'sshfs fdavilakurban@clemente:/mnt/simulations/illustris/TNG300-1/groups_{snap:03}/ ../../../TNG300-1/output/groups_{snap:03}/')
    except FileExistsError:               
        os.system(f'sshfs fdavilakurban@clemente:/mnt/simulations/illustris/TNG300-1/groups_{snap:03}/ ../../../TNG300-1/output/groups_{snap:03}/')
    finally: 
        mass = il.groupcat.loadSubhalos(basePath,snap,fields=['SubhaloMass'])
  
    #except:
    #    os.system(f'sshfs fdavilakurban@clemente:/mnt/simulations/illustris/TNG300-1/\
    #              groups_{snap:03}/ ../../../TNG300-1/output/groups_{snap:03}/')
                                                                                           
    ids = np.where((np.log10(mass)>minmass)&(np.log10(mass)<maxmass))
    mass = mass[ids]

    pos = il.groupcat.loadSubhalos(basePath,snap,fields=['SubhaloPos'])
    pos = pos[ids]

    vel = il.groupcat.loadSubhalos(basePath,snap,fields=['SubhaloVel'])
    vel = vel[ids]


    #gxs = Table(np.column_stack([pos[:,0],pos[:,1],pos[:,2],mass]),names=['x','y','z','mass'])    
    gxs = Table(np.column_stack([pos[:,0],pos[:,1],pos[:,2],vel[:,0],vel[:,1],vel[:,2]]),\
                            names=['x','y','z','vx','vy','vz'])    
    
    del mass,pos,vel

    return gxs

#%%
def cic_stats_jk(tree, n, r, lbox, jkbins):
    """
    Returns Counts in Cells statistics with Jackknife resampling

    Args:
        tree (ckdtree): coordinates
        n (int): Num of spheres
        r (float): Radius of the spheres
        seed (int, optional): Random seed. Defaults to 0.
        jkbins (int): Num. of divisions per axis for JK resampling

    Returns:
        Jackknife Mean and Standard Dev.:
        float: Reduced VPF
        float: Scaling Variable (<N>*<Xi>)
        float: VPF
        float: Mean number of points in spheres of radius r (<N>)
        float: Averaged 2pcf (variance of counts in cells, <Xi>)

    """
    import numpy as np
    from scipy import spatial
    
    a = 0
    b = lbox
    n_ = int(n*2) # I oversample the space and then choose the first n spheres after applying JK mask

    np.random.seed(123123123)
    spheres = (b-a)*np.random.rand(n_,3) + a

    rbins = np.linspace(0.,lbox,jkbins+1)
    P0_jk = np.zeros((jkbins,jkbins,jkbins))
    N_mean_jk = np.zeros((jkbins,jkbins,jkbins))
    xi_mean_jk = np.zeros((jkbins,jkbins,jkbins))
    chi_jk = np.zeros((jkbins,jkbins,jkbins))
    NXi_jk = np.zeros((jkbins,jkbins,jkbins))

    for k in range(jkbins):
        mask_z2 = (spheres[:,0] < rbins[k+1])
        mask_z1 = (spheres[:,0] > rbins[k])        
        mask_z = np.logical_and(mask_z1,mask_z2)

        for j in range(jkbins):
            mask_x2 = (spheres[:,0] < rbins[j+1])
            mask_x1 = (spheres[:,0] > rbins[j])
            mask_x = np.logical_and(mask_x1,mask_x2)

            for i in range(jkbins):
                
                mask_y2 = (spheres[:,1] < rbins[i+1])
                mask_y1 = (spheres[:,1] > rbins[i])
                mask_y = np.logical_and(mask_y1,mask_y2)

                mask_xy = np.logical_and(mask_x,mask_y)
                mask_xyz = np.logical_and(mask_xy,mask_z)

                mask = np.invert(mask_xyz)
                sph = spheres[mask,:]
                sph = sph[:n]

                # ngal = np.zeros(n)
                # sphtree = spatial.cKDTree(sph)
                # idx = sphtree.query_ball_tree(tree,r)
                # for ii in range(n):
                #     ngal[ii] = len(idx[ii])

                #for ii in range(n):
                #    ngal[ii] = len(tree.query_ball_point(sph[ii],r))

                ngal = tree.query_ball_point(sph,r,return_length=True)

                #VPF
                P0_jk[i,j,k] = len(np.where(ngal==0)[0])/n

                N_mean_jk[i,j,k] = np.mean(ngal)

                #xi_mean
                xi_mean_jk[i,j,k] = (np.mean((ngal-N_mean_jk[i,j,k])**2)-N_mean_jk[i,j,k])/N_mean_jk[i,j,k]**2
                
                chi_jk[i,j,k] = -np.log(P0_jk[i,j,k])/N_mean_jk[i,j,k]
                
                NXi_jk[i,j,k] = N_mean_jk[i,j,k]*xi_mean_jk[i,j,k]
    
    P0 = np.mean(P0_jk.flat)
    P0_std = np.std(P0_jk.flat,ddof=1)

    N_mean = np.mean(N_mean_jk.flat)
    N_mean_std = np.std(N_mean_jk.flat,ddof=1)

    xi_mean = np.mean(xi_mean_jk.flat)
    xi_mean_std = np.std(xi_mean_jk.flat,ddof=1)

    # chi = np.mean(chi_jk.flat)
    # chi_std = np.std(chi_jk.flat,ddof=1)
    chi = np.ma.masked_invalid(chi_jk.flat).mean()
    chi_std = np.ma.masked_invalid(chi_jk.flat).std(ddof=1)

    NXi = np.mean(NXi_jk.flat)
    NXi_std = np.std(NXi_jk.flat,ddof=1)

    del ngal, P0_jk, N_mean_jk, xi_mean_jk, chi_jk, NXi_jk
    
    return chi, NXi, P0, N_mean, xi_mean, \
        chi_std, NXi_std, P0_std, N_mean_std, xi_mean_std 

#%%
def perrep(gxs,lbox,overhead):
    """
    PERiodic REPlication of box

    Args:
        gxs (ascii Table): Ascii Table with galaxy data (only uses positions)
        lbox (float): Side length of simulation box
        overhead (float): Side length to replicate 

    Returns:
        ascii Table: Table with the replicated galaxies

    """

    import numpy as np
    from astropy.table import vstack
    
    #"""
    #Single axes
    #"""
    newgxs1_x = gxs[gxs['x']<overhead]
    newgxs1_x['x'] += lbox

    newgxs1_y = gxs[gxs['y']<overhead]
    newgxs1_y['y'] += lbox

    newgxs1_z = gxs[gxs['z']<overhead]
    newgxs1_z['z'] += lbox

    newgxs2_x = gxs[gxs['x']>lbox-overhead]
    newgxs2_x['x'] -= lbox

    newgxs2_y = gxs[gxs['y']>lbox-overhead]
    newgxs2_y['y'] -= lbox

    newgxs2_z = gxs[gxs['z']>lbox-overhead]
    newgxs2_z['z'] -= lbox

    #"""
    #XY
    #"""
    newgxs1_xy = gxs[np.logical_and(gxs['x']<overhead,gxs['y']<overhead)]
    newgxs1_xy['x'] += lbox
    newgxs1_xy['y'] += lbox

    newgxs2_xy = gxs[np.logical_and(gxs['x']>lbox-overhead,gxs['y']>lbox-overhead)]
    newgxs2_xy['x'] -= lbox
    newgxs2_xy['y'] -= lbox

    newgxs3_xy = gxs[np.logical_and(gxs['x']<overhead,gxs['y']>lbox-overhead)]
    newgxs3_xy['x'] += lbox
    newgxs3_xy['y'] -= lbox

    newgxs4_xy = gxs[np.logical_and(gxs['x']>lbox-overhead,gxs['y']<overhead)]
    newgxs4_xy['x'] -= lbox
    newgxs4_xy['y'] += lbox

    #"""
    #XZ
    #"""
    newgxs1_xz = gxs[np.logical_and(gxs['x']<overhead,gxs['z']<overhead)]
    newgxs1_xz['x'] += lbox
    newgxs1_xz['z'] += lbox

    newgxs2_xz = gxs[np.logical_and(gxs['x']>lbox-overhead,gxs['z']>lbox-overhead)]
    newgxs2_xz['x'] -= lbox
    newgxs2_xz['z'] -= lbox

    newgxs3_xz = gxs[np.logical_and(gxs['x']<overhead,gxs['z']>lbox-overhead)]
    newgxs3_xz['x'] += lbox
    newgxs3_xz['z'] -= lbox

    newgxs4_xz = gxs[np.logical_and(gxs['x']>lbox-overhead,gxs['z']<overhead)]
    newgxs4_xz['x'] -= lbox
    newgxs4_xz['z'] += lbox

    #"""
    #YZ
    #"""
    newgxs1_yz = gxs[np.logical_and(gxs['y']<overhead,gxs['z']<overhead)]
    newgxs1_yz['y'] += lbox
    newgxs1_yz['z'] += lbox

    newgxs2_yz = gxs[np.logical_and(gxs['y']>lbox-overhead,gxs['z']>lbox-overhead)]
    newgxs2_yz['y'] -= lbox
    newgxs2_yz['z'] -= lbox

    newgxs3_yz = gxs[np.logical_and(gxs['y']<overhead,gxs['z']>lbox-overhead)]
    newgxs3_yz['y'] += lbox
    newgxs3_yz['z'] -= lbox

    newgxs4_yz = gxs[np.logical_and(gxs['y']>lbox-overhead,gxs['z']<overhead)]
    newgxs4_yz['y'] -= lbox
    newgxs4_yz['z'] += lbox

    newgxs = vstack([gxs,newgxs1_x,newgxs1_y,newgxs1_z,\
        newgxs2_x,newgxs2_y,newgxs2_z,\
            newgxs1_xy,newgxs2_xy,newgxs3_xy,newgxs4_xy,\
                newgxs1_xz,newgxs2_xz,newgxs3_xz,newgxs4_xz,\
                    newgxs1_yz,newgxs2_yz,newgxs3_yz,newgxs4_yz])

    return newgxs
#%%
def perrep_array(gxs,lbox,overhead):
    """
    PERiodic REPlication of box

    Args:
        gxs (ascii Table): Ascii Table with galaxy data (only uses positions)
        lbox (float): Side length of simulation box
        overhead (float): Side length to replicate 

    Returns:
        ascii Table: Table with the replicated galaxies

    """

    import numpy as np
    from astropy.table import vstack
    
    #"""
    #Single axes
    #"""
    newgxs1_x = gxs[gxs[:,0]<overhead]
    newgxs1_x[:,0] += lbox

    newgxs1_y = gxs[gxs[:,1]<overhead]
    newgxs1_y[:,1] += lbox

    newgxs1_z = gxs[gxs[:,2]<overhead]
    newgxs1_z[:,2] += lbox

    newgxs2_x = gxs[gxs[:,0]>lbox-overhead]
    newgxs2_x[:,0] -= lbox

    newgxs2_y = gxs[gxs[:,1]>lbox-overhead]
    newgxs2_y[:,1] -= lbox

    newgxs2_z = gxs[gxs[:,2]>lbox-overhead]
    newgxs2_z[:,2] -= lbox

    #"""
    #XY
    #"""
    newgxs1_xy = gxs[np.logical_and(gxs[:,0]<overhead,gxs[:,1]<overhead)]
    newgxs1_xy[:,0] += lbox
    newgxs1_xy[:,1] += lbox

    newgxs2_xy = gxs[np.logical_and(gxs[:,0]>lbox-overhead,gxs[:,1]>lbox-overhead)]
    newgxs2_xy[:,0] -= lbox
    newgxs2_xy[:,1] -= lbox

    newgxs3_xy = gxs[np.logical_and(gxs[:,0]<overhead,gxs[:,1]>lbox-overhead)]
    newgxs3_xy[:,0] += lbox
    newgxs3_xy[:,1] -= lbox

    newgxs4_xy = gxs[np.logical_and(gxs[:,0]>lbox-overhead,gxs[:,1]<overhead)]
    newgxs4_xy[:,0] -= lbox
    newgxs4_xy[:,1] += lbox

    #"""
    #XZ
    #"""
    newgxs1_xz = gxs[np.logical_and(gxs[:,0]<overhead,gxs[:,2]<overhead)]
    newgxs1_xz[:,0] += lbox
    newgxs1_xz[:,2] += lbox

    newgxs2_xz = gxs[np.logical_and(gxs[:,0]>lbox-overhead,gxs[:,2]>lbox-overhead)]
    newgxs2_xz[:,0] -= lbox
    newgxs2_xz[:,2] -= lbox

    newgxs3_xz = gxs[np.logical_and(gxs[:,0]<overhead,gxs[:,2]>lbox-overhead)]
    newgxs3_xz[:,0] += lbox
    newgxs3_xz[:,2] -= lbox

    newgxs4_xz = gxs[np.logical_and(gxs[:,0]>lbox-overhead,gxs[:,2]<overhead)]
    newgxs4_xz[:,0] -= lbox
    newgxs4_xz[:,2] += lbox

    #"""
    #YZ
    #"""
    newgxs1_yz = gxs[np.logical_and(gxs[:,1]<overhead,gxs[:,2]<overhead)]
    newgxs1_yz[:,1] += lbox
    newgxs1_yz[:,2] += lbox

    newgxs2_yz = gxs[np.logical_and(gxs[:,1]>lbox-overhead,gxs[:,2]>lbox-overhead)]
    newgxs2_yz[:,1] -= lbox
    newgxs2_yz[:,2] -= lbox

    newgxs3_yz = gxs[np.logical_and(gxs[:,1]<overhead,gxs[:,2]>lbox-overhead)]
    newgxs3_yz[:,1] += lbox
    newgxs3_yz[:,2] -= lbox

    newgxs4_yz = gxs[np.logical_and(gxs[:,1]>lbox-overhead,gxs[:,2]<overhead)]
    newgxs4_yz[:,1] -= lbox
    newgxs4_yz[:,2] += lbox

    newgxs = np.vstack([gxs,newgxs1_x,newgxs1_y,newgxs1_z,\
        newgxs2_x,newgxs2_y,newgxs2_z,\
            newgxs1_xy,newgxs2_xy,newgxs3_xy,newgxs4_xy,\
                newgxs1_xz,newgxs2_xz,newgxs3_xz,newgxs4_xz,\
                    newgxs1_yz,newgxs2_yz,newgxs3_yz,newgxs4_yz])

    return newgxs


#%%
def uniform_sphereSampling(n,xv,yv,zv,R):
    """
    Sample spherical volume with uniform distribution of points

    Args:
        n (int): number of uniform points
        xv (float): x-coordinate of center of sphere
        yv (float): y-coordinate of center of sphere
        zv (float): z-coordinate of center of sphere
        R (float): radius of sphere

    Returns:
        float: x-coordinate of points
        float: y-coordinate of points
        float: z-coordinate of points
    """

    import numpy as np

    phi = np.random.uniform(0,2*np.pi,n)
    costheta = np.random.uniform(-1,1,n)
    u = np.random.uniform(0,1,n)

    theta = np.arccos(costheta)
    r = R*u**(1/3)

    x = r * np.sin(theta) * np.cos(phi) + xv
    y = r * np.sin(theta) * np.sin(phi) + yv
    z = r * np.cos(theta) + zv

    return np.column_stack([x,y,z])

#%%
def cic_stats_invoid(voids, tree, n, r_sph):
    """Returns Counts in Cells statistics

    Args:
        voids (ascii Table): void file
        tree (ckdtree): coordinates
        voids (numpy array): voids data
        n (int): Num of spheres
        r_sph (float): Radius of the spheres
        # seed (int, optional): Random seed. Defaults to 0.
        voidsfile (string): voids file location
        minradV (float): minimum void radius

    Returns:
        float: VPF
        float: Mean number of points in spheres of radius r
        float: Averaged 2pcf (variance of counts in cells)
    """
    import numpy as np
    #from scipy import spatial
    #from astropy.io import ascii

    #This section has been cut to the rvpf_jk code for efficiency
    #-----------------------------------
    # voids = ascii.read(voidsfile,\
    #     names=['r','x','y','z','vx','vy','vz',\
    #         'deltaint_1r','maxdeltaint_2-3r','log10Poisson','Nrecenter'])

    # voids = voids[voids['r']>=minradV]

    # voids['r'] = voids['r']*1000
    # voids['x'] = voids['x']*1000
    # voids['y'] = voids['y']*1000
    # voids['z'] = voids['z']*1000
    #------------------------------------

    n_invoid = round(n/len(voids)) #n_invoid is now num of spheres in each voids

    chi = np.zeros(len(voids))
    NXi = np.zeros(len(voids))

    P0 = np.zeros(len(voids))
    N_mean = np.zeros(len(voids))
    xi_mean = np.zeros(len(voids))

    for nv in range(len(voids)):
        #print(voids[nv]['r']-r_sph)
        spheres = uniform_sphereSampling(n_invoid,\
            voids[nv]['x'],voids[nv]['y'],voids[nv]['z'],voids[nv]['r'])

        ngal = np.zeros(n_invoid)
        for k in range(n_invoid):
            ngal[k] = len(tree.query_ball_point(spheres[k],r_sph))


        P0[nv] = len(np.where(ngal==0)[0])/n_invoid
        N_mean[nv] = np.mean(ngal)
        xi_mean[nv] = (np.mean((ngal-N_mean[nv])**2)-N_mean[nv])/N_mean[nv]**2

        chi[nv] = -np.log(P0[nv])/N_mean[nv]
        NXi[nv] = N_mean[nv]*xi_mean[nv]
    
    del ngal
    
    return np.mean(chi), np.mean(NXi), \
        np.mean(P0), np.mean(N_mean), np.mean(xi_mean)

#%%

def cic_stats_invoid_jk(voids, tree, n, r_sph):
    """Returns Counts in Cells statistics

    Args:
        voids (ascii Table): voidfile
        tree (ckdtree): coordinates
        voids (numpy array): voids data
        n (int): Num of spheres
        r_sph (float): Radius of the spheres
        # seed (int, optional): Random seed. Defaults to 0.
        # voidsfile (string): voids file location
        # minradV (float): minimum void radius

    Returns:
        float: VPF
        float: Mean number of points in spheres of radius r
        float: Averaged 2pcf (variance of counts in cells)
    """
    import numpy as np
    #from scipy import spatial
    #from astropy.io import ascii

    #This section has been cut to the rvpf_jk code for efficiency
    #-----------------------------------
    # voids = ascii.read(voidsfile,\
    #     names=['r','x','y','z','vx','vy','vz',\
    #         'deltaint_1r','maxdeltaint_2-3r','log10Poisson','Nrecenter'])

    # voids = voids[voids['r']>=minradV]

    # voids['r'] = voids['r']*1000
    # voids['x'] = voids['x']*1000
    # voids['y'] = voids['y']*1000
    # voids['z'] = voids['z']*1000
    #------------------------------------


    # Quiero 27 remuestreos JK (para que sea igual que el calculo de la VPF en el box,
    # donde tengo 27 remuestreos porque saco cubos de un tercio del largo de cada eje
    # del box (3**3=27) )
    jk = 27

    chi_jk = np.zeros(jk)
    NXi_jk = np.zeros(jk)

    P0_jk = np.zeros(jk)
    N_mean_jk = np.zeros(jk)
    xi_mean_jk = np.zeros(jk)

    # Voy saltando 'step' cantidad de voids
    step = int(len(voids)/jk)
    for i in range(jk):

        mask = np.ones(len(voids),bool)
        mask[step*i:step*i+step] = 0
        jkvoids = voids[mask]

        chi_nv = np.zeros(len(jkvoids))
        NXi_nv = np.zeros(len(jkvoids))

        P0_nv = np.zeros(len(jkvoids))
        N_mean_nv = np.zeros(len(jkvoids))
        xi_mean_nv = np.zeros(len(jkvoids))

        #commented because i define it previously in rvpf_jk.py 
        n_invoid = round(n/len(jkvoids)) #n_invoid is num of spheres in each void

        for nv in range(len(jkvoids)):
            spheres = uniform_sphereSampling(n_invoid,\
                jkvoids[nv]['x'],jkvoids[nv]['y'],jkvoids[nv]['z'],jkvoids[nv]['r'])

            ngal = np.zeros(n_invoid)
            for k in range(n_invoid):
                ngal[k] = len(tree.query_ball_point(spheres[k],r_sph))


            P0_nv[nv] = len(np.where(ngal==0)[0])/n_invoid
            N_mean_nv[nv] = np.mean(ngal)
            xi_mean_nv[nv] = (np.mean((ngal-N_mean_nv[nv])**2)-N_mean_nv[nv])\
                /N_mean_nv[nv]**2

            chi_nv[nv] = -np.log(P0_nv[nv])/N_mean_nv[nv]
            #if r_sph==2254.5: print(chi_nv[nv],P0_nv[nv],N_mean_nv[nv])
            NXi_nv[nv] = N_mean_nv[nv]*xi_mean_nv[nv]

        # chi_jk[i] = np.mean(chi_nv)
        # NXi_jk[i] = np.mean(NXi_nv)
        # P0_jk[i] = np.mean(P0_nv)
        # N_mean_jk[i] = np.mean(N_mean_nv)
        # xi_mean_jk[i] = np.mean(xi_mean_nv)

        chi_jk[i] = np.ma.masked_invalid(chi_nv).mean()
        NXi_jk[i] = np.ma.masked_invalid(NXi_nv).mean()
        P0_jk[i] = np.ma.masked_invalid(P0_nv).mean()
        N_mean_jk[i] = np.ma.masked_invalid(N_mean_nv).mean()
        xi_mean_jk[i] = np.ma.masked_invalid(xi_mean_nv).mean()

    
    del ngal
    
    #print(chi_jk)
    chi = np.mean(chi_jk.flat)
    #chi = np.ma.masked_invalid(chi_jk).mean()
    #print(chi)
    NXi = np.mean(NXi_jk.flat)
    P0 = np.mean(P0_jk.flat)
    N_mean = np.mean(N_mean_jk.flat)
    xi_mean = np.mean(xi_mean_jk.flat)
    # print('chi:',chi_jk)
    # print('NXi:',NXi_jk)
    # print('P0:',P0_jk)
    # print('N_mean:', N_mean_jk)
    # print('xi_mean:', xi_mean_jk)

    chi_std = np.std(chi_jk.flat,ddof=1)
    NXi_std = np.std(NXi_jk.flat,ddof=1)
    P0_std = np.std(P0_jk.flat,ddof=1)
    N_mean_std = np.std(N_mean_jk.flat,ddof=1)
    xi_mean_std = np.std(xi_mean_jk.flat,ddof=1)

    return chi, NXi, P0, N_mean, xi_mean, \
        chi_std, NXi_std, P0_std, N_mean_std, xi_mean_std 

def delta_P0(P0,Nran):
    """Calculates error for P0 as derived in Colombi et al 1995
    Args:

        P0(numpy array): value(s) of P0
        Nran(numpy array): number of volume samples in the data

    Returns:
        aray: uncertainty of P0
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

def delta_NXi(NXi,N_mean,N_mean_std,xi_mean,xi_mean_std):
    """Calculates error for NXi by propagating JK errors of N_mean and xi_mean
    Args:
        NXi(numpy array) = N_mean multiplied by xi_mean
        N_mean(numpy array): mean number of objects in volume(r)
        N_mean_std(numpy array): uncertainty of N_mean calculated with JK resampling
        xi_mean(numpy array): mean variance of objects in volume(r)
        xi_mean_std(numpy array): uncertainty of xi_mean calculated with JK resampling

    Returns:
        array: uncertainty of NXi
    """
    import numpy as np
    return NXi * np.sqrt((N_mean_std/N_mean)**2+(xi_mean_std/xi_mean)**2)

#%%
def cic_stats_jk_try(tree, n, r, lbox, jkbins):
    """
    Returns Counts in Cells statistics with Jackknife resampling

    Args:
        tree (ckdtree): coordinates
        n (int): Num of spheres
        r (float): Radius of the spheres
        seed (int, optional): Random seed. Defaults to 0.
        jkbins (int): Num. of divisions per axis for JK resampling

    Returns:
        Jackknife Mean and Standard Dev.:
        float: Reduced VPF
        float: Scaling Variable (<N>*<Xi>)
        float: VPF
        float: Mean number of points in spheres of radius r (<N>)
        float: Averaged 2pcf (variance of counts in cells, <Xi>)

    """
    import numpy as np
    from scipy import spatial
    
    a = 0
    b = lbox
    n_ = int(n*2) # I oversample the space and then choose the first n spheres after applying JK mask

    np.random.seed(123123123)
    spheres = (b-a)*np.random.rand(n_,3) + a

    rbins = np.linspace(0.,lbox,jkbins+1)
    P0_jk = np.zeros(jkbins)
    N_mean_jk = np.zeros(jkbins)
    xi_mean_jk = np.zeros(jkbins)
    chi_jk = np.zeros(jkbins)
    NXi_jk = np.zeros(jkbins)

    for i in range(jkbins):
        mask2 = (spheres[:,0] < rbins[i+1])
        mask1 = (spheres[:,0] > rbins[i])        
        mask_ = np.logical_and(mask1,mask2)

        mask = np.invert(mask_)
        sph = spheres[mask,:]
        sph = sph[:n]

        # ngal = np.zeros(n)
        # sphtree = spatial.cKDTree(sph)
        # idx = sphtree.query_ball_tree(tree,r)
        # for ii in range(n):
        #     ngal[ii] = len(idx[ii])

        #for ii in range(n):
        #    ngal[ii] = len(tree.query_ball_point(sph[ii],r))

        ngal = tree.query_ball_point(sph,r,return_length=True)

        #VPF
        P0_jk[i] = len(np.where(ngal==0)[0])/n

        N_mean_jk[i] = np.mean(ngal)

        #xi_mean
        xi_mean_jk[i] = (np.mean((ngal-N_mean_jk[i])**2)-N_mean_jk[i])/N_mean_jk[i]**2
        
        chi_jk[i] = -np.log(P0_jk[i])/N_mean_jk[i]
        
        NXi_jk[i] = N_mean_jk[i]*xi_mean_jk[i]
    
    P0 = np.mean(P0_jk.flat)
    P0_std = np.std(P0_jk.flat,ddof=1)

    N_mean = np.mean(N_mean_jk.flat)
    N_mean_std = np.std(N_mean_jk.flat,ddof=1)

    xi_mean = np.mean(xi_mean_jk.flat)
    xi_mean_std = np.std(xi_mean_jk.flat,ddof=1)

    # chi = np.mean(chi_jk.flat)
    # chi_std = np.std(chi_jk.flat,ddof=1)
    chi = np.ma.masked_invalid(chi_jk.flat).mean()
    chi_std = np.ma.masked_invalid(chi_jk.flat).std(ddof=1)

    NXi = np.mean(NXi_jk.flat)
    NXi_std = np.std(NXi_jk.flat,ddof=1)

    del ngal, P0_jk, N_mean_jk, xi_mean_jk, chi_jk, NXi_jk
    
    return chi, NXi, P0, N_mean, xi_mean, \
        chi_std, NXi_std, P0_std, N_mean_std, xi_mean_std 